import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime
import sys
import signal
import warnings
import logging
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast, GradScaler

# Disable MKLDNN RNN kernels on CPU to prevent primitive descriptor failures.
if not torch.cuda.is_available():
    mkldnn_backend = getattr(torch.backends, "mkldnn", None)
    if mkldnn_backend is not None and getattr(mkldnn_backend, "is_available", lambda: False)():
        mkldnn_backend.enabled = False
        warnings.warn(
            "MKLDNN backend disabled for CPU training to avoid LSTM primitive descriptor failures."
        )
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader as TorchLoader
from torch_geometric.utils import subgraph, k_hop_subgraph
import wntr
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from typing import Optional, Sequence, List, Tuple, Dict
from dataclasses import dataclass

try:  # pylint: disable=ungrouped-imports
    from .reproducibility import configure_seeds, save_config
except ImportError:  # pragma: no cover
    from reproducibility import configure_seeds, save_config
import networkx as nx
from tqdm.auto import tqdm

try:  # TensorBoard is optional during tests
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - missing optional dependency
    SummaryWriter = None

logger = logging.getLogger(__name__)

# Ensure the repository root is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.losses import (
    weighted_mtl_loss,
    compute_mass_balance_loss,
    pressure_headloss_consistency_loss,
    scale_physics_losses,
)
from models.loss_utils import pump_curve_loss
from models.gnn_surrogate import (
    HydroConv,
    EnhancedGNNEncoder,
    GCNEncoder,
    RecurrentGNNSurrogate,
    MultiTaskGNNSurrogate,
)

from scripts.metrics import export_table

try:
    from .feature_utils import (
        compute_norm_stats,
        apply_normalization,
        SequenceDataset,
        compute_sequence_norm_stats,
        apply_sequence_normalization,
        sanitize_edge_attr_stats,
        build_edge_attr,
        build_pump_coeffs,
        build_edge_type,
        build_edge_pairs,
        build_node_type,
    )
except ImportError:  # pragma: no cover
    from feature_utils import (
        compute_norm_stats,
        apply_normalization,
        SequenceDataset,
        compute_sequence_norm_stats,
        apply_sequence_normalization,
        sanitize_edge_attr_stats,
        build_edge_attr,
        build_pump_coeffs,
        build_edge_type,
        build_edge_pairs,
        build_node_type,
    )

PUMP_LOSS_WARN_THRESHOLD = 1.0

FLOW_CONVERSION_TO_M3S = {
    "CFS": 0.028316846592,
    "GPM": 0.0000630901964,
    "MGD": 0.0438126364,
    "MLD": 0.011574074074074073,
    "MLPD": 0.011574074074074073,
    "CMD": 1.1574074074074073e-05,
    "M3D": 1.1574074074074073e-05,
    "CMH": 0.0002777777777777778,
    "CMS": 1.0,
    "LPS": 0.001,
    "LPM": 1.6666666666666667e-05,
    "MLPS": 1.0,
}


def _forward_with_auto_checkpoint(model: nn.Module, fn):
    """Execute ``fn`` and enable gradient checkpointing on CUDA OOM.

    Parameters
    ----------
    model:
        Model which may support the ``use_checkpoint`` flag.
    fn:
        Zero-argument callable performing the forward pass.

    Returns
    -------
    Any
        The output of ``fn``.
    """

    try:
        return fn()
    except RuntimeError as e:  # pragma: no cover - retry path
        msg = str(e).lower()
        if "out of memory" in msg and not getattr(model, "use_checkpoint", False):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            setattr(model, "use_checkpoint", True)
            logger.warning(
                "CUDA OOM encountered - enabling gradient checkpointing and retrying"
            )
            return fn()
        raise


def summarize_target_norm_stats(y_mean, y_std, has_chlorine: bool):
    """Return scalar normalization stats for logging.

    ``y_mean`` and ``y_std`` may contain per-node statistics. This helper
    aggregates them across nodes before extracting the pressure and chlorine
    means and standard deviations."""
    if isinstance(y_mean, dict):
        node_mean = y_mean["node_outputs"]
        node_std = y_std["node_outputs"]
    else:
        node_mean = y_mean
        node_std = y_std
    if node_mean.dim() > 1:
        node_mean = node_mean.mean(dim=0)
        node_std = node_std.mean(dim=0)
    pressure = (node_mean[0].item(), node_std[0].item())
    chlorine = None
    if has_chlorine and node_mean.numel() > 1:
        chlorine = (node_mean[1].item(), node_std[1].item())
    return pressure, chlorine


def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Ensure ``value`` is a tensor on ``device`` with ``dtype``."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _pressure_norm_stats(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    node_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return pressure normalisation statistics broadcastable to predictions."""

    if not hasattr(model, "y_mean") or model.y_mean is None:
        return None, None
    if isinstance(model.y_mean, dict):
        mean_val = model.y_mean.get("node_outputs")
        std_val = model.y_std.get("node_outputs")
    else:
        mean_val = model.y_mean
        std_val = model.y_std
    if mean_val is None or std_val is None:
        return None, None

    mean = _as_tensor(mean_val, device, dtype)
    std = _as_tensor(std_val, device, dtype)

    if mean.ndim == 1:
        return mean[0], std[0]
    if mean.ndim == 2:
        press_mean = mean[..., 0]
        press_std = std[..., 0]
        if node_mask is not None:
            if not isinstance(node_mask, torch.Tensor):
                mask = torch.as_tensor(node_mask, dtype=torch.bool, device=device)
            else:
                mask = node_mask.to(device=device, dtype=torch.bool)
            press_mean = press_mean[mask]
            press_std = press_std[mask]
        return press_mean, press_std

    scalar_mean = mean.reshape(-1)[0]
    scalar_std = std.reshape(-1)[0]
    return scalar_mean, scalar_std


def _denormalize_pressures(
    values: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply ``mean`` and ``std`` statistics to ``values`` if available."""

    if mean is None or std is None:
        return values
    if mean.dim() == 0:
        return values * std + mean
    view_shape = [1] * (values.dim() - 1) + [-1]
    return values * std.view(*view_shape) + mean.view(*view_shape)


def _resolve_elevation_index(
    head_idx: Optional[int],
    elev_idx: Optional[int],
    has_chlorine: Optional[bool],
) -> Optional[int]:
    """Return the column index holding elevation features.

    Prefers ``head_idx + 1`` so elevation follows head in the layout.
    Falls back to ``elev_idx`` and finally the historical offsets based on
    chlorine availability when explicit indices are not provided.
    """

    if head_idx is not None:
        return head_idx + 1
    if elev_idx is not None:
        return elev_idx
    if has_chlorine is not None:
        return 3 if has_chlorine else 2
    return None


def _compute_head_loss_from_preds(
    node_outputs: torch.Tensor,
    edge_outputs: torch.Tensor,
    X_seq: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    edge_index: torch.Tensor,
    edge_attr_phys: torch.Tensor,
    edge_type: Optional[torch.Tensor],
    node_type: Optional[torch.Tensor],
    *,
    head_sign_weight: float,
    use_head: bool,
    head_idx: Optional[int],
    elev_idx: Optional[int],
    has_chlorine: Optional[bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return head loss and violation using model predictions."""

    press = node_outputs[..., 0].float()
    flow = edge_outputs.squeeze(-1)

    if hasattr(model, "y_mean") and model.y_mean is not None:
        if isinstance(model.y_mean, dict):
            p_mean = model.y_mean["node_outputs"].to(device)
            p_std = model.y_std["node_outputs"].to(device)
            if p_mean.ndim == 2:
                p_mean = p_mean[..., 0]
                p_std = p_std[..., 0]
            press = press * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
            q_mean = model.y_mean["edge_outputs"].to(device)
            q_std = model.y_std["edge_outputs"].to(device)
            flow = flow * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
        else:
            mean, std = _pressure_norm_stats(
                model,
                device,
                press.dtype,
            )
            press = _denormalize_pressures(press, mean, std)

    elevation = None
    use_head_local = use_head
    elevation_col = _resolve_elevation_index(head_idx, elev_idx, has_chlorine)
    if use_head and elevation_col is not None and X_seq.size(-1) > elevation_col:
        elevation = X_seq[..., elevation_col].float()
        if hasattr(model, "x_mean") and model.x_mean is not None:
            x_mean = model.x_mean.to(device)
            x_std = model.x_std.to(device)
            if x_mean.ndim == 1:
                elevation = elevation * x_std[elevation_col] + x_mean[elevation_col]
            else:
                elevation = (
                    elevation * x_std[:, elevation_col].view(1, 1, -1)
                    + x_mean[:, elevation_col].view(1, 1, -1)
                )
    else:
        use_head_local = False

    return pressure_headloss_consistency_loss(
        press,
        flow,
        edge_index,
        edge_attr_phys,
        elevation=elevation,
        edge_type=edge_type,
        node_type=node_type,
        return_violation=True,
        sign_weight=head_sign_weight,
        use_head=use_head_local,
    )


def _denormalize_feature(
    values: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    feature_idx: int,
    *,
    node_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Undo normalisation for ``feature_idx`` on ``values`` if stats exist."""

    if mean is None or std is None:
        return values

    if not isinstance(mean, torch.Tensor):
        mean_t = torch.as_tensor(mean, device=values.device, dtype=values.dtype)
    else:
        mean_t = mean.to(device=values.device, dtype=values.dtype)

    if not isinstance(std, torch.Tensor):
        std_t = torch.as_tensor(std, device=values.device, dtype=values.dtype)
    else:
        std_t = std.to(device=values.device, dtype=values.dtype)

    if mean_t.ndim == 0 or mean_t.numel() == 1:
        mean_sel = mean_t.reshape(())
        std_sel = std_t.reshape(())
    elif mean_t.ndim == 1:
        if feature_idx >= mean_t.shape[0]:
            raise IndexError(f"feature_idx {feature_idx} out of bounds for mean of shape {mean_t.shape}")
        mean_sel = mean_t[feature_idx]
        std_sel = std_t[feature_idx]
    else:
        mean_sel = mean_t.select(-1, feature_idx)
        std_sel = std_t.select(-1, feature_idx)
        if node_indices is not None and mean_sel.dim() >= 1:
            idx = node_indices.to(device=values.device, dtype=torch.long)
            if mean_sel.size(0) < idx.numel():
                raise IndexError(
                    "Normalization statistics do not cover all requested node indices"
                )
            mean_sel = mean_sel.index_select(0, idx)
            std_sel = std_sel.index_select(0, idx)

    while mean_sel.dim() < values.dim():
        mean_sel = mean_sel.unsqueeze(0)
        std_sel = std_sel.unsqueeze(0)

    return values * std_sel + mean_sel


def ramp_weight(target: float, epoch: int, anneal_epochs: int) -> float:
    """Linearly ramp a weight from zero to ``target``.

    Parameters
    ----------
    target:
        Final desired weight value.
    epoch:
        Current training epoch (0-indexed).
    anneal_epochs:
        Number of epochs over which to increase the weight.

    Returns
    -------
    float
        The scaled weight for the current epoch.
    """

    if anneal_epochs <= 0:
        return target
    factor = min(1.0, float(epoch + 1) / anneal_epochs)
    return target * factor


def load_dataset(
    x_path: str,
    y_path: str,
    edge_index_path: str = "edge_index.npy",
    edge_attr: Optional[np.ndarray] = None,
    node_type: Optional[np.ndarray] = None,
    edge_type: Optional[np.ndarray] = None,
    node_names_path: Optional[str] = None,
    expected_node_names: Optional[Sequence[str]] = None,
) -> List[Data]:
    """Load training data.

    The function supports two dataset layouts:

    1. **Dictionary format** – each element of ``X`` is a dictionary containing
       ``edge_index`` and ``node_features`` arrays.
    2. **Matrix format** – ``X`` is an array of node feature matrices while a
       shared ``edge_index`` array is stored separately.

    If ``expected_node_names`` is provided, the function will attempt to load a
    reference node ordering (saved during data generation) from
    ``node_names_path`` or ``"node_names.npy"`` in the same directory as the
    feature matrix.  The node count and ordering of the loaded dataset must
    match this reference list, otherwise a ``ValueError`` is raised.
    """

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    data_list: List[Data] = []
    edge_attr_tensor = None
    node_type_tensor = None
    edge_type_tensor = None
    if edge_attr is not None:
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    if node_type is not None:
        node_type_tensor = torch.tensor(node_type, dtype=torch.long)
    if edge_type is not None:
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

    # Verify node ordering against the reference list saved during
    # data generation.  ``expected_node_names`` typically comes from
    # ``wn.node_name_list``.  The reference ordering is stored in a
    # separate file (``node_names.npy`` by default) within the dataset
    # directory.  The check is skipped if either argument is ``None``
    # or the reference file is missing.
    node_names_ref: Optional[List[str]] = None
    if expected_node_names is not None:
        if node_names_path is None:
            node_names_path = str(Path(x_path).with_name("node_names.npy"))
        if os.path.exists(node_names_path):
            node_names_ref = np.load(node_names_path, allow_pickle=True).tolist()
            if list(expected_node_names) != list(node_names_ref):
                raise ValueError("Node name ordering mismatch between dataset and reference list")
        else:  # pragma: no cover - reference file missing
            logger.warning("Node names file %s not found; skipping order check", node_names_path)
    ref_count = len(node_names_ref) if node_names_ref is not None else None

    # Detect whether the first entry is a dictionary (object array) or a plain
    # matrix. ``np.ndarray`` with ``dtype=object`` will require calling
    # ``item()`` to obtain the underlying dictionary.
    first_elem = X[0]
    if isinstance(first_elem, dict) or (
        isinstance(first_elem, np.ndarray) and first_elem.dtype == object
    ):
        for graph_dict, label in zip(X, y):
            d = graph_dict if isinstance(graph_dict, dict) else graph_dict.item()
            edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
            node_feat = torch.tensor(d["node_features"], dtype=torch.float32)
            if torch.isnan(node_feat).any():
                # Replace NaNs introduced during data generation (e.g., missing
                # reservoir heads) with zeros so that training does not produce
                # ``NaN`` losses.
                node_feat = torch.nan_to_num(node_feat)
            if ref_count is not None and node_feat.shape[0] != ref_count:
                raise ValueError(
                    "Feature matrix node count does not match reference list length"
                )

            edge_target = None
            node_target = label
            if isinstance(label, dict):
                node_target = label.get("node_outputs")
                edge_target = label.get("edge_outputs")

            data = Data(
                x=node_feat,
                edge_index=edge_index,
                y=torch.tensor(node_target, dtype=torch.float32),
            )
            if edge_target is not None:
                data.edge_y = torch.tensor(edge_target, dtype=torch.float32).unsqueeze(-1)
            if edge_attr_tensor is not None:
                data.edge_attr = edge_attr_tensor
            if node_type_tensor is not None:
                data.node_type = node_type_tensor
            if edge_type_tensor is not None:
                data.edge_type = edge_type_tensor
            data_list.append(data)
    else:
        # ``X`` already contains node feature matrices. Load the shared
        # ``edge_index`` from ``edge_index_path``.
        edge_index = torch.tensor(np.load(edge_index_path), dtype=torch.long)
        for node_feat, label in zip(X, y):
            node_feat = torch.tensor(node_feat, dtype=torch.float32)
            if torch.isnan(node_feat).any():
                node_feat = torch.nan_to_num(node_feat)
            if ref_count is not None and node_feat.shape[0] != ref_count:
                raise ValueError(
                    "Feature matrix node count does not match reference list length"
                )
            edge_target = None
            node_target = label
            if isinstance(label, dict):
                # ``label`` may contain multiple targets (e.g., edge labels).
                node_target = label.get("node_outputs")
                edge_target = label.get("edge_outputs")
            data = Data(
                x=node_feat,
                edge_index=edge_index,
                y=torch.tensor(node_target, dtype=torch.float32),
            )
            if edge_target is not None:
                data.edge_y = torch.tensor(edge_target, dtype=torch.float32).unsqueeze(-1)
            if edge_attr_tensor is not None:
                data.edge_attr = edge_attr_tensor
            if node_type_tensor is not None:
                data.node_type = node_type_tensor
            if edge_type_tensor is not None:
                data.edge_type = edge_type_tensor
            data_list.append(data)

    return data_list


def _to_numpy(seq: Sequence[float]) -> np.ndarray:
    """Convert sequence to NumPy array."""
    return np.asarray(seq, dtype=float)


def predicted_vs_actual_scatter(
    true_pressure: Sequence[float],
    pred_pressure: Sequence[float],
    true_chlorine: Optional[Sequence[float]],
    pred_chlorine: Optional[Sequence[float]],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
    mask: Optional[Sequence[bool]] = None,
) -> Optional[plt.Figure]:
    """Scatter plots comparing surrogate predictions with EPANET results.

    ``true_chlorine`` and ``pred_chlorine`` may be ``None`` when the dataset
    excludes chlorine targets.
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    tp = _to_numpy(true_pressure)
    pp = _to_numpy(pred_pressure)
    tc = _to_numpy(true_chlorine) if true_chlorine is not None else None
    pc = _to_numpy(pred_chlorine) if pred_chlorine is not None else None

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        tp = tp[m]
        pp = pp[m]
        if tc is not None and pc is not None:
            tc = tc[m]
            pc = pc[m]

    if tc is not None and pc is not None:
        # chlorine values are stored in log space (log1p). Convert back to mg/L
        # before plotting so the axes reflect physical units.
        tc = np.expm1(tc) * 1000.0
        pc = np.expm1(pc) * 1000.0
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(tp, pp, label="Pressure", color="tab:blue", alpha=0.7)
        min_p, max_p = tp.min(), tp.max()
        axes[0].plot([min_p, max_p], [min_p, max_p], "k--", lw=1)
        axes[0].set_xlabel("Actual Pressure (m)")
        axes[0].set_ylabel("Predicted Pressure (m)")
        axes[0].set_title("Pressure")

        axes[1].scatter(tc, pc, label="Chlorine", color="tab:orange", alpha=0.7)
        min_c, max_c = tc.min(), tc.max()
        axes[1].plot([min_c, max_c], [min_c, max_c], "k--", lw=1)
        axes[1].set_xlabel("Actual Chlorine (mg/L)")
        axes[1].set_ylabel("Predicted Chlorine (mg/L)")
        axes[1].set_title("Chlorine")

        fig.suptitle("Surrogate Model Prediction Accuracy for Pressure and Chlorine")
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        ax = axes if isinstance(axes, plt.Axes) else axes[0]
        ax.scatter(tp, pp, label="Pressure", color="tab:blue", alpha=0.7)
        min_p, max_p = tp.min(), tp.max()
        ax.plot([min_p, max_p], [min_p, max_p], "k--", lw=1)
        ax.set_xlabel("Actual Pressure (m)")
        ax.set_ylabel("Predicted Pressure (m)")
        ax.set_title("Pressure")
        fig.suptitle("Surrogate Model Prediction Accuracy for Pressure")
        fig.tight_layout()

    fig.savefig(plots_dir / f"pred_vs_actual_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_residual_scatter(
    true_vals: Sequence[float],
    pred_vals: Sequence[float],
    run_name: str,
    label: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
    mask: Optional[Sequence[bool]] = None,
    density: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot residuals (prediction - truth) against true and predicted values.

    ``label`` determines axis titles and units (e.g. ``"pressure"`` or ``"chlorine"``).
    Set ``density`` to ``"kde"`` or ``"hex"`` to overlay a KDE or hexbin density
    highlighting point clusters.
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    t = _to_numpy(true_vals)
    p = _to_numpy(pred_vals)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        t = t[m]
        p = p[m]

    unit = "m"
    if label.lower().startswith("chlor"):
        t = np.expm1(t) * 1000.0
        p = np.expm1(p) * 1000.0
        unit = "mg/L"

    residual = p - t
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def _scatter(ax, x, y):
        if density == "kde":
            try:
                from scipy.stats import gaussian_kde

                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)
                sc = ax.scatter(x, y, c=z, s=10, cmap="viridis")
                fig.colorbar(sc, ax=ax)
            except Exception:
                ax.scatter(x, y, color="tab:blue", alpha=0.7, s=10)
        else:
            ax.scatter(x, y, color="tab:blue", alpha=0.7, s=10)
            if density == "hex":
                hb = ax.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=1, alpha=0.6)
                fig.colorbar(hb, ax=ax)

    _scatter(axes[0], t, residual)
    axes[0].axhline(0.0, color="k", lw=1)
    axes[0].set_xlabel(f"Actual {label.title()} ({unit})")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual vs Actual")

    _scatter(axes[1], p, residual)
    axes[1].axhline(0.0, color="k", lw=1)
    axes[1].set_xlabel(f"Predicted {label.title()} ({unit})")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual vs Predicted")

    fig.suptitle(f"{label.title()} Residuals")
    fig.tight_layout()
    fig.savefig(plots_dir / f"residual_scatter_{label}_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def compute_edge_attr_stats(edge_attr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std for edge attribute matrix."""
    attr_mean = torch.tensor(edge_attr.mean(axis=0), dtype=torch.float32)
    attr_std = torch.tensor(edge_attr.std(axis=0) + 1e-8, dtype=torch.float32)
    return attr_mean, attr_std


@dataclass
class RunningStats:
    """Accumulate error statistics without storing full arrays."""

    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    abs_pct_sum: float = 0.0
    max_err: float = 0.0
    tgt_sum: float = 0.0
    tgt_sq_sum: float = 0.0

    def update(self, pred, true) -> None:
        p = np.asarray(pred, dtype=float)
        t = np.asarray(true, dtype=float)
        diff = p - t
        abs_err = np.abs(diff)
        self.count += abs_err.size
        self.abs_sum += float(abs_err.sum())
        self.sq_sum += float(np.square(diff).sum())
        self.tgt_sum += float(t.sum())
        self.tgt_sq_sum += float(np.square(t).sum())
        denom = np.maximum(np.abs(t), 1e-8)
        self.abs_pct_sum += float((abs_err / denom).sum())
        if abs_err.size:
            self.max_err = max(self.max_err, float(abs_err.max()))

    def metrics(self) -> List[float]:
        if self.count == 0:
            return [float("nan")] * 5
        mae = self.abs_sum / self.count
        rmse = np.sqrt(self.sq_sum / self.count)
        mape = (self.abs_pct_sum / self.count) * 100.0
        ss_tot = self.tgt_sq_sum - (self.tgt_sum ** 2) / self.count
        r2 = float("nan")
        if ss_tot > 1e-8:
            r2 = 1.0 - (self.sq_sum / ss_tot)
        return [mae, rmse, mape, self.max_err, r2]


def save_scatter_plots(
    true_p,
    preds_p,
    true_c,
    preds_c,
    run_name: str,
    plots_dir: Optional[Path] = None,
    mask: Optional[Sequence[bool]] = None,
) -> None:
    """Save enhanced scatter and residual plots for surrogate predictions."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR

    fig = predicted_vs_actual_scatter(
        true_p,
        preds_p,
        true_c,
        preds_c,
        run_name,
        plots_dir=plots_dir,
        return_fig=True,
        mask=mask,
    )
    # also store individual scatter images for backward compatibility
    axes = fig.axes
    axes[0].figure.savefig(plots_dir / f"pred_vs_actual_pressure_{run_name}.png")
    if len(axes) > 1:
        axes[1].figure.savefig(
            plots_dir / f"pred_vs_actual_chlorine_{run_name}.png"
        )
    plt.close(fig)

    plot_residual_scatter(
        true_p,
        preds_p,
        run_name,
        "pressure",
        plots_dir=plots_dir,
        mask=mask,
        density="hex",
    )
    if true_c is not None and preds_c is not None:
        plot_residual_scatter(
            true_c,
            preds_c,
            run_name,
            "chlorine",
            plots_dir=plots_dir,
            mask=mask,
            density="hex",
        )


def save_accuracy_metrics(
    pressure_stats: RunningStats,
    run_name: str,
    logs_dir: Optional[Path] = None,
    chlorine_stats: Optional[RunningStats] = None,
    flow_stats: Optional[RunningStats] = None,
) -> None:
    """Compute and export accuracy metrics to a CSV file."""
    if logs_dir is None:
        logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = {"Pressure (m)": pressure_stats.metrics()}
    if chlorine_stats is not None:
        data["Chlorine (mg/L)"] = chlorine_stats.metrics()
    if flow_stats is not None:
        data["Flow (m^3/h)"] = flow_stats.metrics()

    index = [
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Percentage Error",
        "Maximum Error",
        "R^2",
    ]
    df = pd.DataFrame(data, index=index)
    export_table(df, str(logs_dir / f"accuracy_{run_name}.csv"))


def plot_error_histograms(
    err_p: Sequence[float],
    err_c: Optional[Sequence[float]],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
    mask: Optional[Sequence[bool]] = None,
) -> Optional[plt.Figure]:
    """Histogram and box plots of prediction errors."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    ep = _to_numpy(err_p)
    ec = _to_numpy(err_c) if err_c is not None else None

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        ep = ep[m]
        if ec is not None:
            ec = ec[m]

    if ec is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].hist(ep, bins=50, color="tab:blue", alpha=0.7)
        axes[0, 0].set_title("Pressure Error")
        axes[0, 0].set_xlabel("Prediction Error (m)")
        axes[0, 0].set_ylabel("Count")

        axes[0, 1].hist(ec, bins=50, color="tab:orange", alpha=0.7)
        axes[0, 1].set_title("Chlorine Error")
        axes[0, 1].set_xlabel("Prediction Error (mg/L)")
        axes[0, 1].set_ylabel("Count")

        axes[1, 0].boxplot(ep, vert=False)
        axes[1, 0].set_yticklabels(["Pressure"])
        axes[1, 0].set_xlabel("Prediction Error (m)")
        axes[1, 0].set_title("Pressure Error Box")

        axes[1, 1].boxplot(ec, vert=False)
        axes[1, 1].set_yticklabels(["Chlorine"])
        axes[1, 1].set_xlabel("Prediction Error (mg/L)")
        axes[1, 1].set_title("Chlorine Error Box")

        for ax in axes.ravel():
            ax.tick_params(labelsize=8)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(ep, bins=50, color="tab:blue", alpha=0.7)
        axes[0].set_title("Pressure Error")
        axes[0].set_xlabel("Prediction Error (m)")
        axes[0].set_ylabel("Count")

        axes[1].boxplot(ep, vert=False)
        axes[1].set_yticklabels(["Pressure"])
        axes[1].set_xlabel("Prediction Error (m)")
        axes[1].set_title("Pressure Error Box")

        for ax in axes.ravel():
            ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / f"error_histograms_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_error_heatmap(
    errors: Sequence[float],
    wn: wntr.network.WaterNetworkModel,
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
    mask: Optional[Sequence[bool]] = None,
) -> Optional[plt.Figure]:
    """Visualise per-node pressure errors on the network layout."""

    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    err = np.asarray(errors, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        err = err.copy()
        err[~m] = np.nan

    coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    xs = [coords[n][0] for n in wn.node_name_list]
    ys = [coords[n][1] for n in wn.node_name_list]

    fig, ax = plt.subplots(figsize=(6, 5))
    for link in wn.link_name_list:
        link_obj = wn.get_link(link)
        x1, y1 = coords[link_obj.start_node_name]
        x2, y2 = coords[link_obj.end_node_name]
        ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.5)
    sc = ax.scatter(xs, ys, c=err, cmap="Reds", s=35)
    plt.colorbar(sc, ax=ax, label="Pressure MAE (m)")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / f"error_heatmap_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def compute_node_pressure_mae(
    model: nn.Module,
    dataset,
    loader,
    device: torch.device,
    has_chlorine: bool,
) -> np.ndarray:
    """Return mean absolute pressure error for each node in ``dataset``."""

    if isinstance(dataset, SequenceDataset):
        num_nodes = dataset.X.shape[-2]
        ei = dataset.edge_index.to(device)
        ea = dataset.edge_attr.to(device) if dataset.edge_attr is not None else None
        nt = dataset.node_type.to(device) if dataset.node_type is not None else None
        et = dataset.edge_type.to(device) if dataset.edge_type is not None else None
    else:
        num_nodes = dataset[0].num_nodes

    err_sum = torch.zeros(num_nodes, device=device)
    count = 0

    model.eval()
    with torch.no_grad():
        if isinstance(dataset, SequenceDataset):
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    X_seq, edge_attr_batch, Y_seq = batch
                else:
                    X_seq, Y_seq = batch
                    edge_attr_batch = None
                X_seq = X_seq.to(device)
                attr = edge_attr_batch.to(device) if isinstance(edge_attr_batch, torch.Tensor) else ea

                def _model_forward():
                    return model(X_seq, ei, attr, nt, et)

                out = _forward_with_auto_checkpoint(model, _model_forward)
                node_pred = out["node_outputs"] if isinstance(out, dict) else out
                Y_node = (
                    Y_seq["node_outputs"].to(node_pred.device)
                    if isinstance(Y_seq, dict)
                    else Y_seq.to(node_pred.device)
                )
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        y_mean = model.y_mean["node_outputs"].to(node_pred.device)
                        y_std = model.y_std["node_outputs"].to(node_pred.device)
                    else:
                        y_mean = model.y_mean.to(node_pred.device)
                        y_std = model.y_std.to(node_pred.device)
                    node_pred = node_pred * y_std + y_mean
                    Y_node = Y_node * y_std + y_mean
                pred_p = node_pred[..., 0].reshape(-1, num_nodes)
                true_p = Y_node[..., 0].reshape(-1, num_nodes)
                err_sum += torch.abs(pred_p - true_p).sum(0)
                count += pred_p.shape[0]
        else:
            for batch in loader:
                batch = batch.to(device, non_blocking=True)

                def _model_forward():
                    return model(
                        batch.x,
                        batch.edge_index,
                        getattr(batch, "edge_attr", None),
                        getattr(batch, "node_type", None),
                        getattr(batch, "edge_type", None),
                    )

                out = _forward_with_auto_checkpoint(model, _model_forward)
                node_out = out["node_outputs"] if isinstance(out, dict) else out
                batch_y = batch.y
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        y_mean = model.y_mean["node_outputs"].to(node_out.device)
                        y_std = model.y_std["node_outputs"].to(node_out.device)
                    else:
                        y_mean = model.y_mean.to(node_out.device)
                        y_std = model.y_std.to(node_out.device)
                    node_out = node_out * y_std + y_mean
                    batch_y = batch_y * y_std + y_mean
                pred_p = node_out[:, 0].view(-1, num_nodes)
                true_p = batch_y[:, 0].view(-1, num_nodes)
                err_sum += torch.abs(pred_p - true_p).sum(0)
                count += pred_p.shape[0]

    return (err_sum / max(count, 1)).cpu().numpy()


def correlation_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot a heatmap of pairwise feature correlations."""

    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        mat = mat.reshape(-1, mat.shape[-1])

    corr = np.corrcoef(mat, rowvar=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Feature Correlation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots_dir / f"correlation_heatmap_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_loss_components(
    loss_components: Sequence[Sequence[float]],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot individual loss terms over training epochs."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(loss_components, dtype=float)
    epochs = np.arange(1, arr.shape[0] + 1)
    labels = ["pressure", "chlorine", "flow", "mass", "sym"]
    if arr.shape[1] > len(labels):
        labels.append("head")
    if arr.shape[1] > len(labels):
        labels.append("pump")

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(arr.shape[1]):
        ax.loglog(epochs, arr[:, i], label=labels[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component-wise Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"loss_components_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_sequence_prediction(
    model: nn.Module,
    dataset: "SequenceDataset",
    run_name: str,
    node_idx: int = 0,
    plots_dir: Optional[Path] = None,
) -> None:
    """Plot a single denormalised sequence prediction for ``node_idx``."""

    if len(dataset) == 0:
        return

    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    sample = dataset[0]
    if isinstance(sample, (list, tuple)) and len(sample) == 3:
        X_seq, edge_attr_seq, Y_seq = sample
    else:
        X_seq, Y_seq = sample
        edge_attr_seq = None
    X_seq = X_seq.unsqueeze(0).to(device)
    ei = dataset.edge_index.to(device)
    if edge_attr_seq is not None:
        ea = edge_attr_seq.unsqueeze(0).to(device)
    else:
        ea = dataset.edge_attr.to(device) if dataset.edge_attr is not None else None
    nt = dataset.node_type.to(device) if dataset.node_type is not None else None
    et = dataset.edge_type.to(device) if dataset.edge_type is not None else None

    with torch.no_grad():
        def _model_forward():
            return model(X_seq, ei, ea, nt, et)
        out = _forward_with_auto_checkpoint(model, _model_forward)

    if isinstance(out, dict):
        pred = out["node_outputs"]
    else:
        pred = out

    if isinstance(Y_seq, dict):
        true = Y_seq["node_outputs"]
    else:
        true = Y_seq

    if hasattr(model, "y_mean") and model.y_mean is not None:
        if isinstance(model.y_mean, dict):
            mean = model.y_mean["node_outputs"].to(pred.device)
            std = model.y_std["node_outputs"].to(pred.device)
        else:
            mean = model.y_mean.to(pred.device)
            std = model.y_std.to(pred.device)
        pred = pred * std + mean
        true = true.to(pred.device) * std + mean
    else:
        true = true.to(pred.device)

    pred_np = pred.squeeze(0).cpu().numpy()
    # ``true`` is returned directly from ``SequenceDataset`` without a batch
    # dimension.  When the sequence length is ``1`` this tensor has shape
    # ``[1, N, F]`` and calling ``squeeze(0)`` would drop the time axis and
    # produce a 2-D array.  Skip squeezing so the indexing logic works for
    # both single and multi-step sequences.
    true_np = true.cpu().numpy()

    T = pred_np.shape[0]
    time = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    axes[0].plot(time, true_np[:, node_idx, 0], label="Actual")
    axes[0].plot(time, pred_np[:, node_idx, 0], "--", label="Predicted")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Pressure (m)")
    axes[0].set_title(f"Node {node_idx} Pressure")
    axes[0].legend()

    if pred_np.shape[-1] >= 2:
        axes[1].plot(time, np.expm1(true_np[:, node_idx, 1]) * 1000.0, label="Actual")
        axes[1].plot(time, np.expm1(pred_np[:, node_idx, 1]) * 1000.0, "--", label="Predicted")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Chlorine (mg/L)")
        axes[1].set_title(f"Node {node_idx} Chlorine")
        axes[1].legend()
    else:
        axes[1].remove()

    fig.tight_layout()
    fig.savefig(plots_dir / f"time_series_example_{run_name}.png")
    plt.close(fig)


def _get_flow_conversion_factor(units: Optional[str]) -> float:
    """Return conversion factor from ``units`` to m^3/s."""

    if not units:
        return 1.0
    key = str(units).upper()
    return FLOW_CONVERSION_TO_M3S.get(key, 1.0)


def _estimate_pump_areas(
    wn: wntr.network.WaterNetworkModel,
) -> Dict[str, Optional[float]]:
    """Estimate cross-sectional area (m^2) for each pump using adjacent pipes."""

    areas: Dict[str, Optional[float]] = {}
    for pump_name in wn.pump_name_list:
        try:
            pump = wn.get_link(pump_name)
        except Exception:  # pragma: no cover - defensive guard
            areas[pump_name] = None
            continue
        diameter: Optional[float] = None
        for node_name in (pump.start_node_name, pump.end_node_name):
            try:
                links = wn.get_links_for_node(node_name)
            except Exception:
                links = []
            for link_name in links:
                if link_name == pump_name:
                    continue
                try:
                    link = wn.get_link(link_name)
                except Exception:
                    continue
                diam = getattr(link, "diameter", None)
                if diam is not None and diam > 0:
                    diameter = float(diam)
                    break
            if diameter is not None:
                break
        if diameter is not None and diameter > 0:
            areas[pump_name] = math.pi * (diameter ** 2) / 4.0
        else:
            areas[pump_name] = None
    return areas


def _plot_time_series(
    time: np.ndarray,
    series: "OrderedDict[str, np.ndarray]",
    title: str,
    ylabel: str,
    filename: str,
    run_name: str,
    plots_dir: Path,
) -> None:
    """Plot multiple time series on a shared axis."""

    if not series:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, values in series.items():
        ax.plot(time, values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize="small")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{filename}_{run_name}.png")
    plt.close(fig)


def generate_sequence_diagnostic_plots(
    sequence_data: Dict[str, np.ndarray],
    wn: wntr.network.WaterNetworkModel,
    run_name: str,
    plots_dir: Optional[Path] = None,
) -> None:
    """Generate tank, pump and node time series comparison plots."""

    pred_nodes = sequence_data.get("pred_nodes")
    true_nodes = sequence_data.get("true_nodes")
    pred_edges = sequence_data.get("pred_edges")
    true_edges = sequence_data.get("true_edges")

    if pred_nodes is None or true_nodes is None:
        return
    if plots_dir is None:
        plots_dir = PLOTS_DIR

    pred_nodes = np.asarray(pred_nodes)
    true_nodes = np.asarray(true_nodes)
    time = np.arange(pred_nodes.shape[0])
    node_index_map = {name: idx for idx, name in enumerate(wn.node_name_list)}

    if wn.tank_name_list:
        tank_indices = [node_index_map[name] for name in wn.tank_name_list if name in node_index_map]
        if tank_indices:
            pred_tanks = pred_nodes[:, tank_indices, 0]
            true_tanks = true_nodes[:, tank_indices, 0]
            modeled = OrderedDict(
                (name, pred_tanks[:, i]) for i, name in enumerate(wn.tank_name_list) if name in node_index_map
            )
            actual = OrderedDict(
                (name, true_tanks[:, i]) for i, name in enumerate(wn.tank_name_list) if name in node_index_map
            )
            _plot_time_series(
                time,
                modeled,
                "Modeled Tank Pressure",
                "Pressure (m)",
                "tank_pressure_modeled",
                run_name,
                plots_dir,
            )
            _plot_time_series(
                time,
                actual,
                "Actual Tank Pressure",
                "Pressure (m)",
                "tank_pressure_actual",
                run_name,
                plots_dir,
            )

    if pred_edges is not None and true_edges is not None and wn.pump_name_list:
        pred_edges = np.asarray(pred_edges)
        true_edges = np.asarray(true_edges)
        if pred_edges.ndim == 3:
            pred_edges = pred_edges.squeeze(-1)
        if true_edges.ndim == 3:
            true_edges = true_edges.squeeze(-1)
        link_index_map = {name: idx for idx, name in enumerate(wn.link_name_list)}
        pump_modeled = OrderedDict()
        pump_actual = OrderedDict()
        for pump_name in wn.pump_name_list:
            link_idx = link_index_map.get(pump_name)
            if link_idx is None:
                continue
            edge_idx = link_idx * 2
            pump_modeled[pump_name] = pred_edges[:, edge_idx]
            pump_actual[pump_name] = true_edges[:, edge_idx]
        _plot_time_series(
            time,
            pump_modeled,
            "Modeled Pump Flow",
            "Flow (network units)",
            "pump_flow_modeled",
            run_name,
            plots_dir,
        )
        _plot_time_series(
            time,
            pump_actual,
            "Actual Pump Flow",
            "Flow (network units)",
            "pump_flow_actual",
            run_name,
            plots_dir,
        )

        pump_areas = _estimate_pump_areas(wn)
        units = getattr(getattr(wn, "options", object), "hydraulic", None)
        flow_units = getattr(units, "inpfile_units", None) if units is not None else None
        conv = _get_flow_conversion_factor(flow_units)
        velocity_modeled = OrderedDict()
        velocity_actual = OrderedDict()
        for pump_name in wn.pump_name_list:
            area = pump_areas.get(pump_name)
            if area is None or area <= 0:
                continue
            modeled_flow = pump_modeled.get(pump_name)
            actual_flow = pump_actual.get(pump_name)
            if modeled_flow is None or actual_flow is None:
                continue
            velocity_modeled[pump_name] = np.asarray(modeled_flow) * conv / area
            velocity_actual[pump_name] = np.asarray(actual_flow) * conv / area
        _plot_time_series(
            time,
            velocity_modeled,
            "Modeled Pump Velocity",
            "Velocity (m/s)",
            "pump_velocity_modeled",
            run_name,
            plots_dir,
        )
        _plot_time_series(
            time,
            velocity_actual,
            "Actual Pump Velocity",
            "Velocity (m/s)",
            "pump_velocity_actual",
            run_name,
            plots_dir,
        )

    focus_nodes = ["J219", "J304", "J67", "J431", "J255", "J314", "J363"]
    available_nodes = [name for name in focus_nodes if name in node_index_map]
    if available_nodes:
        pred_focus = OrderedDict(
            (name, pred_nodes[:, node_index_map[name], 0]) for name in available_nodes
        )
        true_focus = OrderedDict(
            (name, true_nodes[:, node_index_map[name], 0]) for name in available_nodes
        )
        _plot_time_series(
            time,
            pred_focus,
            "Modeled Pressure at Key Junctions",
            "Pressure (m)",
            "focus_pressure_modeled",
            run_name,
            plots_dir,
        )
        _plot_time_series(
            time,
            true_focus,
            "Actual Pressure at Key Junctions",
            "Pressure (m)",
            "focus_pressure_actual",
            run_name,
            plots_dir,
        )



def build_loss_mask(wn: wntr.network.WaterNetworkModel) -> torch.Tensor:
    """Return boolean mask marking nodes included in the loss."""

    mask = torch.ones(len(wn.node_name_list), dtype=torch.bool)
    for i, name in enumerate(wn.node_name_list):
        if name in wn.reservoir_name_list:
            mask[i] = False
    return mask


def estimate_tank_flow_scale(
    X_seq: np.ndarray,
    Y_seq: np.ndarray,
    edge_index_np: np.ndarray,
    tank_indices: Sequence[int],
    tank_areas: Sequence[float],
    timestep_seconds: float,
) -> np.ndarray:
    """Estimate per-tank flow scaling factors for volume integration."""

    if not tank_indices:
        return np.array([], dtype=np.float32)
    scales: List[float] = []
    edge_index_np = np.asarray(edge_index_np)
    timestep = float(timestep_seconds) if timestep_seconds else 1.0
    for idx, area in zip(tank_indices, tank_areas):
        src = np.where(edge_index_np[0] == idx)[0]
        tgt = np.where(edge_index_np[1] == idx)[0]
        if src.size == 0 and tgt.size == 0:
            scales.append(1.0)
            continue
        edges = np.concatenate([src, tgt])
        signs = np.concatenate([-np.ones(src.size), np.ones(tgt.size)])
        net_flows: List[np.ndarray] = []
        delta_h: List[np.ndarray] = []
        for seq_idx in range(len(X_seq)):
            flows = Y_seq[seq_idx]["edge_outputs"][:, edges]
            net = (flows * signs).sum(axis=1) / 2.0
            net_flows.append(net)
            pressure_curr = X_seq[seq_idx][:, idx, 1]
            pressure_next = Y_seq[seq_idx]["node_outputs"][:, idx, 0]
            delta_h.append(pressure_next - pressure_curr)
        net_flow_flat = np.concatenate(net_flows)
        delta_h_flat = np.concatenate(delta_h)
        mask = np.abs(net_flow_flat) > 1e-6
        if not np.any(mask):
            scales.append(1.0)
            continue
        ratios = (delta_h_flat[mask] * area) / (net_flow_flat[mask] * timestep)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            scales.append(1.0)
        else:
            median = float(np.median(ratios))
            if not np.isfinite(median):
                median = 1.0
            scales.append(float(np.clip(median, 1e-3, 1e3)))
    return np.asarray(scales, dtype=np.float32)


interrupted = False


def _signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("Received interrupt signal. Stopping after current epoch...")


def handle_keyboard_interrupt(
    model_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    epoch: int,
    norm_stats: Optional[Dict[str, np.ndarray]] = None,
    model_meta: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save final checkpoint when training is interrupted."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }
    if norm_stats is not None:
        checkpoint["norm_stats"] = norm_stats
    if model_meta is not None:
        checkpoint["model_meta"] = model_meta
    torch.save(checkpoint, model_path)
    print(f"Training interrupted. Saved checkpoint to {model_path}")


def partition_graph_greedy(edge_index: np.ndarray, num_nodes: int, cluster_size: int) -> List[np.ndarray]:
    """Split ``edge_index`` into clusters using greedy modularity heuristics."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((int(s), int(t)) for s, t in edge_index.T)
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    clusters: List[List[int]] = []
    for com in communities:
        nodes = list(com)
        for i in range(0, len(nodes), cluster_size):
            clusters.append(nodes[i : i + cluster_size])
    assigned = {n for cl in clusters for n in cl}
    for n in range(num_nodes):
        if n not in assigned:
            smallest = min(range(len(clusters)), key=lambda i: len(clusters[i]))
            clusters[smallest].append(n)
    return [np.array(sorted(c), dtype=np.int64) for c in clusters]


class ClusterSampleDataset(Dataset):
    """Dataset that yields samples restricted to precomputed node clusters."""

    def __init__(self, base_data: List[Data], clusters: List[np.ndarray]):
        self.base_data = base_data
        self.clusters = clusters
        self.num_clusters = len(clusters)
        base = base_data[0]
        self.edge_index = base.edge_index
        self.edge_attr = getattr(base, "edge_attr", None)
        self.node_type = getattr(base, "node_type", None)
        self.edge_type = getattr(base, "edge_type", None)
        self.subgraphs = []
        ei = torch.tensor(self.edge_index, dtype=torch.long)
        for nodes in clusters:
            nodes_t = torch.tensor(nodes, dtype=torch.long)
            sub_ei, mask = subgraph(nodes_t, ei, relabel_nodes=True)
            self.subgraphs.append((nodes_t, sub_ei, mask))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base_data) * self.num_clusters

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample_idx = idx // self.num_clusters
        cluster_idx = idx % self.num_clusters
        nodes_t, ei, mask = self.subgraphs[cluster_idx]
        base = self.base_data[sample_idx]
        data = Data(x=base.x[nodes_t], y=base.y[nodes_t], edge_index=ei)
        if self.edge_attr is not None:
            data.edge_attr = self.edge_attr[mask]
        if self.node_type is not None:
            data.node_type = self.node_type[nodes_t]
        if self.edge_type is not None:
            data.edge_type = self.edge_type[mask]
        return data


class NeighborSampleDataset(Dataset):
    """Dataset performing random neighbor sampling on-the-fly."""

    def __init__(self, base_data: List[Data], edge_index: np.ndarray, sample_size: int, num_hops: int = 1):
        self.base_data = base_data
        self.sample_size = sample_size
        self.num_hops = num_hops
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        base = base_data[0]
        self.edge_attr = getattr(base, "edge_attr", None)
        self.node_type = getattr(base, "node_type", None)
        self.edge_type = getattr(base, "edge_type", None)
        self.num_nodes = int(self.edge_index.max()) + 1

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base_data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        base = self.base_data[idx]
        targets = np.random.choice(self.num_nodes, min(self.sample_size, self.num_nodes), replace=False)
        subset, ei, mapping, mask = k_hop_subgraph(torch.tensor(targets, dtype=torch.long), self.num_hops, self.edge_index, relabel_nodes=True)
        data = Data(x=base.x[subset], y=base.y[subset], edge_index=ei)
        if self.edge_attr is not None:
            data.edge_attr = self.edge_attr[mask]
        if self.node_type is not None:
            data.node_type = self.node_type[subset]
        if self.edge_type is not None:
            data.edge_type = self.edge_type[mask]
        return data


def _apply_loss(pred: torch.Tensor, target: torch.Tensor, loss_fn: str) -> torch.Tensor:
    """Return loss between ``pred`` and ``target`` using selected objective."""

    if loss_fn == "mse":
        return F.mse_loss(pred, target)
    if loss_fn == "mae":
        return F.l1_loss(pred, target)
    if loss_fn == "huber":
        return F.smooth_l1_loss(pred, target, beta=1.0)
    raise ValueError(f"Unknown loss function: {loss_fn}")


def _select_pump_speeds(
    edge_attr_batch: Optional[torch.Tensor],
    edge_attr: Optional[torch.Tensor],
    edge_attr_phys: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return pump speed tensor aligned with ``edge_type`` mask."""

    if edge_type is None:
        return None
    if isinstance(edge_attr_batch, torch.Tensor):
        return edge_attr_batch.to(device)[..., -1]
    if edge_attr is not None:
        return edge_attr[..., -1]
    if edge_attr_phys is not None:
        return edge_attr_phys[..., -1]
    return None


def train(
    model,
    loader,
    optimizer,
    device,
    check_negative: bool = True,
    amp: bool = False,
    loss_fn: str = "mae",
    node_mask: Optional[torch.Tensor] = None,
    progress: bool = True,
):
    model.train()
    scaler = GradScaler(device=device.type, enabled=amp)
    total_loss = press_total = cl_total = flow_total = 0.0
    grad_total = 0.0
    grad_count = 0
    for batch in tqdm(loader, disable=not progress):
        batch = batch.to(device, non_blocking=True)
        if torch.isnan(batch.x).any() or torch.isnan(batch.y).any():
            raise ValueError("NaN detected in training batch")
        if check_negative and ((batch.x[:, 1] < 0).any() or (batch.y[:, 0] < 0).any()):
            raise ValueError("Negative pressures encountered in training batch")
        optimizer.zero_grad()
        def _model_forward():
            return model(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                getattr(batch, "node_type", None),
                getattr(batch, "edge_type", None),
            )
        with autocast(device_type=device.type, enabled=amp):
            out = _forward_with_auto_checkpoint(model, _model_forward)
            if isinstance(out, dict) and getattr(batch, "edge_y", None) is not None:
                pred_nodes = out["node_outputs"].float()
                edge_pred = out["edge_outputs"].float()
                target_nodes = batch.y.float()
                edge_target = batch.edge_y.float()
                if node_mask is not None:
                    repeat = pred_nodes.size(0) // node_mask.numel()
                    mask = node_mask.repeat(repeat)
                    pred_nodes = pred_nodes[mask]
                    target_nodes = target_nodes[mask]
                loss, press_l, cl_l, flow_l = weighted_mtl_loss(
                    pred_nodes,
                    target_nodes,
                    edge_pred,
                    edge_target,
                    loss_fn=loss_fn,
                )
            else:
                out_t = out if not isinstance(out, dict) else out["node_outputs"]
                target = batch.y.float()
                if node_mask is not None:
                    repeat = out_t.size(0) // node_mask.numel()
                    mask = node_mask.repeat(repeat)
                    out_t = out_t[mask]
                    target = target[mask]
                loss = _apply_loss(out_t, target, loss_fn)
                press_l = cl_l = flow_l = torch.tensor(0.0, device=device)
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        # Clip gradients to mitigate exploding gradients that could otherwise
        # result in ``NaN`` loss values when the optimizer updates the weights.
        grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm = float(grad_norm_t)
        if not math.isfinite(grad_norm):
            logger.warning(
                "Non‑finite grad norm detected; skipping optimizer step"
            )
            grad_norm = None
            optimizer.zero_grad()
        if amp:
            if grad_norm is not None:
                scaler.step(optimizer)
            scaler.update()
        else:
            if grad_norm is not None:
                optimizer.step()
        if grad_norm is not None:
            grad_total += grad_norm
            grad_count += 1
        total_loss += loss.item() * batch.num_graphs
        press_total += press_l.item() * batch.num_graphs
        cl_total += cl_l.item() * batch.num_graphs
        flow_total += flow_l.item() * batch.num_graphs
    denom = len(loader.dataset)
    avg_grad = grad_total / grad_count if grad_count > 0 else None
    return (
        total_loss / denom,
        press_total / denom,
        cl_total / denom,
        flow_total / denom,
        avg_grad,
    )


def evaluate(
    model,
    loader,
    device,
    amp: bool = False,
    loss_fn: str = "mae",
    node_mask: Optional[torch.Tensor] = None,
    progress: bool = True,
):
    global interrupted
    model.eval()
    total_loss = press_total = cl_total = flow_total = 0.0
    data_iter = iter(tqdm(loader, disable=not progress))
    # Collect per-node errors for analysis before reduction
    abs_err_total = sq_err_total = count = None
    mask_vec = None
    with torch.no_grad():
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                if interrupted and "DataLoader worker" in str(e):
                    break
                raise
            batch = batch.to(device, non_blocking=True)
            def _model_forward():
                return model(
                    batch.x,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    getattr(batch, "node_type", None),
                    getattr(batch, "edge_type", None),
                )
            with autocast(device_type=device.type, enabled=amp):
                out = _forward_with_auto_checkpoint(model, _model_forward)
            if isinstance(out, dict) and getattr(batch, "edge_y", None) is not None:
                pred_nodes = out["node_outputs"].float()
                edge_pred = out["edge_outputs"].float()
                target_nodes = batch.y.float()
                edge_target = batch.edge_y.float()
                # Preserve full predictions for per-node metrics
                node_count = pred_nodes.size(0) // batch.num_graphs
                pred_nodes_b = pred_nodes.view(batch.num_graphs, node_count, -1)
                target_nodes_b = target_nodes.view(batch.num_graphs, node_count, -1)
                press_pred = pred_nodes_b[..., 0]
                press_true = target_nodes_b[..., 0]
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict) and "node_outputs" in model.y_mean:
                        p_mean = model.y_mean["node_outputs"].to(device)
                        p_std = model.y_std["node_outputs"].to(device)
                        if p_mean.ndim == 2:
                            p_mean = p_mean[..., 0]
                            p_std = p_std[..., 0]
                        press_pred = press_pred * p_std.view(1, -1) + p_mean.view(1, -1)
                        press_true = press_true * p_std.view(1, -1) + p_mean.view(1, -1)
                    elif not isinstance(model.y_mean, dict):
                        mean, std = _pressure_norm_stats(
                            model,
                            device,
                            press_pred.dtype,
                        )
                        press_pred = _denormalize_pressures(press_pred, mean, std)
                        press_true = _denormalize_pressures(press_true, mean, std)
                if abs_err_total is None:
                    total_nodes = node_mask.numel() if node_mask is not None else node_count
                    abs_err_total = torch.zeros(total_nodes, device=device)
                    sq_err_total = torch.zeros(total_nodes, device=device)
                    count = torch.zeros(total_nodes, device=device)
                    mask_vec = (
                        node_mask.to(device)
                        if node_mask is not None
                        else torch.ones(total_nodes, dtype=torch.bool, device=device)
                    )
                diff = press_pred - press_true
                abs_err_total += diff.abs().sum(dim=0) * mask_vec.float()
                sq_err_total += diff.pow(2).sum(dim=0) * mask_vec.float()
                count += mask_vec.float() * batch.num_graphs
                if node_mask is not None:
                    repeat = pred_nodes.size(0) // node_mask.numel()
                    mask = node_mask.repeat(repeat)
                    pred_nodes = pred_nodes[mask]
                    target_nodes = target_nodes[mask]
                loss, press_l, cl_l, flow_l = weighted_mtl_loss(
                    pred_nodes,
                    target_nodes,
                    edge_pred,
                    edge_target,
                    loss_fn=loss_fn,
                )
            else:
                out_t = out if not isinstance(out, dict) else out["node_outputs"]
                target = batch.y.float()
                node_count = out_t.size(0) // batch.num_graphs
                out_b = out_t.view(batch.num_graphs, node_count, -1)
                tgt_b = target.view(batch.num_graphs, node_count, -1)
                press_pred = out_b[..., 0]
                press_true = tgt_b[..., 0]
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict) and "node_outputs" in model.y_mean:
                        p_mean = model.y_mean["node_outputs"].to(device)
                        p_std = model.y_std["node_outputs"].to(device)
                        if p_mean.ndim == 2:
                            p_mean = p_mean[..., 0]
                            p_std = p_std[..., 0]
                        press_pred = press_pred * p_std.view(1, -1) + p_mean.view(1, -1)
                        press_true = press_true * p_std.view(1, -1) + p_mean.view(1, -1)
                    elif not isinstance(model.y_mean, dict):
                        mean, std = _pressure_norm_stats(
                            model,
                            device,
                            press_pred.dtype,
                        )
                        press_pred = _denormalize_pressures(press_pred, mean, std)
                        press_true = _denormalize_pressures(press_true, mean, std)
                if abs_err_total is None:
                    total_nodes = node_mask.numel() if node_mask is not None else node_count
                    abs_err_total = torch.zeros(total_nodes, device=device)
                    sq_err_total = torch.zeros(total_nodes, device=device)
                    count = torch.zeros(total_nodes, device=device)
                    mask_vec = (
                        node_mask.to(device)
                        if node_mask is not None
                        else torch.ones(total_nodes, dtype=torch.bool, device=device)
                    )
                diff = press_pred - press_true
                abs_err_total += diff.abs().sum(dim=0) * mask_vec.float()
                sq_err_total += diff.pow(2).sum(dim=0) * mask_vec.float()
                count += mask_vec.float() * batch.num_graphs
                if node_mask is not None:
                    repeat = out_t.size(0) // node_mask.numel()
                    mask = node_mask.repeat(repeat)
                    out_t = out_t[mask]
                    target = target[mask]
                loss = _apply_loss(out_t, target, loss_fn)
                press_l = cl_l = flow_l = torch.tensor(0.0, device=device)
            total_loss += loss.item() * batch.num_graphs
            press_total += press_l.item() * batch.num_graphs
            cl_total += cl_l.item() * batch.num_graphs
            flow_total += flow_l.item() * batch.num_graphs
            if interrupted:
                break
    denom = len(loader.dataset)
    if abs_err_total is not None:
        mae = abs_err_total / count.clamp(min=1)
        rmse = torch.sqrt(sq_err_total / count.clamp(min=1))
        mae[count == 0] = float("nan")
        rmse[count == 0] = float("nan")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        pd.DataFrame(
            {
                "node_index": np.arange(mae.numel()),
                "mae": mae.cpu().numpy(),
                "rmse": rmse.cpu().numpy(),
            }
        ).to_csv(log_dir / "eval_node_errors.csv", index=False)
    return (
        total_loss / denom,
        press_total / denom,
        cl_total / denom,
        flow_total / denom,
    )


def _extract_next_demand(
    X_seq: torch.Tensor,
    Y_seq,
    node_count: int,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Return demand at ``t+1`` for mass balance checks.

    Raises
    ------
    KeyError
        If ``Y_seq`` does not contain the ``"demand"`` field.
    """

    if not (isinstance(Y_seq, dict) and "demand" in Y_seq):
        raise KeyError("Missing 'demand' in targets; regenerate data with demand outputs")

    return Y_seq["demand"].permute(2, 0, 1).reshape(node_count, -1)


def train_sequence(
    model: nn.Module,
    loader: TorchLoader,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    edge_attr_phys: torch.Tensor,
    node_type: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    edge_pairs: List[Tuple[int, int]],
    optimizer,
    device,
    pump_coeffs: Optional[torch.Tensor] = None,
    loss_fn: str = "mae",
    physics_loss: bool = False,
    pressure_loss: bool = False,
    pump_loss: bool = False,
    node_mask: Optional[torch.Tensor] = None,
    mass_scale: float = 0.0,
    head_scale: float = 0.0,
    pump_scale: float = 0.0,
    w_mass: float = 2.0,
    w_head: float = 1.0,
    w_pump: float = 1.0,
    w_press: float = 3.0,
    w_cl: float = 1.0,
    w_flow: float = 1.0,
    amp: bool = False,
    progress: bool = True,
    head_sign_weight: float = 0.5,
    has_chlorine: bool = True,
    use_head: bool = True,
    head_idx: Optional[int] = None,
    elev_idx: Optional[int] = None,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    global interrupted
    model.train()
    scaler = GradScaler(device=device.type, enabled=amp)
    total_loss = 0.0
    press_total = cl_total = flow_total = 0.0
    mass_total = head_total = sym_total = pump_total = 0.0
    mass_imb_total = head_viol_total = press_mae_total = 0.0
    grad_total = 0.0
    grad_count = 0
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device) if edge_attr is not None else None
    edge_attr_phys = edge_attr_phys.to(device)
    if node_type is not None:
        node_type = node_type.to(device)
    if edge_type is not None:
        edge_type = edge_type.to(device)
    if pump_coeffs is not None:
        pump_coeffs = pump_coeffs.to(device)
    node_count = int(edge_index.max()) + 1
    data_iter = iter(tqdm(loader, disable=not progress))
    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        except RuntimeError as e:
            if interrupted and "DataLoader worker" in str(e):
                break
            raise
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            X_seq, edge_attr_batch, Y_seq = batch
        else:
            X_seq, Y_seq = batch
            edge_attr_batch = None
        X_seq = X_seq.to(device)
        attr_input = edge_attr_batch.to(device) if isinstance(edge_attr_batch, torch.Tensor) else edge_attr
        if isinstance(Y_seq, dict):
            Y_seq = {k: v.to(device) for k, v in Y_seq.items()}
        else:
            Y_seq = Y_seq.to(device)
        if torch.isnan(X_seq).any() or (
            isinstance(Y_seq, dict)
            and any(torch.isnan(v).any() for v in Y_seq.values())
        ) or (
            not isinstance(Y_seq, dict) and torch.isnan(Y_seq).any()
        ):
            raise ValueError("NaN detected in training batch")
        nt = node_type
        et = edge_type
        if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
            init_press = X_seq[:, 0, model.tank_indices, 1]
            x_mean = getattr(model, "x_mean", None)
            x_std = getattr(model, "x_std", None)
            if x_mean is not None and x_std is not None:
                init_press = _denormalize_feature(
                    init_press,
                    x_mean,
                    x_std,
                    1,
                    node_indices=model.tank_indices,
                )
            init_levels = init_press * model.tank_areas
            model.reset_tank_levels(init_levels)
        optimizer.zero_grad()
        def _model_forward():
            return model(
                X_seq,
                edge_index,
                attr_input,
                nt,
                et,
            )
        with autocast(device_type=device.type, enabled=amp):
            preds = _forward_with_auto_checkpoint(model, _model_forward)
        if isinstance(Y_seq, dict):
            target_nodes = Y_seq['node_outputs'].to(device)
            pred_nodes = preds['node_outputs'].float()
            if node_mask is not None:
                pred_nodes = pred_nodes[:, :, node_mask, :]
                target_nodes = target_nodes[:, :, node_mask, :]
            edge_target = Y_seq['edge_outputs'].unsqueeze(-1).to(device)
            edge_preds = preds['edge_outputs'].float()
            loss, loss_press, loss_cl, loss_edge = weighted_mtl_loss(
                pred_nodes,
                target_nodes.float(),
                edge_preds,
                edge_target.float(),
                loss_fn=loss_fn,
                w_press=w_press,
                w_cl=w_cl,
                w_flow=w_flow,
            )
            # De-normalise pressures to compute MAE in metres
            press_pred = pred_nodes[..., 0]
            press_true = target_nodes[..., 0]
            if hasattr(model, "y_mean") and model.y_mean is not None:
                if isinstance(model.y_mean, dict) and "node_outputs" in model.y_mean:
                    p_mean = model.y_mean["node_outputs"].to(device)
                    p_std = model.y_std["node_outputs"].to(device)
                    if p_mean.ndim == 2:
                        p_mean = p_mean[..., 0]
                        p_std = p_std[..., 0]
                        if node_mask is not None:
                            p_mean = p_mean[node_mask]
                            p_std = p_std[node_mask]
                        press_pred = press_pred * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
                        press_true = press_true * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
                    else:
                        press_pred = press_pred * p_std.view(-1)[0] + p_mean.view(-1)[0]
                        press_true = press_true * p_std.view(-1)[0] + p_mean.view(-1)[0]
                elif not isinstance(model.y_mean, dict):
                    mean, std = _pressure_norm_stats(
                        model,
                        device,
                        press_pred.dtype,
                        node_mask=node_mask,
                    )
                    press_pred = _denormalize_pressures(press_pred, mean, std)
                    press_true = _denormalize_pressures(press_true, mean, std)
            press_mae = torch.mean(torch.abs(press_pred - press_true))
            for name, val in [
                ("pressure", loss_press),
                ("chlorine", loss_cl),
                ("flow", loss_edge),
            ]:
                if (not torch.isfinite(val)) or val.item() > 1e6:
                    raise AssertionError(f"{name} loss {val.item():.3e} invalid")
            if physics_loss:
                flows_mb = edge_preds.squeeze(-1)
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    q_mean = model.y_mean["edge_outputs"].to(device)
                    q_std = model.y_std["edge_outputs"].to(device)
                    flows_mb = flows_mb * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                flows_mb = (
                    flows_mb.permute(2, 0, 1)
                    .reshape(edge_index.size(1), -1)
                )
                demand_mb = _extract_next_demand(
                    X_seq, Y_seq, node_count, model, device
                )
                mass_loss, mass_imb = compute_mass_balance_loss(
                    flows_mb,
                    edge_index,
                    node_count,
                    demand=demand_mb,
                    node_type=nt,
                    return_imbalance=True,
                )
                sym_errors = []
                for i, j in edge_pairs:
                    if et is not None:
                        # only enforce symmetry for pipes (edge_type 0)
                        if et[i] != 0 or et[j] != 0:
                            continue
                    sym_errors.append(flows_mb[i] + flows_mb[j])
                if sym_errors:
                    sym_errors = torch.stack(sym_errors, dim=0)
                    sym_loss = torch.mean(sym_errors ** 2)
                else:
                    sym_loss = torch.tensor(0.0, device=device)
            else:
                mass_loss = torch.tensor(0.0, device=device)
                sym_loss = torch.tensor(0.0, device=device)
                mass_imb = torch.tensor(0.0, device=device)
            head_loss = torch.tensor(0.0, device=device)
            head_violation = torch.tensor(0.0, device=device)
            pump_loss_val = torch.tensor(0.0, device=device)
            if pressure_loss:
                head_loss, head_violation = _compute_head_loss_from_preds(
                    preds['node_outputs'],
                    edge_preds,
                    X_seq,
                    model,
                    device,
                    edge_index,
                    edge_attr_phys,
                    et,
                    nt,
                    head_sign_weight=head_sign_weight,
                    use_head=use_head,
                    head_idx=head_idx,
                    elev_idx=elev_idx,
                    has_chlorine=has_chlorine,
                )
            if pump_loss and pump_coeffs is not None:
                flow_pc = edge_preds.squeeze(-1)
                if hasattr(model, 'y_mean') and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        q_mean = model.y_mean['edge_outputs'].to(device)
                        q_std = model.y_std['edge_outputs'].to(device)
                        flow_pc = flow_pc * q_std + q_mean
                    else:
                        flow_pc = flow_pc * model.y_std[-1].to(device) + model.y_mean[-1].to(device)
                pump_speed_tensor = _select_pump_speeds(
                    edge_attr_batch,
                    edge_attr,
                    edge_attr_phys,
                    et,
                    device,
                )
                pump_loss_val = pump_curve_loss(
                    flow_pc,
                    pump_coeffs,
                    edge_index,
                    et,
                    pump_speeds=pump_speed_tensor,
                )
            if physics_loss or pressure_loss or pump_loss:
                logger.debug(
                    'Raw physics losses - mass: %.6e, head: %.6e, pump: %.6e',
                    mass_loss.detach().item(),
                    head_loss.detach().item(),
                    pump_loss_val.detach().item(),
                )
            mass_denom = 1.0
            if physics_loss or pressure_loss or pump_loss:
                (
                    mass_loss,
                    head_loss,
                    pump_loss_val,
                    mass_denom,
                    _,
                    _,
                ) = scale_physics_losses(
                    mass_loss,
                    head_loss,
                    pump_loss_val,
                    mass_scale=mass_scale,
                    head_scale=head_scale,
                    pump_scale=pump_scale,
                    return_denominators=True,
                )
                if mass_scale > 0:
                    sym_loss = sym_loss / mass_denom
            if physics_loss:
                loss = loss + w_mass * (mass_loss + sym_loss)
            if pressure_loss:
                loss = loss + w_head * head_loss
            if pump_loss:
                loss = loss + w_pump * pump_loss_val
        else:
            Y_seq = Y_seq.to(device)
            loss_press = loss_cl = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
            head_loss = pump_loss_val = torch.tensor(0.0, device=device)
            mass_imb = head_violation = torch.tensor(0.0, device=device)
            press_mae = torch.tensor(0.0, device=device)
            with autocast(device_type=device.type, enabled=amp):
                loss = _apply_loss(preds, Y_seq.float(), loss_fn)
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm = float(grad_norm_t)
        if not math.isfinite(grad_norm):
            logger.warning(
                "Non‑finite grad norm detected; skipping optimizer step"
            )
            grad_norm = None
            optimizer.zero_grad()
        if amp:
            if grad_norm is not None:
                scaler.step(optimizer)
            scaler.update()
        else:
            if grad_norm is not None:
                optimizer.step()
        if grad_norm is not None:
            grad_total += grad_norm
            grad_count += 1
        total_loss += loss.item() * X_seq.size(0)
        press_total += loss_press.item() * X_seq.size(0)
        cl_total += loss_cl.item() * X_seq.size(0)
        flow_total += loss_edge.item() * X_seq.size(0)
        mass_total += mass_loss.item() * X_seq.size(0)
        head_total += head_loss.item() * X_seq.size(0)
        sym_total += sym_loss.item() * X_seq.size(0)
        pump_total += pump_loss_val.item() * X_seq.size(0)
        mass_imb_total += mass_imb.item() * X_seq.size(0)
        head_viol_total += head_violation.item() * X_seq.size(0)
        press_mae_total += press_mae.item() * X_seq.size(0)
        if interrupted:
            break
    denom = len(loader.dataset)
    avg_grad = grad_total / grad_count if grad_count > 0 else None
    return (
        total_loss / denom,
        press_total / denom,
        cl_total / denom,
        flow_total / denom,
        mass_total / denom,
        head_total / denom,
        sym_total / denom,
        pump_total / denom,
        mass_imb_total / denom,
        head_viol_total / denom,
        press_mae_total / denom,
        avg_grad,
    )


def estimate_physics_scales_from_data(
    loader: TorchLoader,
    edge_index: torch.Tensor,
    edge_attr_phys: torch.Tensor,
    node_type: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    device,
    model: nn.Module,
    pump_coeffs: Optional[torch.Tensor] = None,
    head_sign_weight: float = 0.5,
    has_chlorine: bool = True,
    use_head: bool = True,
    head_idx: Optional[int] = None,
    elev_idx: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Estimate baseline magnitudes using ground-truth flows and pressures."""

    edge_index = edge_index.to(device)
    edge_attr_phys = edge_attr_phys.to(device)
    if node_type is not None:
        node_type = node_type.to(device)
    if edge_type is not None:
        edge_type = edge_type.to(device)
    if pump_coeffs is not None:
        pump_coeffs = pump_coeffs.to(device)
    node_count = int(edge_index.max()) + 1
    mass_total = head_total = pump_total = 0.0
    num_samples = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                X_seq, _edge_attr_batch, Y_seq = batch
            else:
                X_seq, Y_seq = batch
            if not isinstance(Y_seq, dict):
                continue
            X_seq = X_seq.to(device)
            target_nodes = Y_seq["node_outputs"].to(device)
            edge_target = Y_seq["edge_outputs"].to(device)

            flows = edge_target
            press = target_nodes[..., 0].float()
            if hasattr(model, "y_mean") and model.y_mean is not None:
                if isinstance(model.y_mean, dict):
                    p_mean = model.y_mean["node_outputs"].to(device)
                    p_std = model.y_std["node_outputs"].to(device)
                    if p_mean.ndim == 2:
                        p_mean = p_mean[..., 0]
                        p_std = p_std[..., 0]
                    press = press * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
                    q_mean = model.y_mean["edge_outputs"].to(device)
                    q_std = model.y_std["edge_outputs"].to(device)
                    flows = flows * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                else:
                    mean, std = _pressure_norm_stats(
                        model,
                        device,
                        press.dtype,
                    )
                    press = _denormalize_pressures(press, mean, std)

            flows_mb = flows.permute(2, 0, 1).reshape(edge_index.size(1), -1)
            demand_mb = _extract_next_demand(
                X_seq, Y_seq, node_count, model, device
            )
            mass_loss, _ = compute_mass_balance_loss(
                flows_mb,
                edge_index,
                node_count,
                demand=demand_mb,
                node_type=node_type,
                return_imbalance=True,
            )

            elevation = None
            use_head_local = use_head
            elevation_col = _resolve_elevation_index(head_idx, elev_idx, has_chlorine)
            if (
                use_head
                and elevation_col is not None
                and X_seq.size(-1) > elevation_col
            ):
                elevation = X_seq[..., elevation_col].float()
                if hasattr(model, 'x_mean') and model.x_mean is not None:
                    x_mean = model.x_mean.to(device)
                    x_std = model.x_std.to(device)
                    if x_mean.ndim == 1:
                        elevation = (
                            elevation * x_std[elevation_col] + x_mean[elevation_col]
                        )
                    else:
                        elevation = (
                            elevation * x_std[:, elevation_col].view(1, 1, -1)
                            + x_mean[:, elevation_col].view(1, 1, -1)
                        )
            else:
                use_head_local = False
            head_loss, _ = pressure_headloss_consistency_loss(
                press,
                flows,
                edge_index,
                edge_attr_phys,
                elevation=elevation,
                edge_type=edge_type,
                node_type=node_type,
                return_violation=True,
                sign_weight=head_sign_weight,
                use_head=use_head_local,
            )

            if pump_coeffs is not None:
                flow_pc = edge_target
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        q_mean = model.y_mean["edge_outputs"].to(device)
                        q_std = model.y_std["edge_outputs"].to(device)
                        flow_pc = flow_pc * q_std + q_mean
                    else:
                        flow_pc = (
                            flow_pc * model.y_std[-1].to(device) + model.y_mean[-1].to(device)
                        )
                pump_speed_tensor = _select_pump_speeds(
                    _edge_attr_batch,
                    None,
                    edge_attr_phys,
                    edge_type,
                    device,
                )
                pump_loss_val = pump_curve_loss(
                    flow_pc,
                    pump_coeffs,
                    edge_index,
                    edge_type,
                    pump_speeds=pump_speed_tensor,
                )
            else:
                pump_loss_val = torch.tensor(0.0, device=device)

            bsz = X_seq.size(0)
            mass_total += mass_loss.item() * bsz
            head_total += head_loss.item() * bsz
            pump_total += pump_loss_val.item() * bsz
            num_samples += bsz
    denom = max(num_samples, 1)
    return mass_total / denom, head_total / denom, pump_total / denom


def evaluate_sequence(
    model: nn.Module,
    loader: TorchLoader,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_attr_phys: torch.Tensor,
    node_type: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    edge_pairs: List[Tuple[int, int]],
    device,
    pump_coeffs: Optional[torch.Tensor] = None,
    loss_fn: str = "mae",
    physics_loss: bool = False,
    pressure_loss: bool = False,
    pump_loss: bool = False,
    node_mask: Optional[torch.Tensor] = None,
    mass_scale: float = 0.0,
    head_scale: float = 0.0,
    pump_scale: float = 0.0,
    w_mass: float = 2.0,
    w_head: float = 1.0,
    w_pump: float = 1.0,
    w_press: float = 3.0,
    w_cl: float = 1.0,
    w_flow: float = 1.0,
    amp: bool = False,
    progress: bool = True,
    head_sign_weight: float = 0.5,
    has_chlorine: bool = True,
    use_head: bool = True,
    head_idx: Optional[int] = None,
    elev_idx: Optional[int] = None,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
    global interrupted
    model.eval()
    pressure_loss_flag = bool(pressure_loss)
    total_loss = 0.0
    press_total = cl_total = flow_total = 0.0
    mass_total = head_total = sym_total = pump_total = 0.0
    mass_imb_total = head_viol_total = press_mae_total = 0.0
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device) if edge_attr is not None else None
    edge_attr_phys = edge_attr_phys.to(device)
    if node_type is not None:
        node_type = node_type.to(device)
    if edge_type is not None:
        edge_type = edge_type.to(device)
    if pump_coeffs is not None:
        pump_coeffs = pump_coeffs.to(device)
    node_count = int(edge_index.max()) + 1
    data_iter = iter(tqdm(loader, disable=not progress))
    abs_err_total = sq_err_total = count = None
    with torch.no_grad():
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                if interrupted and "DataLoader worker" in str(e):
                    break
                raise
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                X_seq, edge_attr_batch, Y_seq = batch
            else:
                X_seq, Y_seq = batch
                edge_attr_batch = None
            X_seq = X_seq.to(device)
            attr = edge_attr_batch.to(device) if isinstance(edge_attr_batch, torch.Tensor) else edge_attr
            nt = node_type
            et = edge_type
            if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
                init_press = X_seq[:, 0, model.tank_indices, 1]
                x_mean = getattr(model, "x_mean", None)
                x_std = getattr(model, "x_std", None)
                if x_mean is not None and x_std is not None:
                    init_press = _denormalize_feature(
                        init_press,
                        x_mean,
                        x_std,
                        1,
                        node_indices=model.tank_indices,
                    )
                init_levels = init_press * model.tank_areas
                model.reset_tank_levels(init_levels)
            def _model_forward():
                return model(
                    X_seq,
                    edge_index,
                    attr,
                    nt,
                    et,
                )
            with autocast(device_type=device.type, enabled=amp):
                preds = _forward_with_auto_checkpoint(model, _model_forward)
            loss = torch.tensor(0.0, device=device)
            loss_press = torch.tensor(0.0, device=device)
            loss_cl = torch.tensor(0.0, device=device)
            loss_edge = torch.tensor(0.0, device=device)
            mass_loss = torch.tensor(0.0, device=device)
            sym_loss = torch.tensor(0.0, device=device)
            mass_imb = torch.tensor(0.0, device=device)
            head_loss = torch.tensor(0.0, device=device)
            head_violation = torch.tensor(0.0, device=device)
            pump_loss_val = torch.tensor(0.0, device=device)
            press_mae = torch.tensor(0.0, device=device)
            if pressure_loss_flag and isinstance(preds, dict):
                head_loss, head_violation = _compute_head_loss_from_preds(
                    preds["node_outputs"],
                    preds["edge_outputs"].float(),
                    X_seq,
                    model,
                    device,
                    edge_index,
                    edge_attr_phys,
                    et,
                    nt,
                    head_sign_weight=head_sign_weight,
                    use_head=use_head,
                    head_idx=head_idx,
                    elev_idx=elev_idx,
                    has_chlorine=has_chlorine,
                )
            if isinstance(Y_seq, dict):
                target_nodes_full = Y_seq["node_outputs"].to(device)
                pred_nodes_full = preds["node_outputs"].float()
                press_pred_full = pred_nodes_full[..., 0]
                press_true_full = target_nodes_full[..., 0]
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict) and "node_outputs" in model.y_mean:
                        p_mean_full = model.y_mean["node_outputs"].to(device)
                        p_std_full = model.y_std["node_outputs"].to(device)
                        if p_mean_full.ndim == 2:
                            p_mean_full = p_mean_full[..., 0]
                            p_std_full = p_std_full[..., 0]
                        press_pred_full = (
                            press_pred_full * p_std_full.view(1, 1, -1)
                            + p_mean_full.view(1, 1, -1)
                        )
                        press_true_full = (
                            press_true_full * p_std_full.view(1, 1, -1)
                            + p_mean_full.view(1, 1, -1)
                        )
                    elif not isinstance(model.y_mean, dict):
                        mean, std = _pressure_norm_stats(
                            model,
                            device,
                            press_pred_full.dtype,
                        )
                        press_pred_full = _denormalize_pressures(
                            press_pred_full,
                            mean,
                            std,
                        )
                        press_true_full = _denormalize_pressures(
                            press_true_full,
                            mean,
                            std,
                        )
                if node_mask is not None:
                    pred_nodes = pred_nodes_full[:, :, node_mask, :]
                    target_nodes = target_nodes_full[:, :, node_mask, :]
                    press_pred = press_pred_full[:, :, node_mask]
                    press_true = press_true_full[:, :, node_mask]
                else:
                    pred_nodes = pred_nodes_full
                    target_nodes = target_nodes_full
                    press_pred = press_pred_full
                    press_true = press_true_full
                edge_target = Y_seq["edge_outputs"].unsqueeze(-1).to(device)
                edge_preds = preds["edge_outputs"].float()
                loss, loss_press, loss_cl, loss_edge = weighted_mtl_loss(
                    pred_nodes,
                    target_nodes.float(),
                    edge_preds,
                    edge_target.float(),
                    loss_fn=loss_fn,
                    w_press=w_press,
                    w_cl=w_cl,
                    w_flow=w_flow,
                )
                press_mae = torch.mean(torch.abs(press_pred - press_true))
                if abs_err_total is None:
                    total_nodes = node_mask.numel() if node_mask is not None else press_pred_full.size(2)
                    abs_err_total = torch.zeros(total_nodes, device=device)
                    sq_err_total = torch.zeros(total_nodes, device=device)
                    count = torch.zeros(total_nodes, device=device)
                diff_full = press_pred_full - press_true_full
                if node_mask is not None:
                    mask_vec = node_mask.to(device)
                    abs_err_total[mask_vec] += diff_full[:, :, mask_vec].abs().sum(dim=(0, 1))
                    sq_err_total[mask_vec] += diff_full[:, :, mask_vec].pow(2).sum(dim=(0, 1))
                    count[mask_vec] += diff_full.shape[0] * diff_full.shape[1]
                else:
                    abs_err_total += diff_full.abs().sum(dim=(0, 1))
                    sq_err_total += diff_full.pow(2).sum(dim=(0, 1))
                    count += diff_full.shape[0] * diff_full.shape[1]
                mass_loss = torch.tensor(0.0, device=device)
                sym_loss = torch.tensor(0.0, device=device)
                mass_imb = torch.tensor(0.0, device=device)
                if physics_loss:
                    flows_mb = edge_preds.squeeze(-1)
                    if hasattr(model, "y_mean") and model.y_mean is not None:
                        q_mean = model.y_mean["edge_outputs"].to(device)
                        q_std = model.y_std["edge_outputs"].to(device)
                        flows_mb = flows_mb * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                    flows_mb = (
                        flows_mb.permute(2, 0, 1)
                        .reshape(edge_index.size(1), -1)
                    )
                    demand_mb = _extract_next_demand(
                        X_seq, Y_seq, node_count, model, device
                    )
                    mass_loss, mass_imb = compute_mass_balance_loss(
                        flows_mb,
                        edge_index,
                        node_count,
                        demand=demand_mb,
                        node_type=nt,
                        return_imbalance=True,
                    )
                    sym_errors = []
                    for i, j in edge_pairs:
                        if et is not None:
                            # only enforce symmetry for pipes (edge_type 0)
                            if et[i] != 0 or et[j] != 0:
                                continue
                        sym_errors.append(flows_mb[i] + flows_mb[j])
                    if sym_errors:
                        sym_errors = torch.stack(sym_errors, dim=0)
                        sym_loss = torch.mean(sym_errors ** 2)
                else:
                    mass_imb = torch.tensor(0.0, device=device)
                if not pressure_loss_flag:
                    head_violation = torch.tensor(0.0, device=device)
                if pump_loss and pump_coeffs is not None:
                    flow_pc = edge_preds.squeeze(-1)
                    if hasattr(model, "y_mean") and model.y_mean is not None:
                        if isinstance(model.y_mean, dict):
                            q_mean = model.y_mean['edge_outputs'].to(device)
                            q_std = model.y_std['edge_outputs'].to(device)
                            flow_pc = flow_pc * q_std + q_mean
                        else:
                            flow_pc = flow_pc * model.y_std[-1].to(device) + model.y_mean[-1].to(device)
                    pump_speed_tensor = _select_pump_speeds(
                        edge_attr_batch,
                        edge_attr,
                        edge_attr_phys,
                        et,
                        device,
                    )
                    pump_loss_val = pump_curve_loss(
                        flow_pc,
                        pump_coeffs,
                        edge_index,
                        et,
                        pump_speeds=pump_speed_tensor,
                    )
                if physics_loss or pressure_loss_flag or pump_loss:
                    logger.debug(
                        "Raw physics losses - mass: %.6e, head: %.6e, pump: %.6e",
                        mass_loss.detach().item(),
                        head_loss.detach().item(),
                        pump_loss_val.detach().item(),
                    )
                mass_denom = 1.0
                if physics_loss or pressure_loss_flag or pump_loss:
                    (
                        mass_loss,
                        head_loss,
                        pump_loss_val,
                        mass_denom,
                        _,
                        _,
                    ) = scale_physics_losses(
                        mass_loss,
                        head_loss,
                        pump_loss_val,
                        mass_scale=mass_scale,
                        head_scale=head_scale,
                        pump_scale=pump_scale,
                        return_denominators=True,
                    )
                    if mass_scale > 0:
                        sym_loss = sym_loss / mass_denom
                if physics_loss:
                    loss = loss + w_mass * (mass_loss + sym_loss)
                if pressure_loss_flag:
                    loss = loss + w_head * head_loss
                if pump_loss:
                    loss = loss + w_pump * pump_loss_val
            else:
                Y_seq = Y_seq.to(device)
                loss_press = loss_cl = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
                head_loss = pump_loss_val = torch.tensor(0.0, device=device)
                mass_imb = head_violation = torch.tensor(0.0, device=device)
                press_mae = torch.tensor(0.0, device=device)
                loss = _apply_loss(preds, Y_seq.float(), loss_fn)
                if abs_err_total is None and preds.dim() >= 3:
                    total_nodes = node_mask.numel() if node_mask is not None else preds.size(2)
                    abs_err_total = torch.zeros(total_nodes, device=device)
                    sq_err_total = torch.zeros(total_nodes, device=device)
                    count = torch.zeros(total_nodes, device=device)
                if preds.dim() >= 3:
                    press_pred_full = preds[..., 0]
                    press_true_full = Y_seq[..., 0]
                    diff_full = press_pred_full - press_true_full
                    if node_mask is not None:
                        mask_vec = node_mask.to(device)
                        abs_err_total[mask_vec] += diff_full[:, :, mask_vec].abs().sum(dim=(0, 1))
                        sq_err_total[mask_vec] += diff_full[:, :, mask_vec].pow(2).sum(dim=(0, 1))
                        count[mask_vec] += diff_full.shape[0] * diff_full.shape[1]
                    else:
                        abs_err_total += diff_full.abs().sum(dim=(0, 1))
                        sq_err_total += diff_full.pow(2).sum(dim=(0, 1))
                        count += diff_full.shape[0] * diff_full.shape[1]
            total_loss += loss.item() * X_seq.size(0)
            press_total += loss_press.item() * X_seq.size(0)
            cl_total += loss_cl.item() * X_seq.size(0)
            flow_total += loss_edge.item() * X_seq.size(0)
            mass_total += mass_loss.item() * X_seq.size(0)
            head_total += head_loss.item() * X_seq.size(0)
            sym_total += sym_loss.item() * X_seq.size(0)
            pump_total += pump_loss_val.item() * X_seq.size(0)
            mass_imb_total += mass_imb.item() * X_seq.size(0)
            head_viol_total += head_violation.item() * X_seq.size(0)
            press_mae_total += press_mae.item() * X_seq.size(0)
            if interrupted:
                break
    denom = len(loader.dataset)
    if abs_err_total is not None:
        mae = abs_err_total / count.clamp(min=1)
        rmse = torch.sqrt(sq_err_total / count.clamp(min=1))
        mae[count == 0] = float("nan")
        rmse[count == 0] = float("nan")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        pd.DataFrame(
            {
                "node_index": np.arange(mae.numel()),
                "mae": mae.cpu().numpy(),
                "rmse": rmse.cpu().numpy(),
            }
        ).to_csv(log_dir / "eval_sequence_node_errors.csv", index=False)
    return (
        total_loss / denom,
        press_total / denom,
        cl_total / denom,
        flow_total / denom,
        mass_total / denom,
        head_total / denom,
        sym_total / denom,
        pump_total / denom,
        mass_imb_total / denom,
        head_viol_total / denom,
        press_mae_total / denom,
    )

# Resolve important directories relative to the repository root so that training
# can be launched from any location.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
PLOTS_DIR = REPO_ROOT / "plots"


def main(args: argparse.Namespace):
    configure_seeds(args.seed, args.deterministic)
    signal.signal(signal.SIGINT, _signal_handler)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    edge_index_np = np.load(args.edge_index_path)
    wn = wntr.network.WaterNetworkModel(args.inp_path)
    # Always compute the physical edge attributes from the network
    edge_attr_phys_np = build_edge_attr(wn, edge_index_np)
    if edge_attr_phys_np.shape[1] == 4:
        edge_attr_phys_np = np.concatenate(
            [edge_attr_phys_np, np.zeros((edge_attr_phys_np.shape[0], 1), dtype=edge_attr_phys_np.dtype)],
            axis=1,
        )
    skip_edge_attr_cols: Optional[List[int]] = None
    if edge_attr_phys_np.shape[1] >= 5:
        last_five = range(edge_attr_phys_np.shape[1] - 5, edge_attr_phys_np.shape[1])
        skip_edge_attr_cols = list(last_five)
    elif edge_attr_phys_np.shape[1] >= 2:
        skip_edge_attr_cols = [edge_attr_phys_np.shape[1] - 2, edge_attr_phys_np.shape[1] - 1]
    edge_attr_phys = torch.tensor(edge_attr_phys_np.copy(), dtype=torch.float32)
    if os.path.exists(args.pump_coeffs_path):
        pump_coeffs_np = np.load(args.pump_coeffs_path)
    else:
        pump_coeffs_np = build_pump_coeffs(wn, edge_index_np)
    pump_coeffs_tensor = torch.tensor(pump_coeffs_np.copy(), dtype=torch.float32)

    if os.path.exists(args.edge_attr_path):
        edge_attr = np.load(args.edge_attr_path)
    else:
        edge_attr = edge_attr_phys_np.copy()
        # log-transform roughness like in data generation
        edge_attr[:, 2] = np.log1p(edge_attr[:, 2])
    if edge_attr.shape[1] == 4:
        edge_attr = np.concatenate(
            [edge_attr, np.zeros((edge_attr.shape[0], 1), dtype=edge_attr.dtype)], axis=1
        )
    edge_types = build_edge_type(wn, edge_index_np)
    edge_pairs = build_edge_pairs(edge_index_np, edge_types)
    node_types = build_node_type(wn)
    loss_mask = build_loss_mask(wn).to(device)
    node_mask_np = loss_mask.cpu().numpy()
    # Always allocate a distinct node type for tanks even if they are absent
    # from the network to ensure ``HydroConv`` learns a dedicated transform.
    num_node_types = max(int(np.max(node_types)) + 1, 2)
    num_edge_types = int(np.max(edge_types)) + 1
    edge_mean = edge_std = None
    X_raw = np.load(args.x_path, allow_pickle=True)
    Y_raw = np.load(args.y_path, allow_pickle=True)
    seq_mode = X_raw.ndim == 4
    edge_attr_train_seq = edge_attr_val_seq = edge_attr_test_seq = None
    if seq_mode:
        if os.path.exists(args.edge_attr_train_seq_path):
            edge_attr_train_seq = np.load(args.edge_attr_train_seq_path, allow_pickle=True)
        elif isinstance(Y_raw[0], dict) and "edge_attr_seq" in Y_raw[0]:
            edge_attr_train_seq = np.stack([y["edge_attr_seq"] for y in Y_raw]).astype(np.float32)
        if args.x_val_path and os.path.exists(args.edge_attr_val_seq_path):
            edge_attr_val_seq = np.load(args.edge_attr_val_seq_path, allow_pickle=True)
        elif args.x_val_path and os.path.exists(args.x_val_path):
            Y_val_tmp = np.load(args.y_val_path, allow_pickle=True)
            if isinstance(Y_val_tmp[0], dict) and "edge_attr_seq" in Y_val_tmp[0]:
                edge_attr_val_seq = np.stack([y["edge_attr_seq"] for y in Y_val_tmp]).astype(np.float32)
        if args.x_test_path and os.path.exists(args.edge_attr_test_seq_path):
            edge_attr_test_seq = np.load(args.edge_attr_test_seq_path, allow_pickle=True)
        if edge_attr_train_seq is not None:
            edge_attr_train_seq = np.asarray(edge_attr_train_seq, dtype=np.float32)
        if edge_attr_val_seq is not None:
            edge_attr_val_seq = np.asarray(edge_attr_val_seq, dtype=np.float32)
        if edge_attr_test_seq is not None:
            edge_attr_test_seq = np.asarray(edge_attr_test_seq, dtype=np.float32)
        if edge_attr_train_seq is not None:
            flat = edge_attr_train_seq.reshape(-1, edge_attr_train_seq.shape[-1])
            edge_mean = torch.tensor(flat.mean(axis=0), dtype=torch.float32)
            edge_std = torch.tensor(flat.std(axis=0) + 1e-8, dtype=torch.float32)
            edge_mean, edge_std = sanitize_edge_attr_stats(
                edge_mean, edge_std, skip_edge_attr_cols
            )
    if edge_mean is None or edge_std is None:
        edge_mean, edge_std = compute_edge_attr_stats(edge_attr)
    edge_mean, edge_std = sanitize_edge_attr_stats(edge_mean, edge_std, skip_edge_attr_cols)

    if seq_mode:
        data_ds = SequenceDataset(
            X_raw,
            Y_raw,
            edge_index_np,
            edge_attr,
            node_type=node_types,
            edge_type=edge_types,
            edge_attr_seq=edge_attr_train_seq,
        )
        loader = TorchLoader(
            data_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )
    else:
        data_list = load_dataset(
            args.x_path,
            args.y_path,
            args.edge_index_path,
            edge_attr=edge_attr,
            node_type=node_types,
            edge_type=edge_types,
            expected_node_names=wn.node_name_list,
        )
        loader = DataLoader(
            data_list,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )


    val_ds = None
    if args.x_val_path and os.path.exists(args.x_val_path):
        Xv = np.load(args.x_val_path, allow_pickle=True)
        if seq_mode:
            Yv = np.load(args.y_val_path, allow_pickle=True)
            if Xv.ndim == 3:
                Yv = np.array([{k: v[None, ...] for k, v in y.items()} for y in Yv], dtype=object)
                Xv = Xv[:, None, ...]
            val_ds = SequenceDataset(
                Xv,
                Yv,
                edge_index_np,
                edge_attr,
                node_type=node_types,
                edge_type=edge_types,
                edge_attr_seq=edge_attr_val_seq,
            )
            val_loader = TorchLoader(
                val_ds,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
            val_list = val_ds
        else:
            if Xv.ndim >= 4:
                logger.warning(
                    "Validation dataset at %s has %d dimensions but training data is static; skipping validation set",
                    args.x_val_path,
                    Xv.ndim,
                )
                val_list = []
                val_loader = None
            else:
                Yv = np.load(args.y_val_path, allow_pickle=True)
                val_list = load_dataset(
                    args.x_val_path,
                    args.y_val_path,
                    args.edge_index_path,
                    edge_attr=edge_attr,
                    node_type=node_types,
                    edge_type=edge_types,
                    expected_node_names=wn.node_name_list,
                )
                val_loader = DataLoader(
                    val_list,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=args.workers > 0,
                )
    else:
        val_list = []
        val_loader = None
    pump_count = len(wn.pump_name_list)
    if seq_mode:
        sample_dim = data_ds.X.shape[-1]
        first_target = Y_raw[0]
        if isinstance(first_target, dict):
            node_out = first_target.get("node_outputs")
            target_dim = int(node_out.shape[-1]) if node_out is not None else 1
        else:
            node_arr = np.asarray(first_target)
            target_dim = int(node_arr.shape[-1]) if node_arr.ndim >= 2 else 1
    else:
        sample_dim = data_list[0].num_node_features
        first_target = data_list[0].y
        target_dim = int(first_target.size(-1)) if first_target.ndim >= 2 else 1
    manifest = None
    pump_feature_repeats = 0
    manifest_path = Path(args.x_path).with_name("manifest.json")
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("Failed to read manifest at %s: %s", manifest_path, exc)

    feature_layout: Optional[List[str]] = None
    include_head_feature = False
    has_chlorine: Optional[bool] = None
    head_idx: Optional[int] = None
    manifest_pump_blocks = 0
    if manifest is not None:
        feature_layout = manifest.get("node_feature_layout")
        include_head_feature = bool(manifest.get("include_head"))
        manifest_pump_blocks = int(manifest.get("pump_feature_repeats", manifest_pump_blocks))
        if "include_chlorine" in manifest:
            has_chlorine = bool(manifest["include_chlorine"])
        if feature_layout:
            include_head_feature = "head" in feature_layout
            if include_head_feature:
                try:
                    head_idx = feature_layout.index("head")
                except ValueError:
                    head_idx = None
            if has_chlorine is None:
                has_chlorine = "chlorine" in feature_layout

    pump_feature_repeats = manifest_pump_blocks if manifest_pump_blocks else pump_feature_repeats
    effective_pump_blocks = pump_feature_repeats if pump_feature_repeats > 0 else (1 if pump_count > 0 else 0)
    if pump_count > 0 and manifest_pump_blocks and manifest_pump_blocks != effective_pump_blocks:
        logger.warning("Manifest pump feature repeats (%d) differ from detected layout (%d); using detected value", manifest_pump_blocks, effective_pump_blocks)
    base_dim = sample_dim - pump_count * effective_pump_blocks
    if target_dim <= 1:
        has_chlorine = False
    if has_chlorine is None and target_dim > 1:
        has_chlorine = True
    if has_chlorine is None:
        if base_dim == 5:
            has_chlorine = True
        elif base_dim == 4:
            has_chlorine = False
        else:
            raise ValueError(
                f"Dataset provides {sample_dim} features per node but the network has {pump_count} pumps."
            )
    if not include_head_feature:
        expected_base = 5 if has_chlorine else 4
        if base_dim >= expected_base:
            include_head_feature = True
    if target_dim <= 1 and has_chlorine:
        logger.warning(
            "Target dimension indicates pressure-only dataset; disabling chlorine outputs"
        )
        has_chlorine = False
    if feature_layout:
        layout_pump_blocks = manifest_pump_blocks if manifest_pump_blocks else effective_pump_blocks
        layout_base_len = len(feature_layout) - pump_count * layout_pump_blocks
        if layout_base_len != base_dim:
            logger.warning(
                "Manifest feature layout (%d entries) does not match dataset feature dimension (%d)",
                len(feature_layout),
                base_dim,
            )
    if feature_layout and "elevation" in feature_layout:
        elev_idx = feature_layout.index("elevation")
    else:
        elev_idx = base_dim - 1
    if head_idx is None:
        if feature_layout and "head" in feature_layout:
            head_idx = feature_layout.index("head")
        elif include_head_feature and base_dim >= 2:
            head_idx = base_dim - 2
    if feature_layout and "pressure" in feature_layout:
        pressure_idx = feature_layout.index("pressure")
    else:
        pressure_idx = 1
    demand_idx = (
        feature_layout.index("demand")
        if feature_layout and "demand" in feature_layout
        else 0
    )
    if head_idx is not None:
        pump_start = head_idx + 2
    elif elev_idx is not None:
        pump_start = elev_idx + 1
    else:
        pump_start = base_dim
    args.output_dim = 2 if has_chlorine else 1
    norm_md5 = None
    if args.normalize:
        static_cols = None
        if args.per_node_norm:
            static_cols = [elev_idx]
            if include_head_feature and head_idx is not None:
                if head_idx not in static_cols:
                    static_cols.append(head_idx)
        norm_mask = loss_mask.cpu()
        if seq_mode:
            x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(
                X_raw,
                Y_raw,
                per_node=args.per_node_norm,
                static_cols=static_cols,
                node_mask=norm_mask,
            )
            apply_sequence_normalization(
                data_ds,
                x_mean,
                x_std,
                y_mean,
                y_std,
                edge_mean,
                edge_std,
                per_node=args.per_node_norm,
                static_cols=static_cols,
                skip_edge_attr_cols=skip_edge_attr_cols,
            )
            if isinstance(val_list, SequenceDataset):
                apply_sequence_normalization(
                    val_list,
                    x_mean,
                    x_std,
                    y_mean,
                    y_std,
                    edge_mean,
                    edge_std,
                    per_node=args.per_node_norm,
                    static_cols=static_cols,
                    skip_edge_attr_cols=skip_edge_attr_cols,
                )
        else:
            x_mean, x_std, y_mean, y_std = compute_norm_stats(
                data_list,
                per_node=args.per_node_norm,
                static_cols=static_cols,
                node_mask=norm_mask,
            )
            apply_normalization(
                data_list,
                x_mean,
                x_std,
                y_mean,
                y_std,
                edge_mean,
                edge_std,
                per_node=args.per_node_norm,
                skip_edge_attr_cols=skip_edge_attr_cols,
            )
            if val_list:
                apply_normalization(
                    val_list,
                    x_mean,
                    x_std,
                    y_mean,
                    y_std,
                    edge_mean,
                    edge_std,
                    per_node=args.per_node_norm,
                    skip_edge_attr_cols=skip_edge_attr_cols,
                )
        print("Target normalization stats:")
        pressure_stats, chlorine_stats = summarize_target_norm_stats(
            y_mean, y_std, has_chlorine
        )
        print("Pressure mean/std:", *pressure_stats)
        if chlorine_stats is not None:
            print("Chlorine mean/std:", *chlorine_stats)

        import hashlib

        md5 = hashlib.md5()
        if isinstance(y_mean, dict):
            arrays = [
                x_mean,
                x_std,
                y_mean.get("node_outputs"),
                y_std.get("node_outputs"),
                y_mean.get("edge_outputs"),
                y_std.get("edge_outputs"),
                edge_mean,
                edge_std,
            ]
        else:
            arrays = [x_mean, x_std, y_mean, y_std, edge_mean, edge_std]
        for arr in arrays:
            if arr is None:
                continue
            md5.update(arr.to(torch.float32).cpu().numpy().tobytes())
        norm_md5 = md5.hexdigest()
        norm_stats = {
            "x_mean": x_mean.to(torch.float32).cpu().numpy(),
            "x_std": x_std.to(torch.float32).cpu().numpy(),
            "edge_mean": edge_mean.to(torch.float32).cpu().numpy(),
            "edge_std": edge_std.to(torch.float32).cpu().numpy(),
        }
        if isinstance(y_mean, dict):
            norm_stats["y_mean"] = {k: v.to(torch.float32).cpu().numpy() for k, v in y_mean.items()}
            norm_stats["y_std"] = {k: y_std[k].to(torch.float32).cpu().numpy() for k in y_mean}
        else:
            norm_stats["y_mean"] = y_mean.to(torch.float32).cpu().numpy()
            norm_stats["y_std"] = y_std.to(torch.float32).cpu().numpy()
        norm_stats["hash"] = norm_md5
    else:
        x_mean = x_std = y_mean = y_std = None
        norm_stats = None

    if not seq_mode:
        if args.neighbor_sampling:
            sample_size = args.cluster_batch_size or max(
                1, int(0.2 * data_list[0].num_nodes)
            )
            data_list = NeighborSampleDataset(
                data_list, edge_index_np, sample_size
            )
            loader = DataLoader(
                data_list,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
            if val_loader is not None:
                val_loader = DataLoader(
                    NeighborSampleDataset(val_list, edge_index_np, sample_size),
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=args.workers > 0,
                )
        elif args.cluster_batch_size > 0:
            clusters = partition_graph_greedy(
                edge_index_np, data_list[0].num_nodes, args.cluster_batch_size
            )
            data_list = ClusterSampleDataset(data_list, clusters)
            loader = DataLoader(
                data_list,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
            if val_loader is not None:
                val_loader = DataLoader(
                    ClusterSampleDataset(val_list, clusters),
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=args.workers > 0,
                )
        else:
            loader = DataLoader(
                data_list,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
            if val_loader is not None:
                val_loader = DataLoader(
                    val_list,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=args.workers > 0,
                )

    pump_offset = 5 if has_chlorine else 4
    pump_count = len(wn.pump_name_list)
    min_expected_dim = pump_offset + pump_count
    pump_feature_repeats = 0
    has_pump_head_features = False

    if seq_mode:
        sample_dim = data_ds.X.shape[-1]
        if sample_dim < min_expected_dim:
            raise ValueError(
                f"Dataset provides {sample_dim} features per node but the network "
                f"defines {len(wn.pump_name_list)} pumps (expected at least {min_expected_dim}).\n"
                "Re-generate the training data with pump control inputs using scripts/data_generation.py."
            )
        extra_pump_features = max(sample_dim - pump_offset, 0)
        if pump_count > 0 and extra_pump_features >= 0:
            pump_feature_repeats = min(extra_pump_features // pump_count, 2)
            has_pump_head_features = pump_feature_repeats >= 2
        if has_pump_head_features:
            logger.info("Detected per-node pump head features; enabling extended pump conditioning.")
        if getattr(data_ds, "multi", False):
            model = MultiTaskGNNSurrogate(
                in_channels=sample_dim,
                hidden_channels=args.hidden_dim,
                edge_dim=edge_attr.shape[1],
                node_output_dim=2 if has_chlorine else 1,
                edge_output_dim=1,
                num_layers=args.num_layers,
                use_attention=args.use_attention,
                gat_heads=args.gat_heads,
                dropout=args.dropout,
                residual=args.residual,
                rnn_hidden_dim=args.rnn_hidden_dim,
                share_weights=args.share_weights,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                use_checkpoint=args.checkpoint,
                pressure_feature_idx=pressure_idx,
                num_pumps=len(wn.pump_name_list),
                pump_feature_offset=pump_offset,
                pump_feature_repeats=effective_pump_blocks,
            ).to(device)
            tank_indices = [i for i, n in enumerate(wn.node_name_list) if n in wn.tank_name_list]
            model.tank_indices = torch.tensor(tank_indices, device=device, dtype=torch.long)
            areas = []
            for name in wn.tank_name_list:
                node = wn.get_node(name)
                diam = getattr(node, "diameter", 0.0)
                areas.append(np.pi * (float(diam) ** 2) / 4.0)
            model.tank_areas = torch.tensor(areas, device=device)
            tank_edges = []
            tank_signs = []
            for idx in tank_indices:
                src = np.where(edge_index_np[0] == idx)[0]
                tgt = np.where(edge_index_np[1] == idx)[0]
                tank_edges.append(torch.tensor(np.concatenate([src, tgt]), dtype=torch.long, device=device))
                signs = np.concatenate([-np.ones(len(src)), np.ones(len(tgt))])
                tank_signs.append(torch.tensor(signs, dtype=torch.float32, device=device))
            model.tank_edges = tank_edges
            model.tank_signs = tank_signs
            model.reset_tank_levels(batch_size=1, device=device)
            timestep_seconds = float(getattr(wn.options.time, "hydraulic_timestep", 3600.0))
            if hasattr(model, "timestep_seconds"):
                model.timestep_seconds = timestep_seconds
            if hasattr(model, "flow_unit_scale"):
                flow_scales = estimate_tank_flow_scale(
                    X_raw,
                    Y_raw,
                    edge_index_np,
                    tank_indices,
                    areas,
                    timestep_seconds,
                )
                if flow_scales.size:
                    model.flow_unit_scale = torch.tensor(
                        flow_scales,
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    model.flow_unit_scale = 1.0
        else:
            model = RecurrentGNNSurrogate(
                in_channels=sample_dim,
                hidden_channels=args.hidden_dim,
                edge_dim=edge_attr.shape[1],
                output_dim=args.output_dim,
                num_layers=args.num_layers,
                use_attention=args.use_attention,
                gat_heads=args.gat_heads,
                dropout=args.dropout,
                residual=args.residual,
                rnn_hidden_dim=args.rnn_hidden_dim,
                share_weights=args.share_weights,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                use_checkpoint=args.checkpoint,
                pressure_feature_idx=pressure_idx,
                num_pumps=len(wn.pump_name_list),
                pump_feature_offset=pump_offset,
            ).to(device)
            tank_indices = [i for i, n in enumerate(wn.node_name_list) if n in wn.tank_name_list]
            model.tank_indices = torch.tensor(tank_indices, device=device, dtype=torch.long)
            areas = []
            for name in wn.tank_name_list:
                node = wn.get_node(name)
                diam = getattr(node, "diameter", 0.0)
                areas.append(np.pi * (float(diam) ** 2) / 4.0)
            model.tank_areas = torch.tensor(areas, device=device)
            tank_edges = []
            tank_signs = []
            for idx in tank_indices:
                src = np.where(edge_index_np[0] == idx)[0]
                tgt = np.where(edge_index_np[1] == idx)[0]
                tank_edges.append(torch.tensor(np.concatenate([src, tgt]), dtype=torch.long, device=device))
                signs = np.concatenate([-np.ones(len(src)), np.ones(len(tgt))])
                tank_signs.append(torch.tensor(signs, dtype=torch.float32, device=device))
            model.tank_edges = tank_edges
            model.tank_signs = tank_signs
            model.reset_tank_levels(batch_size=1, device=device)
            timestep_seconds = float(getattr(wn.options.time, "hydraulic_timestep", 3600.0))
            if hasattr(model, "timestep_seconds"):
                model.timestep_seconds = timestep_seconds
            if hasattr(model, "flow_unit_scale"):
                flow_scales = estimate_tank_flow_scale(
                    X_raw,
                    Y_raw,
                    edge_index_np,
                    tank_indices,
                    areas,
                    timestep_seconds,
                )
                if flow_scales.size:
                    model.flow_unit_scale = torch.tensor(
                        flow_scales,
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    model.flow_unit_scale = 1.0
    else:
        sample = data_list[0]
        sample_dim = sample.num_node_features
        if sample_dim < min_expected_dim:
            raise ValueError(
                f"Dataset provides {sample.num_node_features} features per node but the network "
                f"defines {len(wn.pump_name_list)} pumps (expected at least {min_expected_dim}).\n"
                "Re-generate the training data with pump control inputs using scripts/data_generation.py."
            )
        extra_pump_features = max(sample_dim - pump_offset, 0)
        if pump_count > 0 and extra_pump_features >= 0:
            pump_feature_repeats = min(extra_pump_features // pump_count, 2)
            has_pump_head_features = pump_feature_repeats >= 2
        if has_pump_head_features:
            logger.info("Detected per-node pump head features; enabling extended pump conditioning.")
        model = GCNEncoder(
            in_channels=sample.num_node_features,
            hidden_channels=args.hidden_dim,
            out_channels=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            residual=args.residual,
            edge_dim=edge_attr.shape[1],
            use_attention=args.use_attention,
            gat_heads=args.gat_heads,
            share_weights=args.share_weights,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        ).to(device)

    model_meta = {
        "model_class": model.__class__.__name__,
        "in_channels": sample_dim if seq_mode else sample.num_node_features,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "use_attention": args.use_attention,
        "gat_heads": args.gat_heads,
        "residual": args.residual,
        "dropout": args.dropout,
        "activation": args.activation,
        "output_dim": args.output_dim,
        "edge_dim": edge_attr.shape[1],
        "rnn_hidden_dim": args.rnn_hidden_dim,
        "share_weights": args.share_weights,
        "num_node_types": num_node_types,
        "num_edge_types": num_edge_types,
        "edge_scaler": "MinMax",
        "log_roughness": True,
        "pressure_feature_idx": pressure_idx,
        "use_pressure_skip": True,
        "pump_feature_offset": pump_offset,
        "pump_feature_repeats": pump_feature_repeats,
        "has_pump_head_features": has_pump_head_features,
    }
    if norm_md5 is not None:
        model_meta["norm_stats_md5"] = norm_md5

    # expose normalization stats on the model for later un-normalisation
    if args.normalize:
        if seq_mode and getattr(data_ds, "multi", False):
            # for multi-task sequence data retain separate stats for node and
            # edge outputs so physics losses can un-normalize correctly
            model.y_mean = y_mean
            model.y_std = y_std
        else:
            model.y_mean = y_mean
            model.y_std = y_std
        model.x_mean = x_mean
        model.x_std = x_std
    else:
        model.x_mean = model.x_std = model.y_mean = model.y_std = None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

    mass_scale = args.mass_scale
    head_scale = args.head_scale
    pump_scale = args.pump_scale
    if seq_mode and (
        (args.physics_loss and mass_scale <= 0)
        or (args.pressure_loss and head_scale <= 0)
        or (args.pump_loss and pump_scale <= 0)
    ):
        m_base, h_base, p_base = estimate_physics_scales_from_data(
            loader,
            data_ds.edge_index,
            edge_attr_phys,
            data_ds.node_type,
            data_ds.edge_type,
            device,
            model,
            pump_coeffs_tensor if args.pump_loss else None,
            head_sign_weight=getattr(args, "head_sign_weight", 0.5),
            has_chlorine=has_chlorine,
            use_head=args.physics_loss_use_head,
            head_idx=head_idx,
            elev_idx=elev_idx,
        )
        if args.physics_loss and mass_scale <= 0:
            mass_scale = m_base
        if args.pressure_loss and head_scale <= 0:
            head_scale = h_base
        if args.pump_loss and pump_scale <= 0:
            pump_scale = p_base
    args.mass_scale = mass_scale
    args.head_scale = head_scale
    args.pump_scale = pump_scale
    if seq_mode and (args.physics_loss or args.pressure_loss or args.pump_loss):
        print(
            f"Using physics loss scales: mass={mass_scale:.4e}, head={head_scale:.4e}, pump={pump_scale:.4e}"
        )
    pressure_only = args.w_cl <= 0 and args.w_flow <= 0

    # prepare logging
    if args.resume:
        model_path = args.resume
        base_resume, _ = os.path.splitext(args.resume)
        run_name = Path(base_resume).name
        norm_path = f"{base_resume}_norm.npz"
        log_path = os.path.join(DATA_DIR, f"training_{run_name}.log")
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(args.output)
        model_path = f"{base}_{run_name}{ext}"
        norm_path = f"{base}_{run_name}_norm.npz"
        log_path = os.path.join(DATA_DIR, f"training_{run_name}.log")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pred_csv_path = (
        Path(args.pred_csv)
        if args.pred_csv
        else DATA_DIR / f"pressures_{run_name}.csv"
    )
    losses = []
    loss_components = []
    grad_norms = []
    tb_writer = None
    if SummaryWriter is not None:
        tb_log_dir = REPO_ROOT / "logs" / f"tb_{run_name}"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    epoch = start_epoch
    with open(log_path, "w") as f:
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"args: {vars(args)}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"device: {device}\n")
        f.write(
            f"scales: mass={mass_scale}, head={head_scale}, pump={pump_scale}\n"
        )
        if seq_mode:
            if pressure_only:
                f.write(
                    "epoch,train_loss,val_press_loss,press_loss,cl_loss,flow_loss,mass_imbalance,head_violation_pct,press_mae,val_cl_loss,val_flow_loss,val_mass_imbalance,val_head_violation_pct,val_press_mae,lr,grad_norm\n"
                )
            else:
                f.write(
                    "epoch,train_loss,val_loss,press_loss,cl_loss,flow_loss,mass_imbalance,head_violation_pct,press_mae,val_press_loss,val_cl_loss,val_flow_loss,val_mass_imbalance,val_head_violation_pct,val_press_mae,lr,grad_norm\n"
                )
        else:
            if pressure_only:
                f.write(
                    "epoch,train_loss,val_press_loss,press_loss,cl_loss,flow_loss,val_cl_loss,val_flow_loss,lr,grad_norm\n"
                )
            else:
                f.write(
                    "epoch,train_loss,val_loss,press_loss,cl_loss,flow_loss,val_press_loss,val_cl_loss,val_flow_loss,lr,grad_norm\n"
                )
        best_val = float("inf")
        patience = 0
        for epoch in range(start_epoch, args.epochs):
            w_mass_curr = ramp_weight(
                args.w_mass, epoch, getattr(args, "mass_anneal", 0)
            )
            w_head_curr = ramp_weight(
                args.w_head, epoch, getattr(args, "head_anneal", 0)
            )
            w_pump_curr = ramp_weight(
                args.w_pump, epoch, getattr(args, "pump_anneal", 0)
            )
            if seq_mode:
                loss_tuple = train_sequence(
                    model,
                    loader,
                    data_ds.edge_index,
                    data_ds.edge_attr,
                    edge_attr_phys,
                    data_ds.node_type,
                    data_ds.edge_type,
                    edge_pairs,
                    optimizer,
                    device,
                    pump_coeffs_tensor,
                    loss_fn=args.loss_fn,
                    physics_loss=args.physics_loss,
                    pressure_loss=args.pressure_loss,
                    pump_loss=args.pump_loss,
                    node_mask=loss_mask,
                    mass_scale=mass_scale,
                    head_scale=head_scale,
                    pump_scale=pump_scale,
                    w_mass=w_mass_curr,
                    w_head=w_head_curr,
                    w_pump=w_pump_curr,
                    w_press=args.w_press,
                    w_cl=args.w_cl,
                    w_flow=args.w_flow,
                    amp=args.amp,
                    progress=args.progress,
                    head_sign_weight=getattr(args, "head_sign_weight", 0.5),
                    has_chlorine=has_chlorine,
                    use_head=args.physics_loss_use_head,
                    head_idx=head_idx,
                    elev_idx=elev_idx,
                )
                loss = loss_tuple[0]
                (
                    press_l,
                    cl_l,
                    flow_l,
                    mass_l,
                    head_l,
                    sym_l,
                    pump_l,
                    mass_imb,
                    head_viols,
                    press_mae,
                    grad_norm,
                ) = loss_tuple[1:]
                comp = [press_l, cl_l, flow_l, mass_l, sym_l]
                if args.pressure_loss:
                    comp.append(head_l)
                if args.pump_loss:
                    comp.append(pump_l)
                loss_components.append(tuple(comp))
                if grad_norm is not None:
                    grad_norms.append(grad_norm)
                if val_loader is not None and not interrupted:
                    val_tuple = evaluate_sequence(
                        model,
                        val_loader,
                        data_ds.edge_index,
                        data_ds.edge_attr,
                        edge_attr_phys,
                        data_ds.node_type,
                        data_ds.edge_type,
                        edge_pairs,
                        device,
                        pump_coeffs_tensor,
                        loss_fn=args.loss_fn,
                        physics_loss=args.physics_loss,
                        pressure_loss=args.pressure_loss,
                        pump_loss=args.pump_loss,
                        node_mask=loss_mask,
                        mass_scale=mass_scale,
                        head_scale=head_scale,
                        pump_scale=pump_scale,
                        w_mass=w_mass_curr,
                        w_head=w_head_curr,
                        w_pump=w_pump_curr,
                        w_press=args.w_press,
                        w_cl=args.w_cl,
                        w_flow=args.w_flow,
                        amp=args.amp,
                        progress=args.progress,
                        head_sign_weight=getattr(args, "head_sign_weight", 0.5),
                        has_chlorine=has_chlorine,
                        use_head=args.physics_loss_use_head,
                        head_idx=head_idx,
                        elev_idx=elev_idx,
                    )
                    val_loss = val_tuple[0]
                    (
                        val_press_l,
                        val_cl_l,
                        val_flow_l,
                        val_mass_l,
                        val_head_l,
                        val_sym_l,
                        val_pump_l,
                        val_mass_imb,
                        val_head_viols,
                        val_press_mae,
                    ) = val_tuple[1:]
                else:
                    val_loss = loss
                    val_press_l, val_cl_l, val_flow_l = press_l, cl_l, flow_l
                    val_mass_imb, val_head_viols = mass_imb, head_viols
                    val_press_mae = press_mae
            else:
                loss, press_l, cl_l, flow_l, grad_norm = train(
                    model,
                    loader,
                    optimizer,
                    device,
                    check_negative=not args.normalize,
                    amp=args.amp,
                    loss_fn=args.loss_fn,
                    node_mask=loss_mask,
                    progress=args.progress,
                )
                loss_components.append((press_l, cl_l, flow_l))
                if grad_norm is not None:
                    grad_norms.append(grad_norm)
                if val_loader is not None and not interrupted:
                    val_loss, val_press_l, val_cl_l, val_flow_l = evaluate(
                        model,
                        val_loader,
                        device,
                        amp=args.amp,
                        loss_fn=args.loss_fn,
                        node_mask=loss_mask,
                        progress=args.progress,
                    )
                else:
                    val_loss = loss
                    val_press_l, val_cl_l, val_flow_l = press_l, cl_l, flow_l
            val_metric = val_press_l if pressure_only else val_loss
            scheduler.step(val_metric)
            curr_lr = optimizer.param_groups[0]['lr']
            losses.append((loss, val_metric))
            grad_val = grad_norm if grad_norm is not None else float("nan")
            if seq_mode:
                if pressure_only:
                    f.write(
                        f"{epoch},{loss:.6f},{val_press_l:.6f},{press_l:.6f},{cl_l:.6f},{flow_l:.6f},{mass_imb:.6f},{head_viols * 100:.6f},{press_mae:.6f},"
                        f"{val_cl_l:.6f},{val_flow_l:.6f},{val_mass_imb:.6f},{val_head_viols * 100:.6f},{val_press_mae:.6f},{curr_lr:.6e},{grad_val:.6f}\n"
                    )
                else:
                    f.write(
                        f"{epoch},{loss:.6f},{val_loss:.6f},{press_l:.6f},{cl_l:.6f},{flow_l:.6f},{mass_imb:.6f},{head_viols * 100:.6f},{press_mae:.6f},"
                        f"{val_press_l:.6f},{val_cl_l:.6f},{val_flow_l:.6f},{val_mass_imb:.6f},{val_head_viols * 100:.6f},{val_press_mae:.6f},{curr_lr:.6e},{grad_val:.6f}\n"
                    )
                if tb_writer is not None:
                    tb_writer.add_scalars(
                        "loss/train",
                        {
                            "total": loss,
                            "pressure": press_l,
                            "chlorine": cl_l,
                            "flow": flow_l,
                        },
                        epoch,
                    )
                    tb_writer.add_scalars(
                        "loss/val",
                        {
                            "total": val_metric,
                            "pressure": val_press_l,
                            "chlorine": val_cl_l,
                            "flow": val_flow_l,
                        },
                        epoch,
                    )
                    tb_writer.add_scalars(
                        "metrics/mass_imbalance",
                        {"train": mass_imb, "val": val_mass_imb},
                        epoch,
                    )
                    tb_writer.add_scalars(
                        "metrics/head_violation_pct",
                        {"train": head_viols * 100.0, "val": val_head_viols * 100.0},
                        epoch,
                    )
                    tb_writer.add_scalars(
                        "metrics/pressure_mae_m",
                        {"train": press_mae, "val": val_press_mae},
                        epoch,
                    )
                    if grad_norm is not None:
                        tb_writer.add_scalar(
                            "metrics/grad_norm",
                            grad_norm,
                            epoch,
                        )
                if args.physics_loss:
                    msg = (
                        f"Epoch {epoch}: press={press_l:.3f}, cl={cl_l:.3f}, flow={flow_l:.3f}, "
                        f"mass={mass_l:.3f}, sym={sym_l:.3f}, imb={mass_imb:.3f}"
                    )
                    if args.pressure_loss:
                        msg += f", head={head_l:.3f}, viol%={head_viols * 100:.2f}"
                    if args.pump_loss:
                        msg += f", pump={pump_l:.3f}"
                        if pump_l > PUMP_LOSS_WARN_THRESHOLD:
                            warnings.warn(
                                f"Pump loss {pump_l:.3f} exceeds {PUMP_LOSS_WARN_THRESHOLD}",
                                stacklevel=2,
                            )
                    msg += f", pMAE={press_mae:.3f}"
                    print(msg)
                else:
                    print(f"Epoch {epoch}: pMAE={press_mae:.3f}")
            else:
                if pressure_only:
                    f.write(
                        f"{epoch},{loss:.6f},{val_press_l:.6f},{press_l:.6f},{cl_l:.6f},{flow_l:.6f},"
                        f"{val_cl_l:.6f},{val_flow_l:.6f},{curr_lr:.6e},{grad_val:.6f}\n"
                    )
                else:
                    f.write(
                        f"{epoch},{loss:.6f},{val_loss:.6f},{press_l:.6f},{cl_l:.6f},{flow_l:.6f},"
                        f"{val_press_l:.6f},{val_cl_l:.6f},{val_flow_l:.6f},{curr_lr:.6e},{grad_val:.6f}\n"
                    )
                if tb_writer is not None:
                    tb_writer.add_scalars(
                        "loss/train",
                        {
                            "total": loss,
                            "pressure": press_l,
                            "chlorine": cl_l,
                            "flow": flow_l,
                        },
                        epoch,
                    )
                    tb_writer.add_scalars(
                        "loss/val",
                        {
                            "total": val_metric,
                            "pressure": val_press_l,
                            "chlorine": val_cl_l,
                            "flow": val_flow_l,
                        },
                        epoch,
                    )
                    if grad_norm is not None:
                        tb_writer.add_scalar(
                            "metrics/grad_norm",
                            grad_norm,
                            epoch,
                        )
                print(
                    f"Epoch {epoch}: press={press_l:.3f}, cl={cl_l:.3f}, flow={flow_l:.3f}"
                )
            if val_metric < best_val - 1e-6:
                best_val = val_metric
                patience = 0
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                }
                if norm_stats is not None:
                    ckpt["norm_stats"] = norm_stats
                meta = dict(model_meta)
                if norm_stats is not None:
                    meta.update(
                        {
                            "x_mean": norm_stats["x_mean"],
                            "x_std": norm_stats["x_std"],
                            "y_mean": norm_stats["y_mean"],
                            "y_std": norm_stats["y_std"],
                            "edge_mean": norm_stats["edge_mean"],
                            "edge_std": norm_stats["edge_std"],
                        }
                    )
                ckpt["model_meta"] = meta
                torch.save(ckpt, model_path)
                if norm_stats is not None:
                    y_mean_np = norm_stats["y_mean"]
                    y_std_np = norm_stats["y_std"]
                    if isinstance(y_mean_np, dict):
                        np.savez(
                            norm_path,
                            x_mean=norm_stats["x_mean"],
                            x_std=norm_stats["x_std"],
                            y_mean_node=y_mean_np["node_outputs"],
                            y_std_node=y_std_np["node_outputs"],
                            y_mean_edge=y_mean_np["edge_outputs"],
                            y_std_edge=y_std_np["edge_outputs"],
                            edge_mean=norm_stats["edge_mean"],
                            edge_std=norm_stats["edge_std"],
                        )
                    else:
                        np.savez(
                            norm_path,
                            x_mean=norm_stats["x_mean"],
                            x_std=norm_stats["x_std"],
                            y_mean=y_mean_np,
                            y_std=y_std_np,
                            edge_mean=norm_stats["edge_mean"],
                            edge_std=norm_stats["edge_std"],
                        )
            else:
                patience += 1
            if interrupted:
                break
            if patience >= args.early_stop_patience:
                break
        if interrupted:
            handle_keyboard_interrupt(
                model_path, model, optimizer, scheduler, epoch, norm_stats, model_meta
            )

    # Ensure a checkpoint exists even if validation never improved
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        if norm_stats is not None:
            ckpt["norm_stats"] = norm_stats
            y_mean_np = norm_stats["y_mean"]
            y_std_np = norm_stats["y_std"]
            if isinstance(y_mean_np, dict):
                np.savez(
                    norm_path,
                    x_mean=norm_stats["x_mean"],
                    x_std=norm_stats["x_std"],
                    y_mean_node=y_mean_np["node_outputs"],
                    y_std_node=y_std_np["node_outputs"],
                    y_mean_edge=y_mean_np["edge_outputs"],
                    y_std_edge=y_std_np["edge_outputs"],
                    edge_mean=norm_stats["edge_mean"],
                    edge_std=norm_stats["edge_std"],
                )
            else:
                np.savez(
                    norm_path,
                    x_mean=norm_stats["x_mean"],
                    x_std=norm_stats["x_std"],
                    y_mean=y_mean_np,
                    y_std=y_std_np,
                    edge_mean=norm_stats["edge_mean"],
                    edge_std=norm_stats["edge_std"],
                )
        torch.save(ckpt, model_path)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    # plot loss curve
    if losses:
        tr = [l[0] for l in losses]
        vl = [l[1] for l in losses]
        plt.figure()
        plt.loglog(tr, label="train")
        plt.loglog(vl, label="val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"loss_curve_{run_name}.png"))
        plt.close()

    if loss_components:
        plot_loss_components(loss_components, run_name)

    if grad_norms:
        valid_grad_norms = [g for g in grad_norms if g is not None]
        if valid_grad_norms:
            plt.figure()
            plt.plot(valid_grad_norms)
            plt.xlabel("Epoch")
            plt.ylabel("Gradient Norm")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"grad_norm_{run_name}.png"))
            plt.close()

    # scatter plot of predictions vs actual on test set
    test_loader = None
    test_ds = None
    test_list = None
    X_test_raw = None
    if args.x_test_path and os.path.exists(args.x_test_path):
        if seq_mode:
            Xt = np.load(args.x_test_path, allow_pickle=True)
            Yt = np.load(args.y_test_path, allow_pickle=True)
            if Xt.ndim == 3:
                Yt = np.array([{k: v[None, ...] for k, v in y.items()} for y in Yt], dtype=object)
                Xt = Xt[:, None, ...]
            X_test_raw = Xt.copy()
            test_ds = SequenceDataset(
                Xt,
                Yt,
                edge_index_np,
                edge_attr,
                node_type=node_types,
                edge_type=edge_types,
                edge_attr_seq=edge_attr_test_seq,
            )
            if args.normalize:
                apply_sequence_normalization(
                    test_ds,
                    x_mean,
                    x_std,
                    y_mean,
                    y_std,
                    edge_mean,
                    edge_std,
                    per_node=args.per_node_norm,
                    static_cols=static_cols,
                    skip_edge_attr_cols=skip_edge_attr_cols,
                )
            test_loader = TorchLoader(
                test_ds,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
        else:
            X_test_raw = np.load(args.x_test_path, allow_pickle=True)
            if X_test_raw.ndim >= 4:
                logger.warning(
                    "Test dataset at %s has %d dimensions but training data is static; skipping test set",
                    args.x_test_path,
                    X_test_raw.ndim,
                )
                test_list = []
                test_loader = None
            else:
                test_list = load_dataset(
                    args.x_test_path,
                    args.y_test_path,
                    args.edge_index_path,
                    edge_attr=edge_attr,
                    node_type=node_types,
                    edge_type=edge_types,
                    expected_node_names=wn.node_name_list,
                )
                if args.normalize:
                    apply_normalization(
                        test_list,
                        x_mean,
                        x_std,
                        y_mean,
                        y_std,
                        edge_mean,
                        edge_std,
                        per_node=args.per_node_norm,
                        skip_edge_attr_cols=skip_edge_attr_cols,
                    )
                test_loader = DataLoader(
                    test_list,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=args.workers > 0,
                )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state)
        model.eval()

        p_stats = RunningStats()
        c_stats = RunningStats() if has_chlorine else None
        f_stats = RunningStats()
        sample_cap = int(max(args.eval_sample, 0))
        sample_size = 0
        sample_preds_p: List[float] = []
        sample_true_p: List[float] = []
        sample_preds_c: List[float] = []
        sample_true_c: List[float] = []
        sequence_example: Dict[str, np.ndarray] = {}

        exclude = set(wn.reservoir_name_list)
        node_mask_np = np.array([n not in exclude for n in wn.node_name_list])
        node_mask = torch.tensor(node_mask_np, dtype=torch.bool, device=device)

        if test_loader is None:
            logger.warning(
                "Skipping test evaluation because no compatible test loader was created"
            )
        else:
            with torch.no_grad():
                if seq_mode:
                    ei = test_ds.edge_index.to(device)
                    ea = test_ds.edge_attr.to(device) if test_ds.edge_attr is not None else None
                    nt = test_ds.node_type.to(device) if test_ds.node_type is not None else None
                    et = test_ds.edge_type.to(device) if test_ds.edge_type is not None else None
                    for batch in test_loader:
                        if isinstance(batch, (list, tuple)) and len(batch) == 3:
                            X_seq, edge_attr_batch, Y_seq = batch
                        else:
                            X_seq, Y_seq = batch
                            edge_attr_batch = None
                        X_seq = X_seq.to(device)
                        attr = edge_attr_batch.to(device) if isinstance(edge_attr_batch, torch.Tensor) else ea

                        def _model_forward():
                            return model(X_seq, ei, attr, nt, et)
                        with autocast(device_type=device.type, enabled=args.amp):
                            out = _forward_with_auto_checkpoint(model, _model_forward)
                        if isinstance(out, dict):
                            node_pred = out["node_outputs"]
                            edge_pred = out.get("edge_outputs")
                        else:
                            node_pred = out
                            edge_pred = None
                        if isinstance(Y_seq, dict):
                            Y_node = Y_seq["node_outputs"].to(node_pred.device)
                            Y_edge = Y_seq.get("edge_outputs")
                            if Y_edge is not None:
                                Y_edge = Y_edge.to(node_pred.device)
                        else:
                            Y_node = Y_seq.to(node_pred.device)
                            Y_edge = None
                        if hasattr(model, "y_mean") and model.y_mean is not None:
                            if isinstance(model.y_mean, dict):
                                y_mean_node = model.y_mean['node_outputs'].to(node_pred.device)
                                y_std_node = model.y_std['node_outputs'].to(node_pred.device)
                                node_pred = node_pred * y_std_node + y_mean_node
                                Y_node = Y_node * y_std_node + y_mean_node
                                if edge_pred is not None and Y_edge is not None:
                                    y_mean_edge = model.y_mean['edge_outputs'].to(node_pred.device)
                                    y_std_edge = model.y_std['edge_outputs'].to(node_pred.device)
                                    edge_pred = edge_pred.squeeze(-1)
                                    edge_pred = edge_pred * y_std_edge + y_mean_edge
                                    Y_edge = Y_edge * y_std_edge + y_mean_edge
                            else:
                                y_std = model.y_std.to(node_pred.device)
                                y_mean = model.y_mean.to(node_pred.device)
                                node_pred = node_pred * y_std + y_mean
                                Y_node = Y_node * y_std + y_mean

                        pred_p = node_pred[..., 0].reshape(-1, node_mask.numel())
                        true_p = Y_node[..., 0].reshape(-1, node_mask.numel())
                        pred_p = pred_p[:, node_mask].reshape(-1)
                        true_p = true_p[:, node_mask].reshape(-1)
                        p_stats.update(pred_p.cpu().numpy(), true_p.cpu().numpy())
                        if sample_cap and len(sample_preds_p) < sample_cap:
                            take = min(sample_cap - len(sample_preds_p), pred_p.numel())
                            sample_preds_p.extend(pred_p[:take].cpu().numpy())
                            sample_true_p.extend(true_p[:take].cpu().numpy())

                        if has_chlorine and node_pred.shape[-1] > 1:
                            pred_c = node_pred[..., 1].reshape(-1, node_mask.numel())
                            true_c = Y_node[..., 1].reshape(-1, node_mask.numel())
                            pred_c = pred_c[:, node_mask].reshape(-1)
                            true_c = true_c[:, node_mask].reshape(-1)
                            if c_stats is not None:
                                c_stats.update(pred_c.cpu().numpy(), true_c.cpu().numpy())
                            if sample_cap and len(sample_preds_c) < sample_cap:
                                take = min(sample_cap - len(sample_preds_c), pred_c.numel())
                                sample_preds_c.extend(pred_c[:take].cpu().numpy())
                                sample_true_c.extend(true_c[:take].cpu().numpy())

                        if edge_pred is not None and Y_edge is not None:
                            pred_f = edge_pred.reshape(-1)
                            true_f = Y_edge.reshape(-1)
                            f_stats.update(pred_f.cpu().numpy(), true_f.cpu().numpy())
                        if "pred_nodes" not in sequence_example:
                            sequence_example["pred_nodes"] = (
                                node_pred[0].detach().cpu().numpy()
                            )
                            sequence_example["true_nodes"] = (
                                Y_node[0].detach().cpu().numpy()
                            )
                            if edge_pred is not None:
                                sequence_example["pred_edges"] = (
                                    edge_pred[0].detach().cpu().numpy()
                                )
                            if Y_edge is not None:
                                sequence_example["true_edges"] = (
                                    Y_edge[0].detach().cpu().numpy()
                                )
                else:
                    for batch in test_loader:
                        batch = batch.to(device, non_blocking=True)

                        def _model_forward():
                            return model(
                                batch.x,
                                batch.edge_index,
                                getattr(batch, "edge_attr", None),
                                getattr(batch, "node_type", None),
                                getattr(batch, "edge_type", None),
                            )
                        with autocast(device_type=device.type, enabled=args.amp):
                            out = _forward_with_auto_checkpoint(model, _model_forward)
                        if hasattr(model, "y_mean") and model.y_mean is not None:
                            if isinstance(model.y_mean, dict):
                                y_mean_node = model.y_mean['node_outputs'].to(out.device)
                                y_std_node = model.y_std['node_outputs'].to(out.device)
                                node_out = (
                                    out["node_outputs"] if isinstance(out, dict) else out
                                )
                                node_out = node_out * y_std_node + y_mean_node
                                batch_y = batch.y * y_std_node + y_mean_node
                                if isinstance(out, dict) and getattr(batch, "edge_y", None) is not None:
                                    y_mean_edge = model.y_mean['edge_outputs'].to(out.device)
                                    y_std_edge = model.y_std['edge_outputs'].to(out.device)
                                    edge_out = out["edge_outputs"].squeeze(-1)
                                    edge_y = batch.edge_y.squeeze(-1)
                                    edge_out = edge_out * y_std_edge + y_mean_edge
                                    edge_y = edge_y * y_std_edge + y_mean_edge
                                else:
                                    edge_out = edge_y = None
                            else:
                                y_std = model.y_std.to(out.device)
                                y_mean = model.y_mean.to(out.device)
                                node_out = (
                                    out["node_outputs"] if isinstance(out, dict) else out
                                )
                                node_out = node_out * y_std + y_mean
                                batch_y = batch.y * y_std + y_mean
                                edge_out = out.get("edge_outputs") if isinstance(out, dict) else None
                                edge_y = batch.edge_y if getattr(batch, "edge_y", None) is not None else None
                        else:
                            node_out = out["node_outputs"] if isinstance(out, dict) else out
                            batch_y = batch.y
                            edge_out = out.get("edge_outputs") if isinstance(out, dict) else None
                            edge_y = batch.edge_y if getattr(batch, "edge_y", None) is not None else None
                            if edge_out is not None:
                                edge_out = edge_out.squeeze(-1)
                        if edge_y is not None:
                            edge_y = edge_y.squeeze(-1)

                    mask_batch = node_mask.repeat(batch.num_graphs)
                    pred_p = node_out[:, 0][mask_batch]
                    true_p = batch_y[:, 0][mask_batch]
                    p_stats.update(pred_p.cpu().numpy(), true_p.cpu().numpy())
                    if sample_cap and len(sample_preds_p) < sample_cap:
                        take = min(sample_cap - len(sample_preds_p), pred_p.numel())
                        sample_preds_p.extend(pred_p[:take].cpu().numpy())
                        sample_true_p.extend(true_p[:take].cpu().numpy())

                    if has_chlorine and node_out.shape[1] > 1:
                        pred_c = node_out[:, 1][mask_batch]
                        true_c = batch_y[:, 1][mask_batch]
                        if c_stats is not None:
                            c_stats.update(pred_c.cpu().numpy(), true_c.cpu().numpy())
                        if sample_cap and len(sample_preds_c) < sample_cap:
                            take = min(sample_cap - len(sample_preds_c), pred_c.numel())
                            sample_preds_c.extend(pred_c[:take].cpu().numpy())
                            sample_true_c.extend(true_c[:take].cpu().numpy())

                    if edge_out is not None and edge_y is not None:
                        pred_f = edge_out.reshape(-1)
                        true_f = edge_y.reshape(-1)
                        f_stats.update(pred_f.cpu().numpy(), true_f.cpu().numpy())

        save_accuracy_metrics(
            p_stats,
            run_name,
            chlorine_stats=c_stats,
            flow_stats=f_stats if f_stats.count > 0 else None,
        )

        if seq_mode and sequence_example.get("pred_nodes") is not None:
            try:
                generate_sequence_diagnostic_plots(
                    sequence_example,
                    wn,
                    run_name,
                    plots_dir=PLOTS_DIR,
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.exception(
                    "Failed to generate sequence diagnostic plots: %s", exc
                )

        if sample_preds_p:
            preds_p = np.asarray(sample_preds_p)
            true_p = np.asarray(sample_true_p)
            if sample_preds_c:
                preds_c = np.asarray(sample_preds_c)
                true_c = np.asarray(sample_true_c)
                err_c = preds_c - true_c
            else:
                preds_c = true_c = err_c = None
            err_p = preds_p - true_p

            # gather additional node metadata from raw test features
            demand_idx_local = demand_idx
            elev_idx_export = _resolve_elevation_index(
                head_idx, elev_idx, has_chlorine
            )
            if elev_idx_export is None:
                elev_idx_export = 3 if has_chlorine else 2
            pump_start_export = pump_start
            if pump_start_export is None:
                if head_idx is not None:
                    pump_start_export = head_idx + 2
                elif elev_idx_export is not None:
                    pump_start_export = elev_idx_export + 1
                else:
                    pump_start_export = base_dim
            pump_count = len(wn.pump_name_list)
            node_indices = np.arange(len(wn.node_name_list))[node_mask_np]
            X_features = X_test_raw
            if isinstance(X_features, np.ndarray) and X_features.dtype == object and isinstance(X_features[0], dict):
                X_features = np.stack([x["node_features"] for x in X_features], axis=0)
            if seq_mode:
                X_flat = X_features.reshape(-1, X_features.shape[-2], X_features.shape[-1])
            else:
                X_flat = X_features
            demand = X_flat[..., demand_idx_local][:, node_mask_np].reshape(-1)
            elevation = X_flat[..., elev_idx_export][:, node_mask_np].reshape(-1)
            if pump_count > 0 and pump_start_export is not None:
                pumps = X_flat[..., pump_start_export : pump_start_export + pump_count][
                    :, node_mask_np, :
                ].reshape(-1, pump_count)
            else:
                pumps = np.empty((demand.size, 0))
            node_idx_flat = np.tile(node_indices, X_flat.shape[0])
            n_samples = preds_p.shape[0]
            df_dict = {
                "node_index": node_idx_flat[:n_samples],
                "elevation": elevation[:n_samples],
                "demand": demand[:n_samples],
                "actual_pressure": true_p,
                "predicted_pressure": preds_p,
            }
            for i, pname in enumerate(wn.pump_name_list):
                df_dict[f"pump_{pname}"] = pumps[:n_samples, i]
            df = pd.DataFrame(df_dict)
            pred_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(pred_csv_path, index=False)

            save_scatter_plots(
                true_p,
                preds_p,
                true_c,
                preds_c,
                run_name,
            )
            if seq_mode:
                plot_sequence_prediction(model, test_ds, run_name)
            plot_error_histograms(err_p, err_c, run_name)
            labels = ["demand", "pressure"]
            if has_chlorine:
                labels.append("chlorine")
            labels.append("elevation")
            labels += [f"pump_{i}" for i in range(len(wn.pump_name_list))]
            X_flat = X_raw.reshape(-1, X_raw.shape[-1])
            sample_size = min(sample_cap if sample_cap else X_flat.shape[0], X_flat.shape[0])
        if sample_size > 0:
            correlation_heatmap(X_flat[:sample_size], labels, run_name)

    # Generate node-level error heatmap
    heatmap_loader = None
    heatmap_dataset = None
    if test_loader is not None:
        heatmap_loader = test_loader
        heatmap_dataset = test_ds if seq_mode else test_list
    elif val_loader is not None:
        heatmap_loader = val_loader
        heatmap_dataset = val_ds if seq_mode else val_list
    else:
        heatmap_loader = loader
        heatmap_dataset = data_ds if seq_mode else data_list
    node_mae = compute_node_pressure_mae(
        model,
        heatmap_dataset,
        heatmap_loader,
        device,
        has_chlorine,
    )
    plot_error_heatmap(node_mae, wn, run_name, mask=node_mask_np)

    cfg_extra = {
        "norm_stats_md5": norm_md5,
        "model_layers": len(getattr(model, "layers", [])),
        "model_hidden_dim": getattr(getattr(model, "layers", [None])[0], "out_channels", None),
        "run_name": run_name,
    }
    save_config(REPO_ROOT / "logs" / f"config_train_{run_name}.yaml", vars(args), cfg_extra)

    if tb_writer is not None:
        tb_writer.close()
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GCN surrogate model")
    parser.add_argument(
        "--x-path",
        default=os.path.join(DATA_DIR, "X_train.npy"),
        help="Path to graph feature file",
    )
    parser.add_argument(
        "--y-path",
        default=os.path.join(DATA_DIR, "Y_train.npy"),
        help="Path to label file",
    )
    parser.add_argument(
        "--edge-index-path",
        default=os.path.join(DATA_DIR, "edge_index.npy"),
        help="Path to edge index file (used with matrix-format datasets)",
    )
    parser.add_argument(
        "--edge-attr-path",
        default=os.path.join(DATA_DIR, "edge_attr.npy"),
        help="File with edge attributes from data_generation.py",
    )
    parser.add_argument(
        "--edge-attr-train-seq-path",
        default=os.path.join(DATA_DIR, "edge_attr_train_seq.npy"),
        help="File with time-varying training edge attributes",
    )
    parser.add_argument(
        "--edge-attr-val-seq-path",
        default=os.path.join(DATA_DIR, "edge_attr_val_seq.npy"),
        help="File with time-varying validation edge attributes",
    )
    parser.add_argument(
        "--edge-attr-test-seq-path",
        default=os.path.join(DATA_DIR, "edge_attr_test_seq.npy"),
        help="File with time-varying test edge attributes",
    )
    parser.add_argument(
        "--pump-coeffs-path",
        default=os.path.join(DATA_DIR, "pump_coeffs.npy"),
        help="File with pump curve coefficients from data_generation.py",
    )
    parser.add_argument(
        "--x-val-path",
        default=os.path.join(DATA_DIR, "X_val.npy"),
        help="Validation features",
    )
    parser.add_argument(
        "--y-val-path",
        default=os.path.join(DATA_DIR, "Y_val.npy"),
        help="Validation labels",
    )
    parser.add_argument(
        "--x-test-path",
        default=os.path.join(DATA_DIR, "X_test.npy"),
        help="Test features",
    )
    parser.add_argument(
        "--y-test-path",
        default=os.path.join(DATA_DIR, "Y_test.npy"),
        help="Test labels",
    )
    parser.add_argument(
        "--pred-csv",
        default="",
        help="Path to save pressure predictions CSV",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--eval-sample",
        type=int,
        default=1000,
        help="Number of predictions to retain for evaluation plots (0 disables)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        choices=[128, 256, 512],
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument("--use-attention", action="store_true",
                        help="Use GATConv instead of HydroConv for graph convolution")
    parser.add_argument("--gat-heads", type=int, default=4,
                        help="Number of attention heads for GATConv (if attention is enabled)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate applied after each GNN layer")
    parser.add_argument("--num-layers", type=int, choices=[4, 6, 8, 10], default=4,
                        help="Number of convolutional layers in the GNN model")
    parser.add_argument(
        "--activation",
        default="relu",
        help="Activation function (relu, gelu, etc.)",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Use residual connections",
    )
    parser.add_argument(
        "--share-weights",
        action="store_true",
        help="Reuse the same convolution weights across GNN layers",
    )
    parser.add_argument(
        "--rnn-hidden-dim", "--lstm-hidden",
        dest="rnn_hidden_dim",
        type=int,
        choices=[64, 128, 256],
        default=64,
        help="Hidden dimension of the recurrent LSTM layer",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=2,
        help="Dimension of the regression target",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularization factor for optimizer",
    )
    parser.add_argument(
        "--loss-fn",
        choices=["mse", "mae", "huber"],
        default="mae",
        help="Loss function for training: mean squared error (mse), mean absolute error (mae), or Huber loss (huber).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every n epochs",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply normalization to features and targets",
    )
    parser.add_argument(
        "--per-node-norm",
        action="store_true",
        help="Compute normalization statistics for each node index separately",
    )
    parser.add_argument(
        "--physics_loss",
        dest="physics_loss",
        action="store_true",
        help="Enable mass conservation loss (default)",
    )
    parser.add_argument(
        "--no-physics-loss",
        dest="physics_loss",
        action="store_false",
        help="Disable mass conservation loss",
    )
    parser.set_defaults(physics_loss=True)
    parser.add_argument(
        "--pressure_loss",
        action="store_true",
        help="Add pressure-headloss consistency penalty",
    )
    parser.add_argument(
        "--no-pressure-loss",
        dest="pressure_loss",
        action="store_false",
        help="Disable pressure-headloss consistency penalty",
    )
    parser.set_defaults(pressure_loss=True)
    parser.add_argument(
        "--physics-loss-use-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use hydraulic head (pressure + elevation) for head-loss consistency",
    )
    parser.add_argument(
        "--pump-loss",
        action="store_true",
        help="Add pump curve consistency penalty",
    )
    parser.add_argument(
        "--no-pump-loss",
        dest="pump_loss",
        action="store_false",
        help="Disable pump curve consistency penalty",
    )
    parser.set_defaults(pump_loss=True)
    parser.add_argument(
        "--w_mass",
        type=float,
        default=2.0,
        help="Weight of the mass conservation loss term",
    )
    parser.add_argument(
        "--w_head",
        type=float,
        default=1.0,
        help="Weight of the head loss consistency term",
    )
    parser.add_argument(
        "--head-sign-weight",
        type=float,
        default=0.5,
        help="Additional weight for wrong-sign head-loss hinge penalty (0 disables)",
    )
    parser.add_argument(
        "--mass-anneal",
        type=int,
        default=0,
        help="Epochs to linearly ramp the mass loss weight from 0 to --w_mass",
    )
    parser.add_argument(
        "--head-anneal",
        type=int,
        default=0,
        help="Epochs to linearly ramp the head loss weight from 0 to --w_head",
    )
    parser.add_argument(
        "--pump-anneal",
        type=int,
        default=0,
        help="Epochs to linearly ramp the pump loss weight from 0 to --w_pump",
    )
    parser.add_argument(
        "--w_pump",
        type=float,
        default=0.25,
        help="Weight of the pump curve loss term",
    )
    parser.add_argument(
        "--w-press",
        type=float,
        default=5.0,
        help="Weight of the node pressure loss term",
    )
    parser.add_argument(
        "--w-cl",
        type=float,
        default=0.0,
        help="Weight of the node chlorine loss term",
    )
    parser.add_argument(
        "--w-flow",
        type=float,
        default=3.0,
        help="Weight of the edge (flow) loss term",
    )
    parser.add_argument(
        "--mass-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for mass conservation loss (0 = auto-estimate)",
    )
    parser.add_argument(
        "--head-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for head loss consistency (0 = auto-estimate)",
    )
    parser.add_argument(
        "--pump-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for pump curve loss (0 = auto-estimate)",
    )
    parser.add_argument(
        "--cluster-batch-size",
        type=int,
        default=0,
        help="Number of nodes per cluster for sparse training (0 disables)",
    )
    parser.add_argument(
        "--neighbor-sampling",
        action="store_true",
        help="Use random neighbor sampling instead of clustering",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision training",
    )
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Display progress bars during training",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress bars",
    )
    parser.set_defaults(progress=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", default="", help="Optional run name")
    parser.add_argument(
        "--inp-path",
        default=os.path.join(REPO_ROOT, "CTown.inp"),
        help="EPANET input file used to determine the number of pumps",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch ops",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(MODELS_DIR, "gnn_surrogate.pth"),
        help="Output model file base path",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Checkpoint path to resume training from",
    )
    args = parser.parse_args()
    main(args)


