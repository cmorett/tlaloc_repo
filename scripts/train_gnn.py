import argparse
import os
import random
from pathlib import Path
from datetime import datetime
import sys
import signal
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast, GradScaler
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
        build_edge_attr,
        build_pump_coeffs,
        build_edge_type,
        build_edge_pairs,
        build_node_type,
    )

PUMP_LOSS_WARN_THRESHOLD = 1.0


def summarize_target_norm_stats(y_mean, y_std):
    """Return scalar normalization stats for logging.

    ``y_mean`` and ``y_std`` may contain per-node statistics. This helper
    aggregates them across nodes before extracting the pressure statistics."""
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
    return pressure


def load_dataset(
    x_path: str,
    y_path: str,
    edge_index_path: str = "edge_index.npy",
    edge_attr: Optional[np.ndarray] = None,
    node_type: Optional[np.ndarray] = None,
    edge_type: Optional[np.ndarray] = None,
) -> List[Data]:
    """Load training data.

    The function supports two dataset layouts:

    1. **Dictionary format** – each element of ``X`` is a dictionary containing
       ``edge_index`` and ``node_features`` arrays.
    2. **Matrix format** – ``X`` is an array of node feature matrices while a
       shared ``edge_index`` array is stored separately.
    """

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    data_list = []
    edge_attr_tensor = None
    node_type_tensor = None
    edge_type_tensor = None
    if edge_attr is not None:
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    if node_type is not None:
        node_type_tensor = torch.tensor(node_type, dtype=torch.long)
    if edge_type is not None:
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

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


def compute_edge_attr_stats(edge_attr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std for edge attribute matrix."""
    attr_mean = torch.tensor(edge_attr.mean(axis=0), dtype=torch.float32)
    attr_std = torch.tensor(edge_attr.std(axis=0) + 1e-8, dtype=torch.float32)
    return attr_mean, attr_std


def compute_loss_scales(
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    edge_attr_phys_np: np.ndarray,
    edge_types: np.ndarray,
    pump_coeffs_np: np.ndarray,
    args: argparse.Namespace,
    seq_mode: bool,
) -> Tuple[float, float, float]:
    """Determine physics loss scales from the dataset.

    Scales are derived from dataset statistics. When the dataset lacks the
    required information (e.g., no edge flow labels or all-zero flows), the
    corresponding physics loss is disabled with a warning.
    """
    mass_scale = args.mass_scale
    head_scale = args.head_scale
    pump_scale = args.pump_scale
    if seq_mode and (
        (args.physics_loss and mass_scale <= 0)
        or (args.pressure_loss and head_scale <= 0)
        or (args.pump_loss and pump_scale <= 0)
    ):
        if args.physics_loss and mass_scale <= 0:
            demand = X_raw[..., 0]
            active = np.abs(demand) > 1e-6
            if active.any():
                mass_scale = float(np.mean(demand[active] ** 2))
            else:
                warnings.warn(
                    "No demand values found; disabling mass balance loss.",
                    UserWarning,
                )
                args.physics_loss = False
        need_edge_stats = (args.pressure_loss and head_scale <= 0) or (
            args.pump_loss and pump_scale <= 0
        )
        if need_edge_stats:
            edge_flows = None
            if isinstance(Y_raw[0], dict) or (
                isinstance(Y_raw[0], np.ndarray) and Y_raw.dtype == object
            ):
                if "edge_outputs" in Y_raw[0]:
                    edge_flows = np.stack([y["edge_outputs"] for y in Y_raw])
            else:
                edge_flows = Y_raw
            if edge_flows is None:
                if args.pressure_loss:
                    warnings.warn(
                        "No edge flow labels provided; disabling pressure loss.",
                        UserWarning,
                    )
                    args.pressure_loss = False
                if args.pump_loss:
                    warnings.warn(
                        "No edge flow labels provided; disabling pump loss.",
                        UserWarning,
                    )
                    args.pump_loss = False
            else:
                flows_flat = edge_flows.reshape(-1, edge_flows.shape[-1])
                if args.pressure_loss and head_scale <= 0:
                    pipe_mask = edge_types.flatten() == 0
                    q_pipe = flows_flat[:, pipe_mask]
                    active_edges = np.any(np.abs(q_pipe) > 1e-6, axis=0)
                    q_pipe = q_pipe[:, active_edges]
                    if q_pipe.size > 0:
                        length = edge_attr_phys_np[pipe_mask, 0][active_edges]
                        diam = np.clip(
                            edge_attr_phys_np[pipe_mask, 1][active_edges], 1e-6, None
                        )
                        rough = np.clip(
                            edge_attr_phys_np[pipe_mask, 2][active_edges], 1e-6, None
                        )
                        q_m3 = np.abs(q_pipe) * 0.001
                        denom = np.clip(rough ** 1.852 * diam ** 4.87, 1e-6, None)
                        hw_hl = 10.67 * length * (q_m3 ** 1.852) / denom
                        if hw_hl.size > 0:
                            head_scale = float(np.mean(hw_hl ** 2))
                    else:
                        warnings.warn(
                            "No informative pipe flows found; disabling pressure loss.",
                            UserWarning,
                        )
                        args.pressure_loss = False
                if args.pump_loss and pump_scale <= 0:
                    pump_mask = edge_types.flatten() == 1
                    q_pump = flows_flat[:, pump_mask]
                    if q_pump.size > 0:
                        coeff = pump_coeffs_np[pump_mask]
                        a = coeff[:, 0]
                        b = coeff[:, 1]
                        c = coeff[:, 2]
                        head = a - b * np.abs(q_pump) ** c
                        if head.size > 0:
                            pump_scale = float(np.mean(np.abs(head)))
                    else:
                        warnings.warn(
                            "No informative pump flows found; disabling pump loss.",
                            UserWarning,
                        )
                        args.pump_loss = False
    MIN_SCALE = 1.0
    if args.physics_loss:
        mass_scale = max(mass_scale, MIN_SCALE)
    if args.pressure_loss:
        head_scale = max(head_scale, MIN_SCALE)
    if args.pump_loss:
        pump_scale = max(pump_scale, MIN_SCALE)
    args.mass_scale = mass_scale
    args.head_scale = head_scale
    args.pump_scale = pump_scale
    return mass_scale, head_scale, pump_scale


@dataclass
class RunningStats:
    """Accumulate error statistics without storing full arrays."""

    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    abs_pct_sum: float = 0.0
    max_err: float = 0.0

    def update(self, pred, true) -> None:
        p = np.asarray(pred, dtype=float)
        t = np.asarray(true, dtype=float)
        diff = p - t
        abs_err = np.abs(diff)
        self.count += abs_err.size
        self.abs_sum += float(abs_err.sum())
        self.sq_sum += float(np.square(diff).sum())
        denom = np.maximum(np.abs(t), 1e-8)
        self.abs_pct_sum += float((abs_err / denom).sum())
        if abs_err.size:
            self.max_err = max(self.max_err, float(abs_err.max()))

    def metrics(self) -> List[float]:
        if self.count == 0:
            return [float("nan")] * 4
        mae = self.abs_sum / self.count
        rmse = np.sqrt(self.sq_sum / self.count)
        mape = (self.abs_pct_sum / self.count) * 100.0
        return [mae, rmse, mape, self.max_err]



def save_accuracy_metrics(
    pressure_stats: RunningStats,
    run_name: str,
    logs_dir: Optional[Path] = None,
) -> None:
    """Compute and export accuracy metrics to a CSV file."""
    if logs_dir is None:
        logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = {"Pressure (m)": pressure_stats.metrics()}

    index = [
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Percentage Error",
        "Maximum Error",
    ]
    df = pd.DataFrame(data, index=index)
    export_table(df, str(logs_dir / f"accuracy_{run_name}.csv"))



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
    labels = ["pressure", "flow", "mass", "sym"]
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


def pred_vs_actual_scatter(
    pred: Sequence[float],
    true: Sequence[float],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot predicted vs actual pressures as a scatter plot."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(t, p, s=5, alpha=0.5)
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Actual Pressure (m)")
    ax.set_ylabel("Predicted Pressure (m)")
    ax.set_title("Pressure: Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(plots_dir / f"pred_vs_actual_pressure_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_error_histogram(
    errors: Sequence[float],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot histogram and box plot of prediction errors."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    err = np.asarray(errors, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(err, bins=50, color="tab:blue", alpha=0.7)
    axes[0].set_title("Pressure Error")
    axes[1].boxplot(err, vert=True)
    axes[1].set_title("Pressure Error")
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"error_histograms_{run_name}.png")
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

    X_seq, Y_seq = dataset[0]
    X_seq = X_seq.unsqueeze(0).to(device)
    ei = dataset.edge_index.to(device)
    ea = dataset.edge_attr.to(device) if dataset.edge_attr is not None else None
    nt = dataset.node_type.to(device) if dataset.node_type is not None else None
    et = dataset.edge_type.to(device) if dataset.edge_type is not None else None

    with torch.no_grad():
        out = model(X_seq, ei, ea, nt, et)

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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, true_np[:, node_idx, 0], label="Actual")
    ax.plot(time, pred_np[:, node_idx, 0], "--", label="Predicted")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Pressure (m)")
    ax.set_title(f"Node {node_idx} Pressure")
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / f"time_series_example_{run_name}.png")
    plt.close(fig)




def build_loss_mask(wn: wntr.network.WaterNetworkModel) -> torch.Tensor:
    """Return boolean mask marking nodes included in the loss."""

    mask = torch.ones(len(wn.node_name_list), dtype=torch.bool)
    for i, name in enumerate(wn.node_name_list):
        if name in wn.reservoir_name_list or name in wn.tank_name_list:
            mask[i] = False
    return mask


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
    w_press: float = 5.0,
    w_flow: float = 3.0,
):
    model.train()
    scaler = GradScaler(device=device.type, enabled=amp)
    total_loss = press_total = flow_total = 0.0
    for batch in tqdm(loader, disable=not progress):
        batch = batch.to(device, non_blocking=True)
        if torch.isnan(batch.x).any() or torch.isnan(batch.y).any():
            raise ValueError("NaN detected in training batch")
        if check_negative and ((batch.x[:, 1] < 0).any() or (batch.y[:, 0] < 0).any()):
            raise ValueError("Negative pressures encountered in training batch")
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=amp):
            out = model(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                getattr(batch, "node_type", None),
                getattr(batch, "edge_type", None),
            )
            if isinstance(out, dict):
                pred_nodes = out["node_outputs"].float()
                edge_pred = out.get("edge_outputs")
                target_nodes = batch.y.float()
                edge_target = getattr(batch, "edge_y", None)
                if node_mask is not None:
                    repeat = pred_nodes.size(0) // node_mask.numel()
                    mask = node_mask.repeat(repeat)
                    pred_nodes = pred_nodes[mask]
                    target_nodes = target_nodes[mask]
                use_flow_loss = edge_target is not None and w_flow != 0
                if use_flow_loss:
                    edge_target = edge_target.float()
                    loss, press_l, flow_l = weighted_mtl_loss(
                        pred_nodes,
                        target_nodes,
                        edge_pred.float(),
                        edge_target,
                        loss_fn=loss_fn,
                        w_press=w_press,
                        w_flow=w_flow,
                    )
                else:
                    loss_press = _apply_loss(
                        pred_nodes[..., 0], target_nodes[..., 0], loss_fn
                    )
                    loss = w_press * loss_press
                    press_l = loss_press
                    flow_l = torch.tensor(0.0, device=device)
            else:
                out_t = out
                target = batch.y.float()
                if node_mask is not None:
                    repeat = out_t.size(0) // node_mask.numel()
                    mask = node_mask.repeat(repeat)
                    out_t = out_t[mask]
                    target = target[mask]
                loss = _apply_loss(out_t, target, loss_fn)
                press_l = flow_l = torch.tensor(0.0, device=device)
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        # Clip gradients to mitigate exploding gradients that could otherwise
        # result in ``NaN`` loss values when the optimizer updates the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        press_total += press_l.item() * batch.num_graphs
        flow_total += flow_l.item() * batch.num_graphs
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        press_total / denom,
        flow_total / denom,
    )


def evaluate(
    model,
    loader,
    device,
    amp: bool = False,
    loss_fn: str = "mae",
    node_mask: Optional[torch.Tensor] = None,
    progress: bool = True,
    w_press: float = 3.0,
    w_flow: float = 1.0,
):
    global interrupted
    model.eval()
    total_loss = press_total = flow_total = 0.0
    data_iter = iter(tqdm(loader, disable=not progress))
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
            with autocast(device_type=device.type, enabled=amp):
                out = model(
                    batch.x,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    getattr(batch, "node_type", None),
                    getattr(batch, "edge_type", None),
                )
                if isinstance(out, dict):
                    pred_nodes = out["node_outputs"].float()
                    edge_pred = out.get("edge_outputs")
                    target_nodes = batch.y.float()
                    edge_target = getattr(batch, "edge_y", None)
                    if node_mask is not None:
                        repeat = pred_nodes.size(0) // node_mask.numel()
                        mask = node_mask.repeat(repeat)
                        pred_nodes = pred_nodes[mask]
                        target_nodes = target_nodes[mask]
                    use_flow_loss = edge_target is not None and w_flow != 0
                    if use_flow_loss:
                        edge_target = edge_target.float()
                        loss, press_l, flow_l = weighted_mtl_loss(
                            pred_nodes,
                            target_nodes,
                            edge_pred.float(),
                            edge_target,
                            loss_fn=loss_fn,
                            w_press=w_press,
                            w_flow=w_flow,
                        )
                    else:
                        loss_press = _apply_loss(
                            pred_nodes[..., 0], target_nodes[..., 0], loss_fn
                        )
                        loss = w_press * loss_press
                        press_l = loss_press
                        flow_l = torch.tensor(0.0, device=device)
                else:
                    out_t = out
                    target = batch.y.float()
                    if node_mask is not None:
                        repeat = out_t.size(0) // node_mask.numel()
                        mask = node_mask.repeat(repeat)
                        out_t = out_t[mask]
                        target = target[mask]
                    loss = _apply_loss(out_t, target, loss_fn)
                    press_l = flow_l = torch.tensor(0.0, device=device)
            total_loss += loss.item() * batch.num_graphs
            press_total += press_l.item() * batch.num_graphs
            flow_total += flow_l.item() * batch.num_graphs
            if interrupted:
                break
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        press_total / denom,
        flow_total / denom,
    )


def train_sequence(
    model: nn.Module,
    loader: TorchLoader,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
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
    mass_scale: float = 1.0,
    head_scale: float = 1.0,
    pump_scale: float = 1.0,
    w_mass: float = 2.0,
    w_head: float = 1.0,
    w_pump: float = 1.0,
    w_press: float = 3.0,
    w_flow: float = 1.0,
    amp: bool = False,
    progress: bool = True,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    global interrupted
    model.train()
    scaler = GradScaler(device=device.type, enabled=amp)
    total_loss = 0.0
    press_total = flow_total = 0.0
    mass_total = head_total = sym_total = pump_total = 0.0
    mass_imb_total = head_viol_total = 0.0
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
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
            X_seq, Y_seq = next(data_iter)
        except StopIteration:
            break
        except RuntimeError as e:
            if interrupted and "DataLoader worker" in str(e):
                break
            raise
        X_seq = X_seq.to(device)
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
            init_levels = init_press * model.tank_areas
            model.reset_tank_levels(init_levels)
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=amp):
            preds = model(
                X_seq,
                edge_index,
                edge_attr,
                nt,
                et,
            )
        if isinstance(Y_seq, dict):
            target_nodes = Y_seq['node_outputs'].to(device)
            pred_nodes = preds['node_outputs'].float()
            if node_mask is not None:
                pred_nodes = pred_nodes[:, :, node_mask, :]
                target_nodes = target_nodes[:, :, node_mask, :]
            edge_preds = preds.get('edge_outputs')
            edge_target = Y_seq.get('edge_outputs')
            use_flow_loss = edge_target is not None and w_flow != 0
            if use_flow_loss:
                edge_target = edge_target.unsqueeze(-1).to(device)
                loss, loss_press, loss_edge = weighted_mtl_loss(
                    pred_nodes,
                    target_nodes.float(),
                    edge_preds.float(),
                    edge_target.float(),
                    loss_fn=loss_fn,
                    w_press=w_press,
                    w_flow=w_flow,
                )
            else:
                loss_press = _apply_loss(
                    pred_nodes[..., 0], target_nodes[..., 0], loss_fn
                )
                loss = w_press * loss_press
                loss_edge = torch.tensor(0.0, device=device)
            for name, val in [
                ("pressure", loss_press),
                ("flow", loss_edge),
            ]:
                if (not torch.isfinite(val)) or val.item() > 1e6:
                    raise AssertionError(f"{name} loss {val.item():.3e} invalid")
            if physics_loss:
                flows_mb = edge_preds.squeeze(-1)
                if getattr(model, "y_mean_edge", None) is not None:
                    q_mean = model.y_mean_edge.to(device)
                    q_std = model.y_std_edge.to(device)
                    flows_mb = (
                        flows_mb * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                    )
                elif isinstance(getattr(model, "y_mean", None), dict):
                    q_mean = model.y_mean["edge_outputs"].to(device)
                    q_std = model.y_std["edge_outputs"].to(device)
                    flows_mb = (
                        flows_mb * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                    )
                flows_mb = (
                    flows_mb.permute(2, 0, 1)
                    .reshape(edge_index.size(1), -1)
                )
                dem_seq = X_seq[..., 0]
                demand_mb = dem_seq.permute(2, 0, 1).reshape(node_count, -1)
                if hasattr(model, "x_mean") and model.x_mean is not None:
                    if model.x_mean.ndim == 2:
                        dem_mean = model.x_mean[:, 0].to(device).unsqueeze(1)
                        dem_std = model.x_std[:, 0].to(device).unsqueeze(1)
                    else:
                        dem_mean = model.x_mean[0].to(device)
                        dem_std = model.x_std[0].to(device)
                    demand_mb = demand_mb * dem_std + dem_mean
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
            if pressure_loss:
                press = preds['node_outputs'][..., 0].float()
                flow = edge_preds.squeeze(-1)
                if isinstance(getattr(model, 'y_mean', None), dict):
                    p_mean = model.y_mean['node_outputs'].to(device)
                    p_std = model.y_std['node_outputs'].to(device)
                    if p_mean.ndim == 2:
                        p_mean = p_mean[..., 0]
                        p_std = p_std[..., 0]
                    press = press * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
                    if 'edge_outputs' in model.y_mean:
                        q_mean = model.y_mean['edge_outputs'].to(device)
                        q_std = model.y_std['edge_outputs'].to(device)
                        flow = flow * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                elif getattr(model, 'y_mean', None) is not None:
                    p_mean = model.y_mean.to(device)
                    p_std = model.y_std.to(device)
                    if p_mean.ndim == 2:
                        p_mean = p_mean[..., 0]
                        p_std = p_std[..., 0]
                    press = press * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
                    if getattr(model, 'y_mean_edge', None) is not None:
                        q_mean = model.y_mean_edge.to(device)
                        q_std = model.y_std_edge.to(device)
                        flow = flow * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                head_loss, head_violation = pressure_headloss_consistency_loss(
                    press,
                    flow,
                    edge_index,
                    edge_attr_phys,
                    edge_type=et,
                    return_violation=True,
                )
            else:
                head_loss = torch.tensor(0.0, device=device)
                head_violation = torch.tensor(0.0, device=device)
            if pump_loss and pump_coeffs is not None:
                flow_pc = edge_preds.squeeze(-1)
                if getattr(model, 'y_mean_edge', None) is not None:
                    q_mean = model.y_mean_edge.to(device)
                    q_std = model.y_std_edge.to(device)
                    flow_pc = flow_pc * q_std + q_mean
                elif isinstance(getattr(model, 'y_mean', None), dict):
                    q_mean = model.y_mean['edge_outputs'].to(device)
                    q_std = model.y_std['edge_outputs'].to(device)
                    flow_pc = flow_pc * q_std + q_mean
                pump_loss_val = pump_curve_loss(
                    flow_pc,
                    pump_coeffs,
                    edge_index,
                    et,
                )
            else:
                pump_loss_val = torch.tensor(0.0, device=device)
            mass_loss, head_loss, pump_loss_val = scale_physics_losses(
                mass_loss,
                head_loss,
                pump_loss_val,
                mass_scale=mass_scale,
                head_scale=head_scale,
                pump_scale=pump_scale,
            )
            if mass_scale > 0:
                sym_loss = sym_loss / mass_scale
            if physics_loss:
                loss = loss + w_mass * (mass_loss + sym_loss)
            if pressure_loss:
                loss = loss + w_head * head_loss
            if pump_loss:
                loss = loss + w_pump * pump_loss_val
        else:
            Y_seq = Y_seq.to(device)
            loss_press = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
            head_loss = pump_loss_val = torch.tensor(0.0, device=device)
            mass_imb = head_violation = torch.tensor(0.0, device=device)
            with autocast(device_type=device.type, enabled=amp):
                loss = _apply_loss(preds, Y_seq.float(), loss_fn)
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * X_seq.size(0)
        press_total += loss_press.item() * X_seq.size(0)
        flow_total += loss_edge.item() * X_seq.size(0)
        mass_total += mass_loss.item() * X_seq.size(0)
        head_total += head_loss.item() * X_seq.size(0)
        sym_total += sym_loss.item() * X_seq.size(0)
        pump_total += pump_loss_val.item() * X_seq.size(0)
        mass_imb_total += mass_imb.item() * X_seq.size(0)
        head_viol_total += head_violation.item() * X_seq.size(0)
        if interrupted:
            break
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        press_total / denom,
        flow_total / denom,
        mass_total / denom,
        head_total / denom,
        sym_total / denom,
        pump_total / denom,
        mass_imb_total / denom,
        head_viol_total / denom,
    )


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
    mass_scale: float = 1.0,
    head_scale: float = 1.0,
    pump_scale: float = 1.0,
    w_mass: float = 2.0,
    w_head: float = 1.0,
    w_pump: float = 1.0,
    w_press: float = 3.0,
    w_flow: float = 1.0,
    amp: bool = False,
    progress: bool = True,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    global interrupted
    model.eval()
    total_loss = 0.0
    press_total = flow_total = 0.0
    mass_total = head_total = sym_total = pump_total = 0.0
    mass_imb_total = head_viol_total = 0.0
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    edge_attr_phys = edge_attr_phys.to(device)
    if node_type is not None:
        node_type = node_type.to(device)
    if edge_type is not None:
        edge_type = edge_type.to(device)
    if pump_coeffs is not None:
        pump_coeffs = pump_coeffs.to(device)
    node_count = int(edge_index.max()) + 1
    data_iter = iter(tqdm(loader, disable=not progress))
    with torch.no_grad():
        while True:
            try:
                X_seq, Y_seq = next(data_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                if interrupted and "DataLoader worker" in str(e):
                    break
                raise
            X_seq = X_seq.to(device)
            nt = node_type
            et = edge_type
            if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
                init_press = X_seq[:, 0, model.tank_indices, 1]
                init_levels = init_press * model.tank_areas
                model.reset_tank_levels(init_levels)
            with autocast(device_type=device.type, enabled=amp):
                preds = model(
                    X_seq,
                    edge_index,
                    edge_attr,
                    nt,
                    et,
                )
                if isinstance(Y_seq, dict):
                    target_nodes = Y_seq['node_outputs'].to(device)
                    pred_nodes = preds['node_outputs'].float()
                    if node_mask is not None:
                        pred_nodes = pred_nodes[:, :, node_mask, :]
                        target_nodes = target_nodes[:, :, node_mask, :]
                    edge_preds = preds.get('edge_outputs')
                    edge_target = Y_seq.get('edge_outputs')
                    use_flow_loss = edge_target is not None and w_flow != 0
                    if use_flow_loss:
                        edge_target = edge_target.unsqueeze(-1).to(device)
                        loss, loss_press, loss_edge = weighted_mtl_loss(
                            pred_nodes,
                            target_nodes.float(),
                            edge_preds.float(),
                            edge_target.float(),
                            loss_fn=loss_fn,
                            w_press=w_press,
                            w_flow=w_flow,
                        )
                    else:
                        loss_press = _apply_loss(
                            pred_nodes[..., 0], target_nodes[..., 0], loss_fn
                        )
                        loss = w_press * loss_press
                        loss_edge = torch.tensor(0.0, device=device)
                    if physics_loss:
                        flows_mb = edge_preds.squeeze(-1)
                        if getattr(model, "y_mean_edge", None) is not None:
                            q_mean = model.y_mean_edge.to(device)
                            q_std = model.y_std_edge.to(device)
                            flows_mb = (
                                flows_mb * q_std.view(1, 1, -1)
                                + q_mean.view(1, 1, -1)
                            )
                        elif isinstance(getattr(model, "y_mean", None), dict):
                            q_mean = model.y_mean["edge_outputs"].to(device)
                            q_std = model.y_std["edge_outputs"].to(device)
                            flows_mb = (
                                flows_mb * q_std.view(1, 1, -1)
                                + q_mean.view(1, 1, -1)
                            )
                        flows_mb = (
                            flows_mb.permute(2, 0, 1)
                            .reshape(edge_index.size(1), -1)
                        )
                        dem_seq = X_seq[..., 0]
                        demand_mb = dem_seq.permute(2, 0, 1).reshape(node_count, -1)
                        if hasattr(model, "x_mean") and model.x_mean is not None:
                            if model.x_mean.ndim == 2:
                                dem_mean = model.x_mean[:, 0].to(device).unsqueeze(1)
                                dem_std = model.x_std[:, 0].to(device).unsqueeze(1)
                            else:
                                dem_mean = model.x_mean[0].to(device)
                                dem_std = model.x_std[0].to(device)
                            demand_mb = demand_mb * dem_std + dem_mean
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
                    if pressure_loss:
                        press = preds['node_outputs'][..., 0].float()
                        flow = edge_preds.squeeze(-1)
                        if isinstance(getattr(model, 'y_mean', None), dict):
                            p_mean = model.y_mean['node_outputs'].to(device)
                            p_std = model.y_std['node_outputs'].to(device)
                            if p_mean.ndim == 2:
                                p_mean = p_mean[..., 0]
                                p_std = p_std[..., 0]
                            press = press * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
                            if 'edge_outputs' in model.y_mean:
                                q_mean = model.y_mean['edge_outputs'].to(device)
                                q_std = model.y_std['edge_outputs'].to(device)
                                flow = flow * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                        elif getattr(model, 'y_mean', None) is not None:
                            p_mean = model.y_mean.to(device)
                            p_std = model.y_std.to(device)
                            if p_mean.ndim == 2:
                                p_mean = p_mean[..., 0]
                                p_std = p_std[..., 0]
                            press = press * p_std.view(1, 1, -1) + p_mean.view(1, 1, -1)
                            if getattr(model, 'y_mean_edge', None) is not None:
                                q_mean = model.y_mean_edge.to(device)
                                q_std = model.y_std_edge.to(device)
                                flow = flow * q_std.view(1, 1, -1) + q_mean.view(1, 1, -1)
                        head_loss, head_violation = pressure_headloss_consistency_loss(
                            press,
                            flow,
                            edge_index,
                            edge_attr_phys,
                            edge_type=et,
                            return_violation=True,
                        )
                    else:
                        head_loss = torch.tensor(0.0, device=device)
                        head_violation = torch.tensor(0.0, device=device)
                    if pump_loss and pump_coeffs is not None:
                        flow_pc = edge_preds.squeeze(-1)
                        if getattr(model, 'y_mean_edge', None) is not None:
                            q_mean = model.y_mean_edge.to(device)
                            q_std = model.y_std_edge.to(device)
                            flow_pc = flow_pc * q_std + q_mean
                        elif isinstance(getattr(model, 'y_mean', None), dict):
                            q_mean = model.y_mean['edge_outputs'].to(device)
                            q_std = model.y_std['edge_outputs'].to(device)
                            flow_pc = flow_pc * q_std + q_mean
                        pump_loss_val = pump_curve_loss(
                            flow_pc,
                            pump_coeffs,
                            edge_index,
                            et,
                        )
                    else:
                        pump_loss_val = torch.tensor(0.0, device=device)
                    mass_loss, head_loss, pump_loss_val = scale_physics_losses(
                        mass_loss,
                        head_loss,
                        pump_loss_val,
                        mass_scale=mass_scale,
                        head_scale=head_scale,
                        pump_scale=pump_scale,
                    )
                    if mass_scale > 0:
                        sym_loss = sym_loss / mass_scale
                    if physics_loss:
                        loss = loss + w_mass * (mass_loss + sym_loss)
                    if pressure_loss:
                        loss = loss + w_head * head_loss
                    if pump_loss:
                        loss = loss + w_pump * pump_loss_val
                else:
                    Y_seq = Y_seq.to(device)
                    loss_press = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
                    head_loss = pump_loss_val = torch.tensor(0.0, device=device)
                    mass_imb = head_violation = torch.tensor(0.0, device=device)
                    loss = _apply_loss(preds, Y_seq.float(), loss_fn)
            total_loss += loss.item() * X_seq.size(0)
            press_total += loss_press.item() * X_seq.size(0)
            flow_total += loss_edge.item() * X_seq.size(0)
            mass_total += mass_loss.item() * X_seq.size(0)
            head_total += head_loss.item() * X_seq.size(0)
            sym_total += sym_loss.item() * X_seq.size(0)
            pump_total += pump_loss_val.item() * X_seq.size(0)
            mass_imb_total += mass_imb.item() * X_seq.size(0)
            head_viol_total += head_violation.item() * X_seq.size(0)
            if interrupted:
                break
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        press_total / denom,
        flow_total / denom,
        mass_total / denom,
        head_total / denom,
        sym_total / denom,
        pump_total / denom,
        mass_imb_total / denom,
        head_viol_total / denom,
    )

# Resolve important directories relative to the repository root so that training
# can be launched from any location.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
PLOTS_DIR = REPO_ROOT / "plots"


def main(args: argparse.Namespace):
    configure_seeds(args.seed, args.deterministic)
    if args.w_flow == 0 and (
        args.physics_loss or args.pressure_loss or args.pump_loss
    ):
        warnings.warn(
            "Flow loss weight is zero; physics losses will be applied without flow MAE regularization.",
            RuntimeWarning,
        )
    signal.signal(signal.SIGINT, _signal_handler)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index_np = np.load(args.edge_index_path)
    wn = wntr.network.WaterNetworkModel(args.inp_path)
    # Always compute the physical edge attributes from the network
    edge_attr_phys_np = build_edge_attr(wn, edge_index_np)
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
    edge_types = build_edge_type(wn, edge_index_np)
    edge_pairs = build_edge_pairs(edge_index_np, edge_types)
    node_types = build_node_type(wn)
    loss_mask = build_loss_mask(wn).to(device)
    # Always allocate a distinct node type for tanks even if they are absent
    # from the network to ensure ``HydroConv`` learns a dedicated transform.
    num_node_types = max(int(np.max(node_types)) + 1, 2)
    num_edge_types = int(np.max(edge_types)) + 1
    edge_mean, edge_std = compute_edge_attr_stats(edge_attr)
    X_raw = np.load(args.x_path, allow_pickle=True)
    Y_raw = np.load(args.y_path, allow_pickle=True)
    seq_mode = X_raw.ndim == 4

    if seq_mode:
        data_ds = SequenceDataset(
            X_raw,
            Y_raw,
            edge_index_np,
            edge_attr,
            node_type=node_types,
            edge_type=edge_types,
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
        )
        loader = DataLoader(
            data_list,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )


    if args.x_val_path and os.path.exists(args.x_val_path):
        if seq_mode:
            Xv = np.load(args.x_val_path, allow_pickle=True)
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
            val_list = load_dataset(
                args.x_val_path,
                args.y_val_path,
                args.edge_index_path,
                edge_attr=edge_attr,
                node_type=node_types,
                edge_type=edge_types,
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
    else:
        sample_dim = data_list[0].num_node_features
    base_dim = sample_dim - pump_count
    if base_dim != 3:
        raise ValueError(
            f"Dataset provides {sample_dim} features per node but the network has {pump_count} pumps."
        )
    args.output_dim = 1

    norm_md5 = None
    if args.normalize:
        if seq_mode:
            x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(
                X_raw, Y_raw, per_node=args.per_node_norm
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
                )
        else:
            x_mean, x_std, y_mean, y_std = compute_norm_stats(
                data_list, per_node=args.per_node_norm
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
                )
        print("Target normalization stats:")
        pressure_stats = summarize_target_norm_stats(y_mean, y_std)
        print("Pressure mean/std:", *pressure_stats)

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
            node_mean = y_mean.get("node_outputs")
            node_std = y_std.get("node_outputs")
            norm_stats["y_mean_node"] = node_mean.to(torch.float32).cpu().numpy()
            norm_stats["y_std_node"] = node_std.to(torch.float32).cpu().numpy()
            edge_mean_t = y_mean.get("edge_outputs")
            edge_std_t = y_std.get("edge_outputs")
            if edge_mean_t is not None and edge_std_t is not None:
                norm_stats["y_mean_edge"] = edge_mean_t.to(torch.float32).cpu().numpy()
                norm_stats["y_std_edge"] = edge_std_t.to(torch.float32).cpu().numpy()
            # Backward compatibility for older checkpoints
            norm_stats["y_mean"] = norm_stats["y_mean_node"]
            norm_stats["y_std"] = norm_stats["y_std_node"]
        else:
            norm_stats["y_mean_node"] = y_mean.to(torch.float32).cpu().numpy()
            norm_stats["y_std_node"] = y_std.to(torch.float32).cpu().numpy()
            norm_stats["y_mean"] = norm_stats["y_mean_node"]
            norm_stats["y_std"] = norm_stats["y_std_node"]
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

    expected_in_dim = 3 + len(wn.pump_name_list)

    if seq_mode:
        sample_dim = data_ds.X.shape[-1]
        if sample_dim != expected_in_dim:
            raise ValueError(
                f"Dataset provides {sample_dim} features per node but the network "
                f"defines {len(wn.pump_name_list)} pumps (expected {expected_in_dim}).\n"
                "Re-generate the training data with pump control inputs using scripts/data_generation.py."
            )
        if getattr(data_ds, "multi", False):
            model = MultiTaskGNNSurrogate(
                in_channels=sample_dim,
                hidden_channels=args.hidden_dim,
                edge_dim=edge_attr.shape[1],
                node_output_dim=1,
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
    else:
        sample = data_list[0]
        if sample.num_node_features != expected_in_dim:
            raise ValueError(
                f"Dataset provides {sample.num_node_features} features per node but the network "
                f"defines {len(wn.pump_name_list)} pumps (expected {expected_in_dim}).\n"
                "Re-generate the training data with pump control inputs using scripts/data_generation.py."
            )
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
    }
    if norm_md5 is not None:
        model_meta["norm_stats_md5"] = norm_md5

    # expose normalization stats on the model for later un-normalisation
    if args.normalize:
        if isinstance(y_mean, dict):
            model.y_mean = y_mean.get("node_outputs")
            model.y_std = y_std.get("node_outputs")
            model.y_mean_edge = y_mean.get("edge_outputs")
            model.y_std_edge = y_std.get("edge_outputs")
        else:
            model.y_mean = y_mean
            model.y_std = y_std
            model.y_mean_edge = model.y_std_edge = None
        model.x_mean = x_mean
        model.x_std = x_std
    else:
        model.x_mean = model.x_std = model.y_mean = model.y_std = None
        model.y_mean_edge = model.y_std_edge = None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

    mass_scale, head_scale, pump_scale = compute_loss_scales(
        X_raw,
        Y_raw,
        edge_attr_phys_np,
        edge_types,
        pump_coeffs_np,
        args,
        seq_mode,
    )
    if seq_mode and (args.physics_loss or args.pressure_loss or args.pump_loss):
        print(
            f"Using physics loss scales: mass={mass_scale:.4e}, head={head_scale:.4e}, pump={pump_scale:.4e}"
        )

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
    losses = []
    loss_components = []
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
            f.write(
                "epoch,train_loss,val_loss,press_loss,flow_loss,mass_imbalance,head_violation,val_press_loss,val_flow_loss,val_mass_imbalance,val_head_violation,lr\n"
            )
        else:
            f.write(
                "epoch,train_loss,val_loss,press_loss,flow_loss,val_press_loss,val_flow_loss,lr\n"
            )
        best_val = float("inf")
        patience = 0
        for epoch in range(start_epoch, args.epochs):
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
                    w_mass=args.w_mass,
                    w_head=args.w_head,
                    w_pump=args.w_pump,
                    w_press=args.w_press,
                    w_flow=args.w_flow,
                    amp=args.amp,
                    progress=args.progress,
                )
                loss = loss_tuple[0]
                press_l, flow_l, mass_l, head_l, sym_l, pump_l, mass_imb, head_viols = loss_tuple[1:]
                comp = [press_l, flow_l, mass_l, sym_l]
                if args.pressure_loss:
                    comp.append(head_l)
                if args.pump_loss:
                    comp.append(pump_l)
                loss_components.append(tuple(comp))
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
                        w_mass=args.w_mass,
                        w_head=args.w_head,
                        w_pump=args.w_pump,
                        w_press=args.w_press,
                        w_flow=args.w_flow,
                        amp=args.amp,
                        progress=args.progress,
                    )
                    val_loss = val_tuple[0]
                    val_press_l = val_tuple[1]
                    val_flow_l = val_tuple[2]
                    val_mass_imb = val_tuple[7]
                    val_head_viols = val_tuple[8]
                else:
                    val_loss = loss
                    val_press_l, val_flow_l = press_l, flow_l
                    val_mass_imb, val_head_viols = mass_imb, head_viols
            else:
                loss, press_l, flow_l = train(
                    model,
                    loader,
                    optimizer,
                    device,
                    check_negative=not args.normalize,
                    amp=args.amp,
                    loss_fn=args.loss_fn,
                    node_mask=loss_mask,
                    progress=args.progress,
                    w_press=args.w_press,
                    w_flow=args.w_flow,
                )
                loss_components.append((press_l, flow_l))
                if val_loader is not None and not interrupted:
                    val_loss, val_press_l, val_flow_l = evaluate(
                        model,
                        val_loader,
                        device,
                        amp=args.amp,
                        loss_fn=args.loss_fn,
                        node_mask=loss_mask,
                        progress=args.progress,
                        w_press=args.w_press,
                        w_flow=args.w_flow,
                    )
                else:
                    val_loss = loss
                    val_press_l, val_flow_l = press_l, flow_l
            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            losses.append((loss, val_loss))
            if seq_mode:
                f.write(
                    f"{epoch},{loss:.6f},{val_loss:.6f},{press_l:.6f},{flow_l:.6f},{mass_imb:.6f},{head_viols:.6f},"
                    f"{val_press_l:.6f},{val_flow_l:.6f},{val_mass_imb:.6f},{val_head_viols:.6f},{curr_lr:.6e}\n"
                )
                if tb_writer is not None:
                    tb_writer.add_scalars(
                        "loss/train",
                        {
                            "total": loss,
                            "pressure": press_l,
                            "flow": flow_l,
                        },
                        epoch,
                    )
                    tb_writer.add_scalars(
                        "loss/val",
                        {
                            "total": val_loss,
                            "pressure": val_press_l,
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
                if args.physics_loss:
                    msg = (
                        f"Epoch {epoch}: press={press_l:.3f}, flow={flow_l:.3f}, "
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
                    print(msg)
                else:
                    print(f"Epoch {epoch}")
            else:
                f.write(
                    f"{epoch},{loss:.6f},{val_loss:.6f},{press_l:.6f},{flow_l:.6f},"
                    f"{val_press_l:.6f},{val_flow_l:.6f},{curr_lr:.6e}\n"
                )
                if tb_writer is not None:
                    tb_writer.add_scalars(
                        "loss/train",
                        {
                            "total": loss,
                            "pressure": press_l,
                            "flow": flow_l,
                        },
                        epoch,
                    )
                    tb_writer.add_scalars(
                        "loss/val",
                        {
                            "total": val_loss,
                            "pressure": val_press_l,
                            "flow": val_flow_l,
                        },
                        epoch,
                    )
                print(
                    f"Epoch {epoch}: press={press_l:.3f}, flow={flow_l:.3f}"
                )
            if val_loss < best_val - 1e-6:
                best_val = val_loss
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
                            "y_mean_node": norm_stats["y_mean_node"],
                            "y_std_node": norm_stats["y_std_node"],
                            "edge_mean": norm_stats["edge_mean"],
                            "edge_std": norm_stats["edge_std"],
                        }
                    )
                    if "y_mean_edge" in norm_stats:
                        meta.update(
                            {
                                "y_mean_edge": norm_stats["y_mean_edge"],
                                "y_std_edge": norm_stats["y_std_edge"],
                            }
                        )
                    # backward compatibility
                    meta["y_mean"] = norm_stats["y_mean_node"]
                    meta["y_std"] = norm_stats["y_std_node"]
                ckpt["model_meta"] = meta
                torch.save(ckpt, model_path)
                if norm_stats is not None:
                    if "y_mean_edge" in norm_stats:
                        np.savez(
                            norm_path,
                            x_mean=norm_stats["x_mean"],
                            x_std=norm_stats["x_std"],
                            y_mean_node=norm_stats["y_mean_node"],
                            y_std_node=norm_stats["y_std_node"],
                            y_mean_edge=norm_stats["y_mean_edge"],
                            y_std_edge=norm_stats["y_std_edge"],
                            edge_mean=norm_stats["edge_mean"],
                            edge_std=norm_stats["edge_std"],
                        )
                    else:
                        np.savez(
                            norm_path,
                            x_mean=norm_stats["x_mean"],
                            x_std=norm_stats["x_std"],
                            y_mean=norm_stats["y_mean_node"],
                            y_std=norm_stats["y_std_node"],
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

    # scatter plot of predictions vs actual on test set
    if args.x_test_path and os.path.exists(args.x_test_path):
        if seq_mode:
            Xt = np.load(args.x_test_path, allow_pickle=True)
            Yt = np.load(args.y_test_path, allow_pickle=True)
            if Xt.ndim == 3:
                Yt = np.array([{k: v[None, ...] for k, v in y.items()} for y in Yt], dtype=object)
                Xt = Xt[:, None, ...]
            test_ds = SequenceDataset(
                Xt,
                Yt,
                edge_index_np,
                edge_attr,
                node_type=node_types,
                edge_type=edge_types,
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
                )
            test_loader = TorchLoader(
                test_ds,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
        else:
            test_list = load_dataset(
                args.x_test_path,
                args.y_test_path,
                args.edge_index_path,
                edge_attr=edge_attr,
                node_type=node_types,
                edge_type=edge_types,
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

        norm_stats = checkpoint.get("norm_stats") if isinstance(checkpoint, dict) else None
        if norm_stats is None:
            norm_path = Path(str(Path(model_path).with_suffix("")) + "_norm.npz")
            if norm_path.exists():
                arr = np.load(norm_path)
                if "y_mean_node" in arr:
                    norm_stats = {
                        "y_mean_node": arr["y_mean_node"],
                        "y_std_node": arr["y_std_node"],
                    }
                    if "y_mean_edge" in arr:
                        norm_stats["y_mean_edge"] = arr["y_mean_edge"]
                        norm_stats["y_std_edge"] = arr["y_std_edge"]
                elif "y_mean" in arr:
                    norm_stats = {"y_mean": arr["y_mean"], "y_std": arr["y_std"]}
        if norm_stats is not None:
            if "y_mean_node" in norm_stats:
                model.y_mean = torch.tensor(
                    norm_stats["y_mean_node"], dtype=torch.float32, device=device
                )
                model.y_std = torch.tensor(
                    norm_stats["y_std_node"], dtype=torch.float32, device=device
                )
                if "y_mean_edge" in norm_stats:
                    model.y_mean_edge = torch.tensor(
                        norm_stats["y_mean_edge"], dtype=torch.float32, device=device
                    )
                    model.y_std_edge = torch.tensor(
                        norm_stats["y_std_edge"], dtype=torch.float32, device=device
                    )
                else:
                    model.y_mean_edge = model.y_std_edge = None
            else:
                y_mean_np = norm_stats.get("y_mean")
                y_std_np = norm_stats.get("y_std")
                if isinstance(y_mean_np, dict):
                    node_mean = y_mean_np.get("node_outputs")
                    node_std = y_std_np.get("node_outputs")
                    model.y_mean = torch.tensor(
                        node_mean, dtype=torch.float32, device=device
                    )
                    model.y_std = torch.tensor(
                        node_std, dtype=torch.float32, device=device
                    )
                    y_mean_edge_np = y_mean_np.get("edge_outputs")
                    y_std_edge_np = y_std_np.get("edge_outputs")
                    if y_mean_edge_np is not None:
                        model.y_mean_edge = torch.tensor(
                            y_mean_edge_np, dtype=torch.float32, device=device
                        )
                        model.y_std_edge = torch.tensor(
                            y_std_edge_np, dtype=torch.float32, device=device
                        )
                    else:
                        model.y_mean_edge = model.y_std_edge = None
                elif y_mean_np is not None:
                    model.y_mean = torch.tensor(
                        y_mean_np, dtype=torch.float32, device=device
                    )
                    model.y_std = torch.tensor(
                        y_std_np, dtype=torch.float32, device=device
                    )
                    model.y_mean_edge = model.y_std_edge = None
                else:
                    model.y_mean = model.y_std = None
                    model.y_mean_edge = model.y_std_edge = None
        else:
            model.y_mean = model.y_std = None
            model.y_mean_edge = model.y_std_edge = None
        if args.normalize and model.y_mean is None:
            raise RuntimeError("Normalization statistics not found for denormalization")
        model.eval()

        p_stats = RunningStats()
        pred_samples: List[float] = []
        true_samples: List[float] = []

        exclude = set(wn.reservoir_name_list) | set(wn.tank_name_list)
        node_mask_np = np.array([n not in exclude for n in wn.node_name_list])
        node_mask = torch.tensor(node_mask_np, dtype=torch.bool, device=device)

        with torch.no_grad():
            if seq_mode:
                ei = test_ds.edge_index.to(device)
                ea = test_ds.edge_attr.to(device) if test_ds.edge_attr is not None else None
                nt = test_ds.node_type.to(device) if test_ds.node_type is not None else None
                et = test_ds.edge_type.to(device) if test_ds.edge_type is not None else None
                for X_seq, Y_seq in test_loader:
                    X_seq = X_seq.to(device)
                    with autocast(device_type=device.type, enabled=args.amp):
                        out = model(X_seq, ei, ea, nt, et)
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
                            y_mean_node = model.y_mean["node_outputs"].to(node_pred.device)
                            y_std_node = model.y_std["node_outputs"].to(node_pred.device)
                        else:
                            y_mean_node = model.y_mean.to(node_pred.device)
                            y_std_node = model.y_std.to(node_pred.device)
                        node_pred = node_pred * y_std_node + y_mean_node
                        Y_node = Y_node * y_std_node + y_mean_node
                    if (
                        edge_pred is not None
                        and Y_edge is not None
                        and getattr(model, "y_mean_edge", None) is not None
                    ):
                        y_mean_edge = model.y_mean_edge.to(node_pred.device)
                        y_std_edge = model.y_std_edge.to(node_pred.device)
                        edge_pred = edge_pred.squeeze(-1)
                        edge_pred = edge_pred * y_std_edge + y_mean_edge
                        Y_edge = Y_edge * y_std_edge + y_mean_edge

                    pred_p = node_pred[..., 0].reshape(-1, node_mask.numel())
                    true_p = Y_node[..., 0].reshape(-1, node_mask.numel())
                    pred_p = pred_p[:, node_mask].reshape(-1)
                    true_p = true_p[:, node_mask].reshape(-1)
                    pred_np = pred_p.cpu().numpy()
                    true_np = true_p.cpu().numpy()
                    p_stats.update(pred_np, true_np)
                    if args.eval_sample != 0 and len(pred_samples) < args.eval_sample:
                        remain = args.eval_sample - len(pred_samples)
                        pred_samples.extend(pred_np[:remain])
                        true_samples.extend(true_np[:remain])
            else:
                for batch in test_loader:
                    batch = batch.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=args.amp):
                        out = model(
                            batch.x,
                            batch.edge_index,
                            getattr(batch, "edge_attr", None),
                            getattr(batch, "node_type", None),
                            getattr(batch, "edge_type", None),
                        )
                    if hasattr(model, "y_mean") and model.y_mean is not None:
                        node_out = out["node_outputs"] if isinstance(out, dict) else out
                        if isinstance(model.y_mean, dict):
                            y_mean_node = model.y_mean["node_outputs"].to(out.device)
                            y_std_node = model.y_std["node_outputs"].to(out.device)
                            num_nodes = y_mean_node.shape[0]
                            node_out = node_out.view(batch.num_graphs, num_nodes, -1)
                            batch_y = batch.y.view(batch.num_graphs, num_nodes, -1)
                            node_out = node_out * y_std_node.view(1, num_nodes, -1) + y_mean_node.view(1, num_nodes, -1)
                            batch_y = batch_y * y_std_node.view(1, num_nodes, -1) + y_mean_node.view(1, num_nodes, -1)
                            node_out = node_out.view(-1, node_out.shape[-1])
                            batch_y = batch_y.view(-1, batch_y.shape[-1])
                        else:
                            y_std = model.y_std.to(out.device)
                            y_mean = model.y_mean.to(out.device)
                            node_out = node_out * y_std + y_mean
                            batch_y = batch.y * y_std + y_mean
                        edge_out = out.get("edge_outputs") if isinstance(out, dict) else None
                        edge_y = batch.edge_y if getattr(batch, "edge_y", None) is not None else None
                        if (
                            edge_out is not None
                            and edge_y is not None
                            and getattr(model, "y_mean_edge", None) is not None
                        ):
                            y_mean_edge = model.y_mean_edge.to(out.device)
                            y_std_edge = model.y_std_edge.to(out.device)
                            num_edges = y_mean_edge.shape[0]
                            edge_out = edge_out.view(batch.num_graphs, num_edges, -1)
                            edge_y = edge_y.view(batch.num_graphs, num_edges, -1)
                            edge_out = edge_out * y_std_edge.view(1, num_edges, -1) + y_mean_edge.view(1, num_edges, -1)
                            edge_y = edge_y * y_std_edge.view(1, num_edges, -1) + y_mean_edge.view(1, num_edges, -1)
                            edge_out = edge_out.view(-1, edge_out.shape[-1])
                            edge_y = edge_y.view(-1, edge_y.shape[-1])
                        else:
                            if edge_out is not None:
                                edge_out = edge_out.squeeze(-1)
                            if edge_y is not None:
                                edge_y = edge_y.squeeze(-1)
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
                    pred_np = pred_p.cpu().numpy()
                    true_np = true_p.cpu().numpy()
                    p_stats.update(pred_np, true_np)
                    if args.eval_sample != 0 and len(pred_samples) < args.eval_sample:
                        remain = args.eval_sample - len(pred_samples)
                        pred_samples.extend(pred_np[:remain])
                        true_samples.extend(true_np[:remain])

        if args.normalize and getattr(model, "y_mean", None) is None:
            raise RuntimeError("Denormalized metrics requested but normalization stats are missing")
        save_accuracy_metrics(p_stats, run_name)
        if args.eval_sample != 0 and pred_samples:
            pred_arr = np.array(pred_samples, dtype=float)
            true_arr = np.array(true_samples, dtype=float)
            pred_vs_actual_scatter(pred_arr, true_arr, run_name)
            plot_error_histogram(pred_arr - true_arr, run_name)

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
        "--hidden-dim",
        type=int,
        choices=[128, 256],
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument("--use-attention", action="store_true",
                        help="Use GATConv instead of HydroConv for graph convolution")
    parser.add_argument("--gat-heads", type=int, default=4,
                        help="Number of attention heads for GATConv (if attention is enabled)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate applied after each GNN layer")
    parser.add_argument("--num-layers", type=int, choices=[4, 6, 8], default=4,
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
        choices=[64, 128],
        default=64,
        help="Hidden dimension of the recurrent LSTM layer",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=1,
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
        "--pump-loss",
        "--pump_loss",
        dest="pump_loss",
        action="store_true",
        help="Add pump curve consistency penalty",
    )
    parser.add_argument(
        "--no-pump-loss",
        "--no-pump_loss",
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
        "--w-flow",
        type=float,
        default=3.0,
        help="Weight of the edge (flow) loss term",
    )
    parser.add_argument(
        "--mass-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for mass conservation loss (0 = auto-compute; clamped to ≥1.0)",
    )
    parser.add_argument(
        "--head-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for head loss consistency (0 = auto-compute; clamped to ≥1.0)",
    )
    parser.add_argument(
        "--pump-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for pump curve loss (0 = auto-compute; clamped to ≥1.0)",
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
        "--eval-sample",
        type=int,
        default=1000,
        help="Number of predictions to retain for evaluation plots (0 disables)",
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
