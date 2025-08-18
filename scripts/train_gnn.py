import argparse
import os
import random
from pathlib import Path
from datetime import datetime
import sys
import signal
import warnings

import numpy as np
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

from scripts.metrics import accuracy_metrics, export_table

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


def compute_edge_attr_stats(edge_attr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std for edge attribute matrix."""
    attr_mean = torch.tensor(edge_attr.mean(axis=0), dtype=torch.float32)
    attr_std = torch.tensor(edge_attr.std(axis=0) + 1e-8, dtype=torch.float32)
    return attr_mean, attr_std


def save_scatter_plots(
    true_p,
    preds_p,
    true_c,
    preds_c,
    run_name: str,
    plots_dir: Optional[Path] = None,
    mask: Optional[Sequence[bool]] = None,
) -> None:
    """Save enhanced scatter plots for surrogate predictions."""
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


def save_accuracy_metrics(
    true_p,
    preds_p,
    true_c,
    preds_c,
    run_name: str,
    logs_dir: Optional[Path] = None,
    mask: Optional[Sequence[bool]] = None,
    true_f: Optional[Sequence[float]] = None,
    preds_f: Optional[Sequence[float]] = None,
) -> None:
    """Compute and export accuracy metrics to a CSV file."""
    if logs_dir is None:
        logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tp = _to_numpy(true_p)
    pp = _to_numpy(preds_p)
    tc = _to_numpy(true_c) if true_c is not None else None
    pc = _to_numpy(preds_c) if preds_c is not None else None

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        tp = tp[m]
        pp = pp[m]
        if tc is not None and pc is not None:
            tc = tc[m]
            pc = pc[m]

    df = accuracy_metrics(
        tp,
        pp,
        None if tc is None else np.expm1(tc) * 1000.0,
        None if pc is None else np.expm1(pc) * 1000.0,
    )
    if true_f is not None and preds_f is not None:
        tf = _to_numpy(true_f)
        pf = _to_numpy(preds_f)
        abs_f = np.abs(tf - pf)
        mae_f = abs_f.mean()
        rmse_f = np.sqrt(((tf - pf) ** 2).mean())
        mape_f = (abs_f / np.maximum(np.abs(tf), 1e-8)).mean() * 100.0
        max_err_f = abs_f.max()
        df["Flow (m^3/h)"] = [mae_f, rmse_f, mape_f, max_err_f]
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
):
    model.train()
    scaler = GradScaler(device=device.type, enabled=amp)
    total_loss = press_total = cl_total = flow_total = 0.0
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        press_total += press_l.item() * batch.num_graphs
        cl_total += cl_l.item() * batch.num_graphs
        flow_total += flow_l.item() * batch.num_graphs
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        press_total / denom,
        cl_total / denom,
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
):
    global interrupted
    model.eval()
    total_loss = press_total = cl_total = flow_total = 0.0
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
            total_loss += loss.item() * batch.num_graphs
            press_total += press_l.item() * batch.num_graphs
            cl_total += cl_l.item() * batch.num_graphs
            flow_total += flow_l.item() * batch.num_graphs
            if interrupted:
                break
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        press_total / denom,
        cl_total / denom,
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
    w_cl: float = 1.0,
    w_flow: float = 1.0,
    amp: bool = False,
    progress: bool = True,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    global interrupted
    model.train()
    scaler = GradScaler(device=device.type, enabled=amp)
    total_loss = 0.0
    press_total = cl_total = flow_total = 0.0
    mass_total = head_total = sym_total = pump_total = 0.0
    mass_imb_total = head_viol_total = 0.0
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
        if node_type is not None:
            nt = node_type.to(device)
        else:
            nt = None
        if edge_type is not None:
            et = edge_type.to(device)
        else:
            et = None
        if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
            init_press = X_seq[:, 0, model.tank_indices, 1]
            init_levels = init_press * model.tank_areas
            model.reset_tank_levels(init_levels)
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=amp):
            preds = model(
                X_seq,
                edge_index.to(device),
                edge_attr.to(device),
                nt,
                et,
            )
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
                dem_seq = X_seq[..., 0]
                if dem_seq.size(1) > 1:
                    dem_seq = torch.cat([dem_seq[:, 1:], dem_seq[:, -1:]], dim=1)
                demand_mb = dem_seq.permute(2, 0, 1).reshape(node_count, -1)
                if hasattr(model, "x_mean") and model.x_mean is not None:
                    dem_mean = model.x_mean[0].to(device)
                    dem_std = model.x_std[0].to(device)
                    demand_mb = demand_mb * dem_std + dem_mean
                mass_loss, mass_imb = compute_mass_balance_loss(
                    flows_mb,
                    edge_index.to(device),
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
                if hasattr(model, 'y_mean') and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        p_mean = model.y_mean['node_outputs'][0].to(device)
                        p_std = model.y_std['node_outputs'][0].to(device)
                        q_mean = model.y_mean['edge_outputs'].to(device)
                        q_std = model.y_std['edge_outputs'].to(device)
                        press = press * p_std + p_mean
                        flow = flow * q_std + q_mean
                    else:
                        press = press * model.y_std[0].to(device) + model.y_mean[0].to(device)
                head_loss, head_violation = pressure_headloss_consistency_loss(
                    press,
                    flow,
                    edge_index.to(device),
                    edge_attr_phys.to(device),
                    edge_type=et,
                    return_violation=True,
                )
            else:
                head_loss = torch.tensor(0.0, device=device)
                head_violation = torch.tensor(0.0, device=device)
            if pump_loss and pump_coeffs is not None:
                flow_pc = edge_preds.squeeze(-1)
                if hasattr(model, 'y_mean') and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        q_mean = model.y_mean['edge_outputs'].to(device)
                        q_std = model.y_std['edge_outputs'].to(device)
                        flow_pc = flow_pc * q_std + q_mean
                    else:
                        flow_pc = flow_pc * model.y_std[-1].to(device) + model.y_mean[-1].to(device)
                pump_loss_val = pump_curve_loss(
                    flow_pc,
                    pump_coeffs.to(device),
                    edge_index.to(device),
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
            loss_press = loss_cl = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
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
        cl_total += loss_cl.item() * X_seq.size(0)
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
        cl_total / denom,
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
    w_cl: float = 1.0,
    w_flow: float = 1.0,
    amp: bool = False,
    progress: bool = True,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    global interrupted
    model.eval()
    total_loss = 0.0
    press_total = cl_total = flow_total = 0.0
    mass_total = head_total = sym_total = pump_total = 0.0
    mass_imb_total = head_viol_total = 0.0
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
            if node_type is not None:
                nt = node_type.to(device)
            else:
                nt = None
            if edge_type is not None:
                et = edge_type.to(device)
            else:
                et = None
            if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
                init_press = X_seq[:, 0, model.tank_indices, 1]
                init_levels = init_press * model.tank_areas
                model.reset_tank_levels(init_levels)
            with autocast(device_type=device.type, enabled=amp):
                preds = model(
                    X_seq,
                    edge_index.to(device),
                    edge_attr.to(device),
                    nt,
                    et,
                )
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
                        dem_seq = X_seq[..., 0]
                        if dem_seq.size(1) > 1:
                            dem_seq = torch.cat([dem_seq[:, 1:], dem_seq[:, -1:]], dim=1)
                        demand_mb = dem_seq.permute(2, 0, 1).reshape(node_count, -1)
                        if hasattr(model, "x_mean") and model.x_mean is not None:
                            dem_mean = model.x_mean[0].to(device)
                            dem_std = model.x_std[0].to(device)
                            demand_mb = demand_mb * dem_std + dem_mean
                        mass_loss, mass_imb = compute_mass_balance_loss(
                            flows_mb,
                            edge_index.to(device),
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
                        if hasattr(model, 'y_mean') and model.y_mean is not None:
                            if isinstance(model.y_mean, dict):
                                p_mean = model.y_mean['node_outputs'][0].to(device)
                                p_std = model.y_std['node_outputs'][0].to(device)
                                q_mean = model.y_mean['edge_outputs'].to(device)
                                q_std = model.y_std['edge_outputs'].to(device)
                                press = press * p_std + p_mean
                                flow = flow * q_std + q_mean
                            else:
                                press = press * model.y_std[0].to(device) + model.y_mean[0].to(device)
                        head_loss, head_violation = pressure_headloss_consistency_loss(
                            press,
                            flow,
                            edge_index.to(device),
                            edge_attr_phys.to(device),
                            edge_type=et,
                            return_violation=True,
                        )
                    else:
                        head_loss = torch.tensor(0.0, device=device)
                        head_violation = torch.tensor(0.0, device=device)
                    if pump_loss and pump_coeffs is not None:
                        flow_pc = edge_preds.squeeze(-1)
                        if hasattr(model, 'y_mean') and model.y_mean is not None:
                            if isinstance(model.y_mean, dict):
                                q_mean = model.y_mean['edge_outputs'].to(device)
                                q_std = model.y_std['edge_outputs'].to(device)
                                flow_pc = flow_pc * q_std + q_mean
                            else:
                                flow_pc = flow_pc * model.y_std[-1].to(device) + model.y_mean[-1].to(device)
                        pump_loss_val = pump_curve_loss(
                            flow_pc,
                            pump_coeffs.to(device),
                            edge_index.to(device),
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
                    loss_press = loss_cl = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
                    head_loss = pump_loss_val = torch.tensor(0.0, device=device)
                    mass_imb = head_violation = torch.tensor(0.0, device=device)
                    loss = _apply_loss(preds, Y_seq.float(), loss_fn)
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
            if interrupted:
                break
    denom = len(loader.dataset)
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
    if base_dim == 4:
        has_chlorine = True
    elif base_dim == 3:
        has_chlorine = False
    else:
        raise ValueError(
            f"Dataset provides {sample_dim} features per node but the network has {pump_count} pumps."
        )
    args.output_dim = 2 if has_chlorine else 1

    norm_md5 = None
    if args.normalize:
        if seq_mode:
            x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(
                X_raw, Y_raw
            )
            apply_sequence_normalization(
                data_ds,
                x_mean,
                x_std,
                y_mean,
                y_std,
                edge_mean,
                edge_std,
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
                )
        else:
            x_mean, x_std, y_mean, y_std = compute_norm_stats(data_list)
            apply_normalization(
                data_list, x_mean, x_std, y_mean, y_std, edge_mean, edge_std
            )
            if val_list:
                apply_normalization(
                    val_list, x_mean, x_std, y_mean, y_std, edge_mean, edge_std
                )
        print("Target normalization stats:")
        if isinstance(y_mean, dict):
            print(
                "Pressure mean/std:",
                y_mean["node_outputs"][0].item(),
                y_std["node_outputs"][0].item(),
            )
            if has_chlorine:
                print(
                    "Chlorine mean/std:",
                    y_mean["node_outputs"][1].item(),
                    y_std["node_outputs"][1].item(),
                )
        else:
            print("Pressure mean/std:", y_mean[0].item(), y_std[0].item())
            if has_chlorine and len(y_mean) > 1:
                print("Chlorine mean/std:", y_mean[1].item(), y_std[1].item())

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

    expected_in_dim = (4 if has_chlorine else 3) + len(wn.pump_name_list)

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
        base_eval = evaluate_sequence(
            model,
            loader,
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
            mass_scale=1.0,
            head_scale=1.0,
            pump_scale=1.0,
            w_mass=args.w_mass,
            w_head=args.w_head,
            w_pump=args.w_pump,
            w_press=args.w_press,
            w_cl=args.w_cl,
            w_flow=args.w_flow,
            amp=args.amp,
            progress=False,
        )
        if args.physics_loss and mass_scale <= 0:
            mass_scale = float(base_eval[4]) if base_eval[4] > 0 else 1.0
        if args.pressure_loss and head_scale <= 0:
            head_scale = float(base_eval[5]) if base_eval[5] > 0 else 1.0
        if args.pump_loss and pump_scale <= 0:
            pump_scale = float(base_eval[7]) if base_eval[7] > 0 else 1.0
    args.mass_scale = mass_scale
    args.head_scale = head_scale
    args.pump_scale = pump_scale
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
                "epoch,train_loss,val_loss,press_loss,cl_loss,flow_loss,mass_imbalance,head_violation,val_press_loss,val_cl_loss,val_flow_loss,val_mass_imbalance,val_head_violation,lr\n"
            )
        else:
            f.write(
                "epoch,train_loss,val_loss,press_loss,cl_loss,flow_loss,val_press_loss,val_cl_loss,val_flow_loss,lr\n"
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
                    w_cl=args.w_cl,
                    w_flow=args.w_flow,
                    amp=args.amp,
                    progress=args.progress,
                )
                loss = loss_tuple[0]
                press_l, cl_l, flow_l, mass_l, head_l, sym_l, pump_l, mass_imb, head_viols = loss_tuple[1:]
                comp = [press_l, cl_l, flow_l, mass_l, sym_l]
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
                        w_cl=args.w_cl,
                        w_flow=args.w_flow,
                        amp=args.amp,
                        progress=args.progress,
                    )
                    val_loss = val_tuple[0]
                    val_press_l, val_cl_l, val_flow_l, val_mass_imb, val_head_viols = val_tuple[1:6]
                else:
                    val_loss = loss
                    val_press_l, val_cl_l, val_flow_l = press_l, cl_l, flow_l
                    val_mass_imb, val_head_viols = mass_imb, head_viols
            else:
                loss, press_l, cl_l, flow_l = train(
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
            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            losses.append((loss, val_loss))
            if seq_mode:
                f.write(
                    f"{epoch},{loss:.6f},{val_loss:.6f},{press_l:.6f},{cl_l:.6f},{flow_l:.6f},{mass_imb:.6f},{head_viols:.6f},"
                    f"{val_press_l:.6f},{val_cl_l:.6f},{val_flow_l:.6f},{val_mass_imb:.6f},{val_head_viols:.6f},{curr_lr:.6e}\n"
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
                            "total": val_loss,
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
                    print(msg)
                else:
                    print(f"Epoch {epoch}")
            else:
                f.write(
                    f"{epoch},{loss:.6f},{val_loss:.6f},{press_l:.6f},{cl_l:.6f},{flow_l:.6f},"
                    f"{val_press_l:.6f},{val_cl_l:.6f},{val_flow_l:.6f},{curr_lr:.6e}\n"
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
                            "total": val_loss,
                            "pressure": val_press_l,
                            "chlorine": val_cl_l,
                            "flow": val_flow_l,
                        },
                        epoch,
                    )
                print(
                    f"Epoch {epoch}: press={press_l:.3f}, cl={cl_l:.3f}, flow={flow_l:.3f}"
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
                apply_normalization(test_list, x_mean, x_std, y_mean, y_std, edge_mean, edge_std)
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
        preds_p = []
        preds_c = []
        preds_f = []
        true_p = []
        true_c = []
        true_f = []
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
                            y_mean_node = model.y_mean['node_outputs'].to(node_pred.device)
                            y_std_node = model.y_std['node_outputs'].to(node_pred.device)
                            node_pred = node_pred * y_std_node + y_mean_node
                            Y_node = Y_node * y_std_node + y_mean_node
                            if edge_pred is not None and Y_edge is not None:
                                y_mean_edge = model.y_mean['edge_outputs'].to(node_pred.device)
                                y_std_edge = model.y_std['edge_outputs'].to(node_pred.device)
                                edge_pred = edge_pred * y_std_edge + y_mean_edge
                                Y_edge = Y_edge * y_std_edge + y_mean_edge
                        else:
                            y_std = model.y_std.to(node_pred.device)
                            y_mean = model.y_mean.to(node_pred.device)
                            node_pred = node_pred * y_std + y_mean
                            Y_node = Y_node * y_std + y_mean
                    preds_p.extend(node_pred[..., 0].cpu().numpy().ravel())
                    true_p.extend(Y_node[..., 0].cpu().numpy().ravel())
                    if has_chlorine and node_pred.shape[-1] > 1:
                        preds_c.extend(node_pred[..., 1].cpu().numpy().ravel())
                        true_c.extend(Y_node[..., 1].cpu().numpy().ravel())
                    if edge_pred is not None and Y_edge is not None:
                        preds_f.extend(edge_pred.squeeze(-1).cpu().numpy().ravel())
                        true_f.extend(Y_edge.squeeze(-1).cpu().numpy().ravel())
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
                                edge_out = out["edge_outputs"] * y_std_edge + y_mean_edge
                                edge_y = batch.edge_y * y_std_edge + y_mean_edge
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
                    preds_p.extend(node_out[:, 0].cpu().numpy())
                    true_p.extend(batch_y[:, 0].cpu().numpy())
                    if has_chlorine and node_out.shape[1] > 1:
                        preds_c.extend(node_out[:, 1].cpu().numpy())
                        true_c.extend(batch_y[:, 1].cpu().numpy())
                    if edge_out is not None and edge_y is not None:
                        preds_f.extend(edge_out.squeeze(-1).cpu().numpy())
                        true_f.extend(edge_y.squeeze(-1).cpu().numpy())
        if preds_p:
            preds_p = np.array(preds_p)
            true_p = np.array(true_p)
            if preds_f:
                preds_f = np.array(preds_f)
                true_f = np.array(true_f)
            else:
                preds_f = true_f = None
            if has_chlorine and preds_c:
                preds_c = np.array(preds_c)
                true_c = np.array(true_c)
                err_c = preds_c - true_c
            else:
                preds_c = true_c = err_c = None
            err_p = preds_p - true_p

            exclude = set(wn.reservoir_name_list) | set(wn.tank_name_list)
            node_mask = np.array([n not in exclude for n in wn.node_name_list])
            repeat = preds_p.size // node_mask.size
            full_mask = np.tile(node_mask, repeat)

            save_scatter_plots(
                true_p,
                preds_p,
                true_c,
                preds_c,
                run_name,
                mask=full_mask,
            )
            if seq_mode:
                plot_sequence_prediction(model, test_ds, run_name)
            save_accuracy_metrics(
                true_p,
                preds_p,
                true_c,
                preds_c,
                run_name,
                mask=full_mask,
                true_f=true_f,
                preds_f=preds_f,
            )
            plot_error_histograms(err_p, err_c, run_name, mask=full_mask)
            labels = ["demand", "pressure"]
            if has_chlorine:
                labels.append("chlorine")
            labels.append("elevation")
            labels += [f"pump_{i}" for i in range(len(wn.pump_name_list))]
            X_flat = X_raw.reshape(-1, X_raw.shape[-1])
            correlation_heatmap(X_flat, labels, run_name)

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
        help="Baseline magnitude for mass conservation loss (0 = auto-compute)",
    )
    parser.add_argument(
        "--head-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for head loss consistency (0 = auto-compute)",
    )
    parser.add_argument(
        "--pump-scale",
        type=float,
        default=0.0,
        help="Baseline magnitude for pump curve loss (0 = auto-compute)",
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
