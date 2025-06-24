import argparse
import os
from pathlib import Path
from datetime import datetime
import sys
import signal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader as TorchLoader
from torch_geometric.nn import GCNConv, MessagePassing, GATConv, LayerNorm
from torch_geometric.utils import subgraph, k_hop_subgraph
import wntr
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Sequence
import networkx as nx

# Ensure the repository root is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loss_utils import (
    compute_mass_balance_loss,
    pressure_headloss_consistency_loss,
)


class HydroConv(MessagePassing):
    """Mass-conserving convolution supporting heterogeneous components."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ) -> None:
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Separate linear layers per node type so different components can
        # learn distinct transformations.
        self.lin = nn.ModuleList(
            [nn.Linear(in_channels, out_channels) for _ in range(num_node_types)]
        )

        # Each edge type obtains its own small MLP to compute a transmissibility
        # weight from the edge attributes.
        self.edge_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(edge_dim, 1), nn.Softplus()) for _ in range(num_edge_types)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run convolution with optional node/edge type information."""

        if edge_attr is None:
            raise ValueError("edge_attr cannot be None for HydroConv")

        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_type=edge_type)
        out = torch.zeros((x.size(0), self.out_channels), device=x.device, dtype=aggr.dtype)
        for t, lin in enumerate(self.lin):
            idx = node_type == t
            if torch.any(idx):
                out[idx] = lin(aggr[idx])
        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        weight = torch.zeros(edge_attr.size(0), 1, device=edge_attr.device)
        for t, mlp in enumerate(self.edge_mlps):
            idx = edge_type == t
            if torch.any(idx):
                weight[idx] = mlp(edge_attr[idx])
        return weight * (x_j - x_i)


class EnhancedGNNEncoder(nn.Module):
    """Flexible GNN encoder supporting heterogeneous graphs and attention.

    The forward pass expects explicit tensor arguments so that the model can be
    compiled with TorchScript.  ``x`` is a node feature matrix of shape
    ``[N, F]`` and ``edge_index`` contains the directed edges with shape
    ``[2, E]``. Optional edge attributes and type indices can also be provided.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        residual: bool = False,
        edge_dim: Optional[int] = None,
        use_attention: bool = False,
        gat_heads: int = 4,
        share_weights: bool = False,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ) -> None:
        super().__init__()

        self.use_attention = use_attention
        self.dropout = dropout
        self.residual = residual
        self.act_fn = getattr(F, activation)
        self.share_weights = share_weights
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        def make_conv(in_c: int, out_c: int):
            if use_attention:
                return GATConv(in_c, out_c // gat_heads, heads=gat_heads, edge_dim=edge_dim)
            if edge_dim is not None:
                return HydroConv(
                    in_c,
                    out_c,
                    edge_dim,
                    num_node_types=self.num_node_types,
                    num_edge_types=self.num_edge_types,
                )
            return GCNConv(in_c, out_c)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if share_weights:
            first = make_conv(in_channels, hidden_channels)
            self.convs.append(first)
            shared = first if in_channels == hidden_channels else make_conv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1):
                self.convs.append(shared)
        else:
            for i in range(num_layers):
                in_c = in_channels if i == 0 else hidden_channels
                self.convs.append(make_conv(in_c, hidden_channels))

        for _ in range(num_layers):
            self.norms.append(LayerNorm(hidden_channels))

        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            identity = x
            if isinstance(conv, HydroConv):
                x = conv(x, edge_index, edge_attr, node_type, edge_type)
            elif conv.__class__.__name__ == "GATConv":
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = norm(x)
            if self.residual and identity.shape == x.shape:
                x = x + identity
        return self.fc_out(x)


# Backwards compatibility for tests and older scripts
GCNEncoder = EnhancedGNNEncoder


class RecurrentGNNSurrogate(nn.Module):
    """GNN encoder followed by an LSTM for sequence prediction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        output_dim: int,
        num_layers: int,
        use_attention: bool,
        gat_heads: int,
        dropout: float,
        residual: bool,
        rnn_hidden_dim: int,
        share_weights: bool = False,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = EnhancedGNNEncoder(
            in_channels,
            hidden_channels,
            hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            edge_dim=edge_dim,
            use_attention=use_attention,
            gat_heads=gat_heads,
            share_weights=share_weights,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        )
        self.rnn = nn.LSTM(hidden_channels, rnn_hidden_dim, batch_first=True)
        self.decoder = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(
        self,
        X_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        # Expand edge index for ``batch_size`` separate graphs
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        edge_attr_rep = edge_attr.repeat(batch_size, 1) if edge_attr is not None else None
        node_type_rep = (
            node_type.repeat(batch_size) if node_type is not None else None
        )
        edge_type_rep = (
            edge_type.repeat(batch_size) if edge_type is not None else None
        )

        node_embeddings = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)
            gnn_out = self.encoder(
                x_t,
                batch_edge_index,
                edge_attr_rep,
                node_type_rep,
                edge_type_rep,
            )  # [batch_size*num_nodes, hidden]
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)
            node_embeddings.append(gnn_out)

        emb = torch.stack(node_embeddings, dim=1)  # [B, T, N, H]
        # reshape so each node has its own sequence in the batch dimension
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)
        out = self.decoder(rnn_out)
        # Clamp pressure and chlorine in the normalized domain so that 0
        # corresponds to 0 in physical units when denormalised.
        out = out.clone()
        min_p = min_c = 0.0
        if getattr(self, "y_mean", None) is not None:
            if isinstance(self.y_mean, dict):
                p_mean = self.y_mean["node_outputs"][0].to(out.device)
                p_std = self.y_std["node_outputs"][0].to(out.device)
                if self.y_mean["node_outputs"].numel() > 1:
                    c_mean = self.y_mean["node_outputs"][1].to(out.device)
                    c_std = self.y_std["node_outputs"][1].to(out.device)
                else:
                    c_mean = torch.tensor(0.0, device=out.device)
                    c_std = torch.tensor(1.0, device=out.device)
            else:
                p_mean = self.y_mean[0].to(out.device)
                p_std = self.y_std[0].to(out.device)
                if self.y_mean.numel() > 1:
                    c_mean = self.y_mean[1].to(out.device)
                    c_std = self.y_std[1].to(out.device)
                else:
                    c_mean = torch.tensor(0.0, device=out.device)
                    c_std = torch.tensor(1.0, device=out.device)
            min_p = -p_mean / p_std
            min_c = -c_mean / c_std

        if out.size(-1) >= 1:
            out[..., 0] = torch.clamp(out[..., 0], min=min_p)
        if out.size(-1) >= 2:
            out[..., 1] = torch.clamp(out[..., 1], min=min_c)
        return out


class MultiTaskGNNSurrogate(nn.Module):
    """Recurrent GNN surrogate predicting node and edge level targets."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        node_output_dim: int,
        edge_output_dim: int,
        num_layers: int,
        use_attention: bool,
        gat_heads: int,
        dropout: float,
        residual: bool,
        rnn_hidden_dim: int,
        share_weights: bool = False,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = EnhancedGNNEncoder(
            in_channels,
            hidden_channels,
            hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            edge_dim=edge_dim,
            use_attention=use_attention,
            gat_heads=gat_heads,
            share_weights=share_weights,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        )
        self.rnn = nn.LSTM(hidden_channels, rnn_hidden_dim, batch_first=True)
        self.time_att = MultiheadAttention(rnn_hidden_dim, num_heads=4, batch_first=True)
        self.node_decoder = nn.Linear(rnn_hidden_dim, node_output_dim)

        self.edge_decoder = nn.Linear(rnn_hidden_dim * 3, edge_output_dim)


    def reset_tank_levels(
        self,
        init_levels: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize internal tank volumes.

        Parameters
        ----------
        init_levels : torch.Tensor, optional
            Tensor of initial tank volumes (``[B, num_tanks]``).  When ``None``
            the levels are reset to zero.
        batch_size : int
            Number of sequences in the batch when ``init_levels`` is ``None``.
        device : torch.device, optional
            Target device for the tensor.
        """
        if not hasattr(self, "tank_indices"):
            return
        if device is None:
            device = next(self.parameters()).device
        if init_levels is None:
            init_levels = torch.zeros(batch_size, len(self.tank_indices), device=device)
        else:
            init_levels = init_levels.to(device)
        self.tank_levels = init_levels

    def forward(
        self,
        X_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ):
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        edge_attr_rep = edge_attr.repeat(batch_size, 1) if edge_attr is not None else None
        node_type_rep = (
            node_type.repeat(batch_size) if node_type is not None else None
        )
        edge_type_rep = (
            edge_type.repeat(batch_size) if edge_type is not None else None
        )

        node_embeddings = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)
            gnn_out = self.encoder(
                x_t,
                batch_edge_index,
                edge_attr_rep,
                node_type_rep,
                edge_type_rep,
            )
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)
            node_embeddings.append(gnn_out)

        emb = torch.stack(node_embeddings, dim=1)
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)

        # apply temporal self-attention so each node can weigh its history
        att_in = rnn_out.reshape(batch_size * num_nodes, T, -1)
        att_out, _ = self.time_att(att_in, att_in, att_in)
        att_out = att_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)

        node_pred = self.node_decoder(att_out)
        node_pred = node_pred.clone()

        src = edge_index[0]
        tgt = edge_index[1]
        h_src = att_out[:, :, src, :]
        h_tgt = att_out[:, :, tgt, :]
        h_diff = h_src - h_tgt
        edge_emb = torch.cat([h_src, h_tgt, h_diff], dim=-1)
        edge_pred = self.edge_decoder(edge_emb)

        if hasattr(self, "tank_indices") and len(getattr(self, "tank_indices")) > 0:
            if not hasattr(self, "tank_levels") or self.tank_levels.size(0) != batch_size:
                self.tank_levels = torch.zeros(batch_size, len(self.tank_indices), device=device)
            flows = edge_pred[..., 0]  # [B, T, E]
            updates = torch.zeros(batch_size, T, num_nodes, device=device)
            for t in range(T):
                net = []
                for edges, signs in zip(self.tank_edges, self.tank_signs):
                    if edges.numel() == 0:
                        net.append(torch.zeros(batch_size, device=device))
                    else:
                        net.append((flows[:, t, edges] * signs).sum(dim=1))
                net_flow = torch.stack(net, dim=1)
                delta_vol = net_flow * 3600.0
                self.tank_levels += delta_vol
                # Prevent negative volumes accumulating
                self.tank_levels = self.tank_levels.clamp(min=0.0)
                delta_h = delta_vol / self.tank_areas
                for i, tank_idx in enumerate(self.tank_indices):
                    updates[:, t, tank_idx] = delta_h[:, i]
            update_tensor = torch.zeros_like(node_pred)
            update_tensor[..., 0] = updates
            node_pred = node_pred + update_tensor
        # Clamp pressure and chlorine in normalized units so that the lower
        # bound corresponds to 0 in physical units.
        min_p = min_c = 0.0
        if getattr(self, "y_mean", None) is not None:
            if isinstance(self.y_mean, dict):
                p_mean = self.y_mean["node_outputs"][0].to(node_pred.device)
                p_std = self.y_std["node_outputs"][0].to(node_pred.device)
                if self.y_mean["node_outputs"].numel() > 1:
                    c_mean = self.y_mean["node_outputs"][1].to(node_pred.device)
                    c_std = self.y_std["node_outputs"][1].to(node_pred.device)
                else:
                    c_mean = torch.tensor(0.0, device=node_pred.device)
                    c_std = torch.tensor(1.0, device=node_pred.device)
            else:
                p_mean = self.y_mean[0].to(node_pred.device)
                p_std = self.y_std[0].to(node_pred.device)
                if self.y_mean.numel() > 1:
                    c_mean = self.y_mean[1].to(node_pred.device)
                    c_std = self.y_std[1].to(node_pred.device)
                else:
                    c_mean = torch.tensor(0.0, device=node_pred.device)
                    c_std = torch.tensor(1.0, device=node_pred.device)
            min_p = -p_mean / p_std
            min_c = -c_mean / c_std

        pressure = torch.clamp(node_pred[..., 0], min=min_p)
        if node_pred.size(-1) >= 2:
            chlorine = torch.clamp(node_pred[..., 1], min=min_c)
            other = node_pred[..., 2:]
            node_pred = torch.cat(
                [pressure.unsqueeze(-1), chlorine.unsqueeze(-1), other], dim=-1
            )
        else:
            node_pred = pressure.unsqueeze(-1)

        return {
            "node_outputs": node_pred,
            "edge_outputs": edge_pred,
        }


def load_dataset(
    x_path: str,
    y_path: str,
    edge_index_path: str = "edge_index.npy",
    edge_attr: Optional[np.ndarray] = None,
    node_type: Optional[np.ndarray] = None,
    edge_type: Optional[np.ndarray] = None,
) -> list[Data]:
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
            data = Data(x=node_feat, edge_index=edge_index, y=torch.tensor(label))
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
            data = Data(x=node_feat, edge_index=edge_index, y=torch.tensor(label))
            if edge_attr_tensor is not None:
                data.edge_attr = edge_attr_tensor
            if node_type_tensor is not None:
                data.node_type = node_type_tensor
            if edge_type_tensor is not None:
                data.edge_type = edge_type_tensor
            data_list.append(data)

    return data_list


def compute_norm_stats(data_list):
    """Compute mean and std per feature/target dimension from ``data_list``."""
    all_x = torch.cat([d.x for d in data_list], dim=0)
    all_y = torch.cat([d.y for d in data_list], dim=0)
    x_mean = all_x.mean(dim=0)
    x_std = all_x.std(dim=0) + 1e-8
    y_mean = all_y.mean(dim=0)
    y_std = all_y.std(dim=0) + 1e-8
    return x_mean, x_std, y_mean, y_std


def _to_numpy(seq: Sequence[float]) -> np.ndarray:
    """Convert sequence to NumPy array."""
    return np.asarray(seq, dtype=float)


def predicted_vs_actual_scatter(
    true_pressure: Sequence[float],
    pred_pressure: Sequence[float],
    true_chlorine: Sequence[float],
    pred_chlorine: Sequence[float],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Scatter plots comparing surrogate predictions with EPANET results."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    tp = _to_numpy(true_pressure)
    pp = _to_numpy(pred_pressure)
    tc = _to_numpy(true_chlorine)
    pc = _to_numpy(pred_chlorine)

    # chlorine values are stored in log space (log1p). Convert back to mg/L
    # before plotting so the axes reflect physical units.
    tc = np.expm1(tc)
    pc = np.expm1(pc)

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

    fig.savefig(plots_dir / f"pred_vs_actual_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def compute_edge_attr_stats(edge_attr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std for edge attribute matrix."""
    attr_mean = torch.tensor(edge_attr.mean(axis=0), dtype=torch.float32)
    attr_std = torch.tensor(edge_attr.std(axis=0) + 1e-8, dtype=torch.float32)
    return attr_mean, attr_std


def save_scatter_plots(
    true_p, preds_p, true_c, preds_c, run_name: str, plots_dir: Optional[Path] = None
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
    )
    # also store individual scatter images for backward compatibility
    axes = fig.axes
    axes[0].figure.savefig(plots_dir / f"pred_vs_actual_pressure_{run_name}.png")
    axes[1].figure.savefig(
        plots_dir / f"pred_vs_actual_chlorine_{run_name}.png"
    )
    plt.close(fig)


def apply_normalization(
    data_list,
    x_mean,
    x_std,
    y_mean,
    y_std,
    edge_attr_mean=None,
    edge_attr_std=None,
):
    for d in data_list:
        d.x = (d.x - x_mean) / x_std
        d.y = (d.y - y_mean) / y_std
        if edge_attr_mean is not None and getattr(d, "edge_attr", None) is not None:
            d.edge_attr = (d.edge_attr - edge_attr_mean) / edge_attr_std


class SequenceDataset(Dataset):
    """Simple ``Dataset`` for sequence data supporting multi-task labels."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, edge_index: np.ndarray, edge_attr: Optional[np.ndarray], node_type: Optional[np.ndarray] = None, edge_type: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_attr = None
        if edge_attr is not None:
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        self.node_type = None
        if node_type is not None:
            self.node_type = torch.tensor(node_type, dtype=torch.long)
        self.edge_type = None
        if edge_type is not None:
            self.edge_type = torch.tensor(edge_type, dtype=torch.long)

        first = Y[0]
        if isinstance(first, dict) or (isinstance(first, np.ndarray) and Y.dtype == object):
            self.multi = True
            self.Y = {
                "node_outputs": torch.stack(
                    [torch.tensor(y["node_outputs"], dtype=torch.float32) for y in Y]
                ),
                "edge_outputs": torch.stack(
                    [torch.tensor(y["edge_outputs"], dtype=torch.float32) for y in Y]
                ),
            }
        else:
            self.multi = False
            self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        if self.multi:
            return self.X[idx], {k: v[idx] for k, v in self.Y.items()}
        return self.X[idx], self.Y[idx]


def compute_sequence_norm_stats(X: np.ndarray, Y: np.ndarray):
    """Return mean and std for sequence arrays including multi-task targets."""

    x_flat = X.reshape(-1, X.shape[-1])
    x_mean = torch.tensor(x_flat.mean(axis=0), dtype=torch.float32)
    x_std = torch.tensor(x_flat.std(axis=0) + 1e-8, dtype=torch.float32)

    first = Y[0]
    if isinstance(first, dict) or (isinstance(first, np.ndarray) and Y.dtype == object):
        node = np.concatenate([y["node_outputs"].reshape(-1, y["node_outputs"].shape[-1]) for y in Y], axis=0)
        edge = np.concatenate([y["edge_outputs"].reshape(-1, y["edge_outputs"].shape[-1]) for y in Y], axis=0)
        y_mean = {
            "node_outputs": torch.tensor(node.mean(axis=0), dtype=torch.float32),
            "edge_outputs": torch.tensor(edge.mean(axis=0), dtype=torch.float32),
        }
        y_std = {
            "node_outputs": torch.tensor(node.std(axis=0) + 1e-8, dtype=torch.float32),
            "edge_outputs": torch.tensor(edge.std(axis=0) + 1e-8, dtype=torch.float32),
        }
    else:
        y_flat = Y.reshape(-1, Y.shape[-1])
        y_mean = torch.tensor(y_flat.mean(axis=0), dtype=torch.float32)
        y_std = torch.tensor(y_flat.std(axis=0) + 1e-8, dtype=torch.float32)

    return x_mean, x_std, y_mean, y_std


def apply_sequence_normalization(
    dataset: SequenceDataset,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean,
    y_std,
    edge_attr_mean: Optional[torch.Tensor] = None,
    edge_attr_std: Optional[torch.Tensor] = None,
) -> None:
    dataset.X = (dataset.X - x_mean) / x_std
    if dataset.multi:
        for k in dataset.Y:
            if k in y_mean:
                dataset.Y[k] = (dataset.Y[k] - y_mean[k]) / y_std[k]
    else:
        dataset.Y = (dataset.Y - y_mean) / y_std
    if edge_attr_mean is not None and dataset.edge_attr is not None:
        dataset.edge_attr = (dataset.edge_attr - edge_attr_mean) / edge_attr_std


def build_edge_attr(
    wn: wntr.network.WaterNetworkModel, edge_index: np.ndarray
) -> np.ndarray:
    """Return edge attribute matrix [E,3] for given edge index."""
    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    attr_dict: dict[tuple[int, int], list[float]] = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_map[link.start_node.name]
        j = node_map[link.end_node.name]
        length = getattr(link, "length", 0.0) or 0.0
        diam = getattr(link, "diameter", 0.0) or 0.0
        rough = getattr(link, "roughness", 0.0) or 0.0
        attr = [float(length), float(diam), float(rough)]
        attr_dict[(i, j)] = attr
        attr_dict[(j, i)] = attr
    return np.array([attr_dict[(int(s), int(t))] for s, t in edge_index.T], dtype=np.float32)


def build_edge_type(
    wn: wntr.network.WaterNetworkModel, edge_index: np.ndarray
) -> np.ndarray:
    """Return integer edge type array matching ``edge_index``."""

    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    type_dict: dict[tuple[int, int], int] = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_map[link.start_node.name]
        j = node_map[link.end_node.name]
        if link_name in wn.pipe_name_list:
            t = 0  # pipe connection
        elif link_name in wn.pump_name_list:
            t = 1  # pump connection
        elif link_name in wn.valve_name_list:
            t = 2  # valve connection
        else:
            t = 0
        type_dict[(i, j)] = t
        type_dict[(j, i)] = t

    return np.array([type_dict[(int(s), int(t))] for s, t in edge_index.T], dtype=np.int64)


def build_edge_pairs(edge_index: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (i, j) tuples pairing forward and reverse edges."""

    pair_map: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    for eid in range(edge_index.shape[1]):
        u = int(edge_index[0, eid])
        v = int(edge_index[1, eid])
        if (v, u) in pair_map:
            j = pair_map[(v, u)]
            pairs.append((j, eid))
        else:
            pair_map[(u, v)] = eid
    return pairs


def build_node_type(wn: wntr.network.WaterNetworkModel) -> np.ndarray:
    """Return integer node type array for the network nodes."""

    types = []
    for n in wn.node_name_list:
        if n in wn.junction_name_list:
            types.append(0)
        elif n in wn.tank_name_list:
            types.append(1)
        elif n in wn.reservoir_name_list:
            types.append(2)
        else:
            # Pumps and valves are modeled as edges; keep unique index
            types.append(0)
    return np.array(types, dtype=np.int64)


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


def handle_keyboard_interrupt(model_path: str) -> None:
    """Notify user that training was interrupted."""
    print(f"Training interrupted. Using best weights saved to {model_path}")


def partition_graph_greedy(edge_index: np.ndarray, num_nodes: int, cluster_size: int) -> list[np.ndarray]:
    """Split ``edge_index`` into clusters using greedy modularity heuristics."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((int(s), int(t)) for s, t in edge_index.T)
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    clusters: list[list[int]] = []
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

    def __init__(self, base_data: list[Data], clusters: list[np.ndarray]):
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

    def __init__(self, base_data: list[Data], edge_index: np.ndarray, sample_size: int, num_hops: int = 1):
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


def train(model, loader, optimizer, device, check_negative=True):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        if torch.isnan(batch.x).any() or torch.isnan(batch.y).any():
            raise ValueError("NaN detected in training batch")
        if check_negative and ((batch.x[:, 1] < 0).any() or (batch.y[:, 0] < 0).any()):
            raise ValueError("Negative pressures encountered in training batch")
        optimizer.zero_grad()
        out = model(
            batch.x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            getattr(batch, "node_type", None),
            getattr(batch, "edge_type", None),
        )
        loss = F.mse_loss(out, batch.y.float())
        loss.backward()
        # Clip gradients to mitigate exploding gradients that could otherwise
        # result in ``NaN`` loss values when the optimizer updates the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                getattr(batch, "node_type", None),
                getattr(batch, "edge_type", None),
            )
            loss = F.mse_loss(out, batch.y.float())
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def train_sequence(
    model: nn.Module,
    loader: TorchLoader,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_attr_phys: torch.Tensor,
    node_type: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    edge_pairs: list[tuple[int, int]],
    optimizer,
    device,
    physics_loss: bool = False,
    pressure_loss: bool = False,
    node_mask: Optional[torch.Tensor] = None,
    w_mass: float = 1.0,
    w_head: float = 1.0,
    w_edge: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    model.train()
    total_loss = 0.0
    node_total = edge_total = mass_total = head_total = sym_total = 0.0
    node_count = int(edge_index.max()) + 1
    for X_seq, Y_seq in loader:
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
        optimizer.zero_grad()
        preds = model(
            X_seq,
            edge_index.to(device),
            edge_attr.to(device),
            nt,
            et,
        )
        if isinstance(Y_seq, dict):
            target_nodes = Y_seq['node_outputs'].to(device)
            pred_nodes = preds['node_outputs']
            if node_mask is not None:
                pred_nodes = pred_nodes[:, :, node_mask, :]
                target_nodes = target_nodes[:, :, node_mask, :]
            loss_node = F.mse_loss(pred_nodes, target_nodes)
            edge_target = Y_seq['edge_outputs'].unsqueeze(-1).to(device)
            loss_edge = F.mse_loss(preds['edge_outputs'], edge_target)
            if physics_loss:
                flows_mb = (
                    preds['edge_outputs'].squeeze(-1).permute(2, 0, 1).reshape(
                        edge_index.size(1), -1
                    )
                )
                demand_mb = (
                    X_seq[..., 0].permute(2, 0, 1).reshape(node_count, -1)
                )
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        q_mean = model.y_mean["edge_outputs"][0].to(device)
                        q_std = model.y_std["edge_outputs"][0].to(device)
                    else:
                        q_mean = model.y_mean[-1].to(device)
                        q_std = model.y_std[-1].to(device)
                    flows_mb = flows_mb * q_std + q_mean
                if hasattr(model, "x_mean") and model.x_mean is not None:
                    dem_mean = model.x_mean[0].to(device)
                    dem_std = model.x_std[0].to(device)
                    demand_mb = demand_mb * dem_std + dem_mean
                mass_loss = compute_mass_balance_loss(
                    flows_mb,
                    edge_index.to(device),
                    node_count,
                    demand=demand_mb,
                    node_type=nt,
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
            if pressure_loss:
                press = preds['node_outputs'][..., 0]
                flow = preds['edge_outputs'].squeeze(-1)
                if hasattr(model, 'y_mean') and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        p_mean = model.y_mean['node_outputs'][0].to(device)
                        p_std = model.y_std['node_outputs'][0].to(device)
                        q_mean = model.y_mean['edge_outputs'][0].to(device)
                        q_std = model.y_std['edge_outputs'][0].to(device)
                        press = press * p_std + p_mean
                        flow = flow * q_std + q_mean
                    else:
                        press = press * model.y_std[0].to(device) + model.y_mean[0].to(device)
                head_loss = pressure_headloss_consistency_loss(
                    press,
                    flow,
                    edge_index.to(device),
                    edge_attr_phys.to(device),
                    edge_type=et,
                )
            else:
                head_loss = torch.tensor(0.0, device=device)
            loss = loss_node + w_edge * loss_edge
            if physics_loss:
                loss = loss + w_mass * (mass_loss + sym_loss)
            if pressure_loss:
                loss = loss + w_head * head_loss
        else:
            Y_seq = Y_seq.to(device)
            loss_node = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
            loss = F.mse_loss(preds, Y_seq.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X_seq.size(0)
        node_total += loss_node.item() * X_seq.size(0)
        edge_total += loss_edge.item() * X_seq.size(0)
        mass_total += mass_loss.item() * X_seq.size(0)
        head_total += head_loss.item() * X_seq.size(0)
        sym_total += sym_loss.item() * X_seq.size(0)
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        node_total / denom,
        edge_total / denom,
        mass_total / denom,
        head_total / denom,
        sym_total / denom,
    )


def evaluate_sequence(
    model: nn.Module,
    loader: TorchLoader,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_attr_phys: torch.Tensor,
    node_type: Optional[torch.Tensor],
    edge_type: Optional[torch.Tensor],
    edge_pairs: list[tuple[int, int]],
    device,
    physics_loss: bool = False,
    pressure_loss: bool = False,
    node_mask: Optional[torch.Tensor] = None,
    w_mass: float = 1.0,
    w_head: float = 1.0,
    w_edge: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    node_total = edge_total = mass_total = head_total = sym_total = 0.0
    node_count = int(edge_index.max()) + 1
    with torch.no_grad():
        for X_seq, Y_seq in loader:
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
            preds = model(
                X_seq,
                edge_index.to(device),
                edge_attr.to(device),
                nt,
                et,
            )
            if isinstance(Y_seq, dict):
                target_nodes = Y_seq['node_outputs'].to(device)
                pred_nodes = preds['node_outputs']
                if node_mask is not None:
                    pred_nodes = pred_nodes[:, :, node_mask, :]
                    target_nodes = target_nodes[:, :, node_mask, :]
                loss_node = F.mse_loss(pred_nodes, target_nodes)
                edge_target = Y_seq['edge_outputs'].unsqueeze(-1).to(device)
                loss_edge = F.mse_loss(preds['edge_outputs'], edge_target)
                if physics_loss:
                    flows_mb = (
                        preds['edge_outputs'].squeeze(-1).permute(2, 0, 1).reshape(
                            edge_index.size(1), -1
                        )
                    )
                    demand_mb = (
                        X_seq[..., 0].permute(2, 0, 1).reshape(node_count, -1)
                    )
                    if hasattr(model, "y_mean") and model.y_mean is not None:
                        if isinstance(model.y_mean, dict):
                            q_mean = model.y_mean["edge_outputs"][0].to(device)
                            q_std = model.y_std["edge_outputs"][0].to(device)
                        else:
                            q_mean = model.y_mean[-1].to(device)
                            q_std = model.y_std[-1].to(device)
                        flows_mb = flows_mb * q_std + q_mean
                    if hasattr(model, "x_mean") and model.x_mean is not None:
                        dem_mean = model.x_mean[0].to(device)
                        dem_std = model.x_std[0].to(device)
                        demand_mb = demand_mb * dem_std + dem_mean
                    mass_loss = compute_mass_balance_loss(
                        flows_mb,
                        edge_index.to(device),
                        node_count,
                        demand=demand_mb,
                        node_type=nt,
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
                if pressure_loss:
                    press = preds['node_outputs'][..., 0]
                    flow = preds['edge_outputs'].squeeze(-1)
                    if hasattr(model, 'y_mean') and model.y_mean is not None:
                        if isinstance(model.y_mean, dict):
                            p_mean = model.y_mean['node_outputs'][0].to(device)
                            p_std = model.y_std['node_outputs'][0].to(device)
                            q_mean = model.y_mean['edge_outputs'][0].to(device)
                            q_std = model.y_std['edge_outputs'][0].to(device)
                            press = press * p_std + p_mean
                            flow = flow * q_std + q_mean
                        else:
                            press = press * model.y_std[0].to(device) + model.y_mean[0].to(device)
                    head_loss = pressure_headloss_consistency_loss(
                        press,
                        flow,
                        edge_index.to(device),
                        edge_attr_phys.to(device),
                        edge_type=et,
                    )
                else:
                    head_loss = torch.tensor(0.0, device=device)
                loss = loss_node + w_edge * loss_edge
                if physics_loss:
                    loss = loss + w_mass * (mass_loss + sym_loss)
                if pressure_loss:
                    loss = loss + w_head * head_loss
            else:
                Y_seq = Y_seq.to(device)
                loss_node = loss_edge = mass_loss = sym_loss = torch.tensor(0.0, device=device)
                loss = F.mse_loss(preds, Y_seq.float())
            total_loss += loss.item() * X_seq.size(0)
            node_total += loss_node.item() * X_seq.size(0)
        edge_total += loss_edge.item() * X_seq.size(0)
        mass_total += mass_loss.item() * X_seq.size(0)
        head_total += head_loss.item() * X_seq.size(0)
        sym_total += sym_loss.item() * X_seq.size(0)
    denom = len(loader.dataset)
    return (
        total_loss / denom,
        node_total / denom,
        edge_total / denom,
        mass_total / denom,
        head_total / denom,
        sym_total / denom,
    )

# Resolve important directories relative to the repository root so that training
# can be launched from any location.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
PLOTS_DIR = REPO_ROOT / "plots"


def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    signal.signal(signal.SIGINT, _signal_handler)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index_np = np.load(args.edge_index_path)
    wn = wntr.network.WaterNetworkModel(args.inp_path)
    edge_attr = build_edge_attr(wn, edge_index_np)
    # log-transform roughness like in data generation
    edge_attr[:, 2] = np.log1p(edge_attr[:, 2])
    # preserve physical units before normalisation
    edge_attr_phys = torch.tensor(edge_attr.copy(), dtype=torch.float32)
    edge_types = build_edge_type(wn, edge_index_np)
    edge_pairs = build_edge_pairs(edge_index_np)
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
    if not seq_mode:
        first_label = Y_raw[0]
        if isinstance(first_label, dict) or (
            isinstance(first_label, np.ndarray) and Y_raw.dtype == object
        ):
            # Treat single-step multi-task data as sequences of length one
            X_raw = X_raw[:, None, ...]
            Y_raw = np.array([
                {k: v[None, ...] for k, v in y.items()} for y in Y_raw
            ], dtype=object)
            seq_mode = True

    if seq_mode:
        data_ds = SequenceDataset(
            X_raw,
            Y_raw,
            edge_index_np,
            edge_attr,
            node_type=node_types,
            edge_type=edge_types,
        )
        loader = TorchLoader(data_ds, batch_size=args.batch_size, shuffle=True)
    else:
        data_list = load_dataset(
            args.x_path,
            args.y_path,
            args.edge_index_path,
            edge_attr=edge_attr,
            node_type=node_types,
            edge_type=edge_types,
        )
        loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)


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
            val_loader = TorchLoader(val_ds, batch_size=args.batch_size)
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
            val_loader = DataLoader(val_list, batch_size=args.batch_size)
    else:
        val_list = []
        val_loader = None

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
            print(
                "Chlorine mean/std:",
                y_mean["node_outputs"][1].item(),
                y_std["node_outputs"][1].item(),
            )
        else:
            if len(y_mean) >= 2:
                print("Pressure mean/std:", y_mean[0].item(), y_std[0].item())
                print("Chlorine mean/std:", y_mean[1].item(), y_std[1].item())
    else:
        x_mean = x_std = y_mean = y_std = None

    if not seq_mode:
        if args.neighbor_sampling:
            sample_size = args.cluster_batch_size or max(1, int(0.2 * data_list[0].num_nodes))
            data_list = NeighborSampleDataset(data_list, edge_index_np, sample_size)
            loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)
            if val_loader is not None:
                val_loader = DataLoader(NeighborSampleDataset(val_list, edge_index_np, sample_size), batch_size=args.batch_size)
        elif args.cluster_batch_size > 0:
            clusters = partition_graph_greedy(edge_index_np, data_list[0].num_nodes, args.cluster_batch_size)
            data_list = ClusterSampleDataset(data_list, clusters)
            loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)
            if val_loader is not None:
                val_loader = DataLoader(ClusterSampleDataset(val_list, clusters), batch_size=args.batch_size)

    expected_in_dim = 4 + len(wn.pump_name_list)

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
                node_output_dim=2,
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # prepare logging
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(args.output)
    model_path = f"{base}_{run_name}{ext}"
    norm_path = f"{base}_{run_name}_norm.npz"
    log_path = os.path.join(DATA_DIR, f"training_{run_name}.log")
    losses = []
    with open(log_path, "w") as f:
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"args: {vars(args)}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"device: {device}\n")
        f.write("epoch,train_loss,val_loss,lr\n")
        best_val = float("inf")
        patience = 0
        for epoch in range(args.epochs):
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
                    physics_loss=args.physics_loss,
                    pressure_loss=args.pressure_loss,
                    node_mask=loss_mask,
                    w_mass=args.w_mass,
                    w_head=args.w_head,
                    w_edge=args.w_edge,
                )
                loss = loss_tuple[0]
                node_l, edge_l, mass_l, head_l, sym_l = loss_tuple[1:]
                if val_loader is not None:
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
                        physics_loss=args.physics_loss,
                        pressure_loss=args.pressure_loss,
                        node_mask=loss_mask,
                        w_mass=args.w_mass,
                        w_head=args.w_head,
                        w_edge=args.w_edge,
                    )
                    val_loss = val_tuple[0]
                else:
                    val_loss = loss
            else:
                loss = train(model, loader, optimizer, device, check_negative=not args.normalize)
                if val_loader is not None:
                    val_loss = evaluate(model, val_loader, device)
                else:
                    val_loss = loss
            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            losses.append((loss, val_loss))
            f.write(f"{epoch},{loss:.6f},{val_loss:.6f},{curr_lr:.6e}\n")
            if args.physics_loss and seq_mode:
                msg = (
                    f"Epoch {epoch}: node={node_l:.3f}, edge={edge_l:.3f}, "
                    f"mass={mass_l:.3f}, sym={sym_l:.3f}"
                )
                if args.pressure_loss:
                    msg += f", head={head_l:.3f}"
                print(msg)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience = 0
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                if x_mean is not None:
                    if isinstance(y_mean, dict):
                        np.savez(
                            norm_path,
                            x_mean=x_mean.numpy(),
                            x_std=x_std.numpy(),
                            y_mean_node=y_mean["node_outputs"].numpy(),
                            y_std_node=y_std["node_outputs"].numpy(),
                            y_mean_edge=y_mean["edge_outputs"].numpy(),
                            y_std_edge=y_std["edge_outputs"].numpy(),
                            edge_mean=edge_mean.numpy(),
                            edge_std=edge_std.numpy(),
                        )
                    else:
                        np.savez(
                            norm_path,
                            x_mean=x_mean.numpy(),
                            x_std=x_std.numpy(),
                            y_mean=y_mean.numpy(),
                            y_std=y_std.numpy(),
                            edge_mean=edge_mean.numpy(),
                            edge_std=edge_std.numpy(),
                        )
            else:
                patience += 1
            if interrupted:
                break
            if patience >= args.early_stop_patience:
                break
        if interrupted:
            handle_keyboard_interrupt(model_path)

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
            test_loader = TorchLoader(test_ds, batch_size=args.batch_size)
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
            test_loader = DataLoader(test_list, batch_size=args.batch_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        preds_p = []
        preds_c = []
        true_p = []
        true_c = []
        with torch.no_grad():
            if seq_mode:
                ei = test_ds.edge_index.to(device)
                ea = test_ds.edge_attr.to(device) if test_ds.edge_attr is not None else None
                nt = test_ds.node_type.to(device) if test_ds.node_type is not None else None
                et = test_ds.edge_type.to(device) if test_ds.edge_type is not None else None
                for X_seq, Y_seq in test_loader:
                    X_seq = X_seq.to(device)
                    out = model(X_seq, ei, ea, nt, et)
                    if isinstance(out, dict):
                        node_pred = out["node_outputs"]
                    else:
                        node_pred = out
                    if isinstance(Y_seq, dict):
                        Y_node = Y_seq["node_outputs"].to(node_pred.device)
                    else:
                        Y_node = Y_seq.to(node_pred.device)
                    if hasattr(model, "y_mean") and model.y_mean is not None:
                        if isinstance(model.y_mean, dict):
                            y_mean_node = model.y_mean['node_outputs'].to(node_pred.device)
                            y_std_node = model.y_std['node_outputs'].to(node_pred.device)
                            node_pred = node_pred * y_std_node + y_mean_node
                            Y_node = Y_node * y_std_node + y_mean_node
                        else:
                            y_std = model.y_std.to(node_pred.device)
                            y_mean = model.y_mean.to(node_pred.device)
                            node_pred = node_pred * y_std + y_mean
                            Y_node = Y_node * y_std + y_mean
                    preds_p.extend(node_pred[..., 0].cpu().numpy().ravel())
                    preds_c.extend(node_pred[..., 1].cpu().numpy().ravel())
                    true_p.extend(Y_node[..., 0].cpu().numpy().ravel())
                    true_c.extend(Y_node[..., 1].cpu().numpy().ravel())
            else:
                for batch in test_loader:
                    batch = batch.to(device)
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
                            out = out * y_std_node + y_mean_node
                            batch_y = batch.y * y_std_node + y_mean_node
                        else:
                            y_std = model.y_std.to(out.device)
                            y_mean = model.y_mean.to(out.device)
                            out = out * y_std + y_mean
                            batch_y = batch.y * y_std + y_mean
                    else:
                        batch_y = batch.y
                    preds_p.extend(out[:, 0].cpu().numpy())
                    preds_c.extend(out[:, 1].cpu().numpy())
                    true_p.extend(batch_y[:, 0].cpu().numpy())
                    true_c.extend(batch_y[:, 1].cpu().numpy())
        if preds_p:
            preds_p = np.array(preds_p)
            preds_c = np.array(preds_c)
            true_p = np.array(true_p)
            true_c = np.array(true_c)
            save_scatter_plots(true_p, preds_p, true_c, preds_c, run_name)

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
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension",
    )
    parser.add_argument("--use-attention", action="store_true",
                        help="Use GATConv instead of HydroConv for graph convolution")
    parser.add_argument("--gat-heads", type=int, default=4,
                        help="Number of attention heads for GATConv (if attention is enabled)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate applied after each GNN layer")
    parser.add_argument("--num-layers", type=int, default=4,
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
        "--rnn-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of the recurrent layer when sequence data is used",
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
    parser.set_defaults(pressure_loss=True)
    parser.add_argument(
        "--w_mass",
        type=float,
        default=1.0,
        help="Weight of the mass conservation loss term",
    )
    parser.add_argument(
        "--w_head",
        type=float,
        default=1.0,
        help="Weight of the head loss consistency term",
    )
    parser.add_argument(
        "--w_edge",
        type=float,
        default=1.0,
        help="Weight of the edge (flow) loss term",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", default="", help="Optional run name")
    parser.add_argument(
        "--inp-path",
        default=os.path.join(REPO_ROOT, "CTown.inp"),
        help="EPANET input file used to determine the number of pumps",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(MODELS_DIR, "gnn_surrogate.pth"),
        help="Output model file base path",
    )
    args = parser.parse_args()
    main(args)
