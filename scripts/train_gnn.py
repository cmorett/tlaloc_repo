import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader as TorchLoader
from torch_geometric.nn import GCNConv, MessagePassing, GATConv, LayerNorm
import wntr
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional


class HydroConv(MessagePassing):
    """Mass-conserving convolution using edge attributes."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.lin(self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        weight = self.edge_mlp(edge_attr).view(-1, 1)
        return weight * (x_j - x_i)


class EnhancedGNNEncoder(nn.Module):
    """GNN encoder supporting attention and residual connections."""

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
    ) -> None:
        super().__init__()

        self.use_attention = use_attention
        self.dropout = dropout
        self.residual = residual
        self.act_fn = getattr(F, activation)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if use_attention:
            self.convs.append(
                GATConv(in_channels, hidden_channels // gat_heads, heads=gat_heads, edge_dim=edge_dim)
            )
        else:
            conv_cls = (
                (lambda in_c, out_c: HydroConv(in_c, out_c, edge_dim))
                if edge_dim is not None
                else (lambda in_c, out_c: GCNConv(in_c, out_c))
            )
            self.convs.append(conv_cls(in_channels, hidden_channels))

        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, edge_dim=edge_dim)
                )
            else:
                self.convs.append(conv_cls(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        self.fc_out = nn.Linear(hidden_channels, out_channels)

        self.conv1 = self.convs[0]
        self.conv2 = self.convs[-1]

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        for conv, norm in zip(self.convs, self.norms):
            identity = x
            if isinstance(conv, HydroConv) or not self.use_attention:
                if isinstance(conv, HydroConv):
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_attr)
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
        )
        self.rnn = nn.LSTM(hidden_channels, rnn_hidden_dim, batch_first=True)
        self.decoder = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(
        self,
        X_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        # Expand edge index for ``batch_size`` separate graphs
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        if edge_attr is not None:
            edge_attr_rep = edge_attr.repeat(batch_size, 1)
        else:
            edge_attr_rep = None

        node_embeddings = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)
            data = Data(x=x_t, edge_index=batch_edge_index)
            if edge_attr_rep is not None:
                data.edge_attr = edge_attr_rep
            gnn_out = self.encoder(data)  # [batch_size*num_nodes, hidden]
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)
            node_embeddings.append(gnn_out)

        emb = torch.stack(node_embeddings, dim=1)  # [B, T, N, H]
        # reshape so each node has its own sequence in the batch dimension
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)
        out = self.decoder(rnn_out)
        return out


class MultiTaskGNNSurrogate(nn.Module):
    """Recurrent GNN surrogate predicting node and edge level targets."""

    def __init__(self, in_channels: int, hidden_channels: int, edge_dim: int, node_output_dim: int, edge_output_dim: int, energy_output_dim: int, num_layers: int, use_attention: bool, gat_heads: int, dropout: float, residual: bool, rnn_hidden_dim: int) -> None:
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
        )
        self.rnn = nn.LSTM(hidden_channels, rnn_hidden_dim, batch_first=True)
        self.node_decoder = nn.Linear(rnn_hidden_dim, node_output_dim)
        self.edge_decoder = nn.Linear(rnn_hidden_dim * 2, edge_output_dim)
        self.energy_decoder = nn.Linear(rnn_hidden_dim, energy_output_dim)

    def forward(self, X_seq: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        edge_attr_rep = edge_attr.repeat(batch_size, 1) if edge_attr is not None else None

        node_embeddings = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)
            data = Data(x=x_t, edge_index=batch_edge_index)
            if edge_attr_rep is not None:
                data.edge_attr = edge_attr_rep
            gnn_out = self.encoder(data)
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)
            node_embeddings.append(gnn_out)

        emb = torch.stack(node_embeddings, dim=1)
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)

        node_pred = self.node_decoder(rnn_out)

        src = edge_index[0]
        tgt = edge_index[1]
        h_src = rnn_out[:, :, src, :]
        h_tgt = rnn_out[:, :, tgt, :]
        edge_emb = torch.cat([h_src, h_tgt], dim=-1)
        edge_pred = self.edge_decoder(edge_emb)
        energy_pred = self.energy_decoder(rnn_out.mean(dim=2))

        return {
            "node_outputs": node_pred,
            "edge_outputs": edge_pred,
            "pump_energy": energy_pred,
        }


def load_dataset(
    x_path: str,
    y_path: str,
    edge_index_path: str = "edge_index.npy",
    edge_attr: Optional[np.ndarray] = None,
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
    if edge_attr is not None:
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)

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


def apply_normalization(data_list, x_mean, x_std, y_mean, y_std):
    for d in data_list:
        d.x = (d.x - x_mean) / x_std
        d.y = (d.y - y_mean) / y_std


class SequenceDataset(Dataset):
    """Simple ``Dataset`` for sequence data supporting multi-task labels."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, edge_index: np.ndarray, edge_attr: Optional[np.ndarray]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_attr = None
        if edge_attr is not None:
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        first = Y[0]
        if isinstance(first, dict) or (isinstance(first, np.ndarray) and Y.dtype == object):
            self.multi = True
            self.Y = {
                "node_outputs": torch.stack([torch.tensor(y["node_outputs"], dtype=torch.float32) for y in Y]),
                "edge_outputs": torch.stack([torch.tensor(y["edge_outputs"], dtype=torch.float32) for y in Y]),
                "pump_energy": torch.stack([torch.tensor(y["pump_energy"], dtype=torch.float32) for y in Y]),
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
        energy = np.concatenate([y["pump_energy"].reshape(-1, y["pump_energy"].shape[-1]) for y in Y], axis=0)
        y_mean = {
            "node_outputs": torch.tensor(node.mean(axis=0), dtype=torch.float32),
            "edge_outputs": torch.tensor(edge.mean(axis=0), dtype=torch.float32),
            "pump_energy": torch.tensor(energy.mean(axis=0), dtype=torch.float32),
        }
        y_std = {
            "node_outputs": torch.tensor(node.std(axis=0) + 1e-8, dtype=torch.float32),
            "edge_outputs": torch.tensor(edge.std(axis=0) + 1e-8, dtype=torch.float32),
            "pump_energy": torch.tensor(energy.std(axis=0) + 1e-8, dtype=torch.float32),
        }
    else:
        y_flat = Y.reshape(-1, Y.shape[-1])
        y_mean = torch.tensor(y_flat.mean(axis=0), dtype=torch.float32)
        y_std = torch.tensor(y_flat.std(axis=0) + 1e-8, dtype=torch.float32)

    return x_mean, x_std, y_mean, y_std


def apply_sequence_normalization(dataset: SequenceDataset, x_mean: torch.Tensor, x_std: torch.Tensor, y_mean, y_std) -> None:
    dataset.X = (dataset.X - x_mean) / x_std
    if dataset.multi:
        for k in dataset.Y:
            dataset.Y[k] = (dataset.Y[k] - y_mean[k]) / y_std[k]
    else:
        dataset.Y = (dataset.Y - y_mean) / y_std


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
        out = model(batch)
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
            out = model(batch)
            loss = F.mse_loss(out, batch.y.float())
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def train_sequence(model: nn.Module, loader: TorchLoader, edge_index: torch.Tensor, edge_attr: torch.Tensor, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for X_seq, Y_seq in loader:
        X_seq = X_seq.to(device)
        optimizer.zero_grad()
        preds = model(X_seq, edge_index.to(device), edge_attr.to(device))
        if isinstance(Y_seq, dict):
            loss_node = F.mse_loss(
                preds['node_outputs'],
                Y_seq['node_outputs'].to(device)
            )
            # ``edge_outputs`` in ``SequenceDataset`` does not include the final
            # feature dimension, whereas the model predicts ``[B, T, E, 1]``.
            # Add the singleton dimension here so broadcasting does not occur
            # during loss computation.
            edge_target = Y_seq['edge_outputs'].unsqueeze(-1).to(device)
            loss_edge = F.mse_loss(preds['edge_outputs'], edge_target)
            loss_energy = F.mse_loss(preds['pump_energy'], Y_seq['pump_energy'].to(device))
            loss = loss_node + 0.5 * loss_edge + 0.1 * loss_energy
        else:
            Y_seq = Y_seq.to(device)
            loss = F.mse_loss(preds, Y_seq.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X_seq.size(0)
    return total_loss / len(loader.dataset)


def evaluate_sequence(model: nn.Module, loader: TorchLoader, edge_index: torch.Tensor, edge_attr: torch.Tensor, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_seq, Y_seq in loader:
            X_seq = X_seq.to(device)
            preds = model(X_seq, edge_index.to(device), edge_attr.to(device))
            if isinstance(Y_seq, dict):
                loss_node = F.mse_loss(
                    preds['node_outputs'],
                    Y_seq['node_outputs'].to(device)
                )
                edge_target = Y_seq['edge_outputs'].unsqueeze(-1).to(device)
                loss_edge = F.mse_loss(preds['edge_outputs'], edge_target)
                loss_energy = F.mse_loss(
                    preds['pump_energy'],
                    Y_seq['pump_energy'].to(device)
                )
                loss = loss_node + 0.5 * loss_edge + 0.1 * loss_energy
            else:
                Y_seq = Y_seq.to(device)
                loss = F.mse_loss(preds, Y_seq.float())
            total_loss += loss.item() * X_seq.size(0)
    return total_loss / len(loader.dataset)

# Resolve important directories relative to the repository root so that training
# can be launched from any location.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"


def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index_np = np.load(args.edge_index_path)
    wn = wntr.network.WaterNetworkModel(args.inp_path)
    edge_attr = build_edge_attr(wn, edge_index_np)
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
        data_ds = SequenceDataset(X_raw, Y_raw, edge_index_np, edge_attr)
        loader = TorchLoader(data_ds, batch_size=args.batch_size, shuffle=True)
    else:
        data_list = load_dataset(
            args.x_path, args.y_path, args.edge_index_path, edge_attr=edge_attr
        )
        loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)


    if args.x_val_path and os.path.exists(args.x_val_path):
        if seq_mode:
            Xv = np.load(args.x_val_path, allow_pickle=True)
            Yv = np.load(args.y_val_path, allow_pickle=True)
            if Xv.ndim == 3:
                Yv = np.array([{k: v[None, ...] for k, v in y.items()} for y in Yv], dtype=object)
                Xv = Xv[:, None, ...]
            val_ds = SequenceDataset(Xv, Yv, edge_index_np, edge_attr)
            val_loader = TorchLoader(val_ds, batch_size=args.batch_size)
            val_list = val_ds
        else:
            val_list = load_dataset(
                args.x_val_path,
                args.y_val_path,
                args.edge_index_path,
                edge_attr=edge_attr,
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
            apply_sequence_normalization(data_ds, x_mean, x_std, y_mean, y_std)
            if isinstance(val_list, SequenceDataset):
                apply_sequence_normalization(val_list, x_mean, x_std, y_mean, y_std)
        else:
            x_mean, x_std, y_mean, y_std = compute_norm_stats(data_list)
            apply_normalization(data_list, x_mean, x_std, y_mean, y_std)
            if val_list:
                apply_normalization(val_list, x_mean, x_std, y_mean, y_std)
    else:
        x_mean = x_std = y_mean = y_std = None

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
                energy_output_dim=len(wn.pump_name_list),
                num_layers=args.num_layers,
                use_attention=args.use_attention,
                gat_heads=args.gat_heads,
                dropout=args.dropout,
                residual=args.residual,
                rnn_hidden_dim=args.rnn_hidden_dim,
            ).to(device)
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
            ).to(device)
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
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
                loss = train_sequence(model, loader, data_ds.edge_index, data_ds.edge_attr, optimizer, device)
                if val_loader is not None:
                    val_loss = evaluate_sequence(model, val_loader, data_ds.edge_index, data_ds.edge_attr, device)
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
                            y_mean_energy=y_mean["pump_energy"].numpy(),
                            y_std_energy=y_std["pump_energy"].numpy(),
                        )
                    else:
                        np.savez(norm_path, x_mean=x_mean.numpy(), x_std=x_std.numpy(), y_mean=y_mean.numpy(), y_std=y_std.numpy())
            else:
                patience += 1
            if patience >= args.early_stop_patience:
                break

    # plot loss curve
    if losses:
        tr = [l[0] for l in losses]
        vl = [l[1] for l in losses]
        plt.figure()
        plt.plot(tr, label="train")
        plt.plot(vl, label="val")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, f"training_loss_{run_name}.png"))
        plt.close()

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
    parser.add_argument("--normalize", action="store_true", help="Normalize data")
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
