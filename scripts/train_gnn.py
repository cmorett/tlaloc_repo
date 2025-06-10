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
from torch_geometric.nn import GCNConv, MessagePassing
import wntr
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


class HydroConv(MessagePassing):
    """Mass-conserving convolution using edge attributes."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr="add")
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


class GCNEncoder(nn.Module):
    """Flexible GCN architecture used for surrogate training."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        residual: bool = False,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.edge_dim = edge_dim
        conv_cls = (
            (lambda in_c, out_c: HydroConv(in_c, out_c, edge_dim))
            if edge_dim is not None
            else (lambda in_c, out_c: GCNConv(in_c, out_c))
        )

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(conv_cls(in_channels, out_channels))
        else:
            self.layers.append(conv_cls(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(conv_cls(hidden_channels, hidden_channels))
            self.layers.append(conv_cls(hidden_channels, out_channels))

        # Expose first/last conv for backward compatibility with existing code
        self.conv1 = self.layers[0]
        self.conv2 = self.layers[-1]
        self.dropout = dropout
        self.residual = residual
        self.act_fn = getattr(F, activation)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        for i, conv in enumerate(self.layers):
            residual = x
            if isinstance(conv, HydroConv):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act_fn(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                if self.residual:
                    x = x + residual
        return x


def load_dataset(
    x_path: str,
    y_path: str,
    edge_index_path: str = "edge_index.npy",
    edge_attr: np.ndarray | None = None,
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
    data_list = load_dataset(
        args.x_path, args.y_path, args.edge_index_path, edge_attr=edge_attr
    )
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)

    if args.x_val_path and os.path.exists(args.x_val_path):
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
        x_mean, x_std, y_mean, y_std = compute_norm_stats(data_list)
        apply_normalization(data_list, x_mean, x_std, y_mean, y_std)
        if val_list:
            apply_normalization(val_list, x_mean, x_std, y_mean, y_std)
    else:
        x_mean = x_std = y_mean = y_std = None

    expected_in_dim = 4 + len(wn.pump_name_list)

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
    parser.add_argument("--num-layers", type=int, default=2, help="GCN layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")
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
