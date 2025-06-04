import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    """Minimal two layer GCN used for examples."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def load_dataset(x_path: str, y_path: str, edge_index_path: str) -> list[Data]:
    """Load graphs saved as numpy arrays.

    The function supports two formats:
    1. ``X`` contains dictionaries with ``edge_index`` and ``node_features``.
    2. ``X`` is an array of node feature matrices and ``edge_index`` is stored
       separately in ``edge_index_path``.
    """

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    data_list: list[Data] = []
    if isinstance(X[0], dict):
        for graph_dict, label in zip(X, y):
            edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
            node_feat = torch.tensor(graph_dict["node_features"], dtype=torch.float)
            data_list.append(Data(x=node_feat, edge_index=edge_index, y=torch.tensor(label)))
    else:
        shared_edge_index = torch.tensor(np.load(edge_index_path), dtype=torch.long)
        for node_feat, label in zip(X, y):
            data_list.append(
                Data(x=torch.tensor(node_feat, dtype=torch.float), edge_index=shared_edge_index, y=torch.tensor(label))
            )
    return data_list


def train(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out.squeeze(), batch.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = load_dataset(args.x_path, args.y_path, args.edge_index_path)
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)

    sample = data_list[0]
    model = SimpleGCN(
        in_channels=sample.num_node_features,
        hidden_channels=args.hidden_dim,
        out_channels=args.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train(model, loader, optimizer, device)
        if (epoch + 1) % args.log_every == 0:
            print(f"Epoch {epoch + 1:03d}\tLoss: {loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple GCN model")
    parser.add_argument("--x-path", default="X_train.npy", help="Path to graph feature file")
    parser.add_argument("--y-path", default="y_train.npy", help="Path to label file")
    parser.add_argument("--edge-index-path", default="edge_index.npy", help="Path to shared edge index")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=10, help="Log every n epochs")
    parser.add_argument("--output", default="model.pt", help="Output model file")
    main(parser.parse_args())
