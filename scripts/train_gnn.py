import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def load_dataset(x_path: str, y_path: str):
    """Load graph data saved as numpy object arrays.

    Each entry of ``X`` should be a dictionary containing ``edge_index`` and
    ``node_features`` arrays.
    """
    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    data_list = []
    for graph_dict, label in zip(X, y):
        edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(
            graph_dict['node_features'], dtype=torch.float
        )
        data = Data(x=node_feat, edge_index=edge_index, y=torch.tensor(label))
        data_list.append(data)
    return data_list


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out.squeeze(), batch.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def main(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = load_dataset(args.x_path, args.y_path)
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
            print(f"Epoch {epoch+1:03d} \t Loss: {loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple GCN model")
    parser.add_argument(
        "--x-path",
        default="X_train.npy",
        help="Path to graph feature file",
    )
    parser.add_argument(
        "--y-path",
        default="y_train.npy",
        help="Path to label file",
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes",
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
        "--output",
        default="model.pt",
        help="Output model file",
    )
    args = parser.parse_args()
    main(args)
