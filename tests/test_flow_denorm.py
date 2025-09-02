import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import SequenceDataset, train_sequence

class DummyModel(torch.nn.Module):
    def __init__(self, edge_pred, node_pred, y_mean, y_std):
        super().__init__()
        self.edge_pred = edge_pred
        self.node_pred = node_pred
        self.y_mean = y_mean
        self.y_std = y_std
        # single parameter so optimizer is not empty
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X_seq, edge_index, edge_attr, nt, et):
        edge_out = self.edge_pred + 0 * self.dummy
        node_out = self.node_pred + 0 * self.dummy
        return {"node_outputs": node_out, "edge_outputs": edge_out}


def test_mass_balance_denorm_per_edge():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 3), dtype=torch.float32)
    T = 1
    N = 2
    E = 2
    X = np.zeros((1, T, N, 4), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((T, N, 2), dtype=np.float32),
                "edge_outputs": np.zeros((T, E), dtype=np.float32),
                "demand": np.zeros((T, N), dtype=np.float32),
            }
        ],
        dtype=object,
    )
    ds = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())
    loader = TorchLoader(ds, batch_size=1)
    y_mean = {"edge_outputs": torch.ones(E), "node_outputs": torch.zeros(2)}
    y_std = {"edge_outputs": torch.ones(E), "node_outputs": torch.ones(2)}
    edge_pred = torch.zeros(1, T, E, 1)
    node_pred = torch.zeros(1, T, N, 2)
    model = DummyModel(edge_pred, node_pred, y_mean, y_std)
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    losses = train_sequence(
        model,
        loader,
        ds.edge_index,
        ds.edge_attr,
        ds.edge_attr,
        None,
        None,
        [(0, 1)],
        opt,
        torch.device("cpu"),
        physics_loss=True,
        pressure_loss=False,
    )
    assert abs(losses[3]) < 1e-6


def test_edge_denorm_shapes_match():
    edge_pred = torch.zeros(1, 1, 2, 1)
    y_std_edge = torch.ones(2)
    y_mean_edge = torch.zeros(2)
    edge_pred = edge_pred.squeeze(-1)
    edge_pred = edge_pred * y_std_edge + y_mean_edge
    true_edge = torch.zeros(1, 1, 2)
    true_edge = true_edge * y_std_edge + y_mean_edge
    assert edge_pred.shape == true_edge.shape
