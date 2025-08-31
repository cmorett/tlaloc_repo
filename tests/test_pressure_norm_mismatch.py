import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import SequenceDataset, train_sequence


class DummyModel(torch.nn.Module):
    def __init__(self, node_pred, edge_pred, y_mean, y_std):
        super().__init__()
        self.node_pred = node_pred
        self.edge_pred = edge_pred
        self.y_mean = y_mean
        self.y_std = y_std
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X_seq, edge_index, edge_attr, nt, et):  # pragma: no cover - simple stub
        node_out = self.node_pred + 0 * self.dummy
        edge_out = self.edge_pred + 0 * self.dummy
        return {"node_outputs": node_out, "edge_outputs": edge_out}


def test_pressure_norm_mismatch():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 3), dtype=torch.float32)
    T, N, E = 1, 2, 2
    X = np.zeros((1, T, N, 4), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((T, N, 2), dtype=np.float32),
                "edge_outputs": np.zeros((T, E), dtype=np.float32),
            }
        ],
        dtype=object,
    )
    dataset = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())
    loader = TorchLoader(dataset, batch_size=1)

    node_pred = torch.zeros(1, T, N, 2)
    node_pred[..., 0] = 0.5
    edge_pred = torch.zeros(1, T, E, 1)
    # Provide normalization stats for three nodes while predictions only cover two
    y_mean = {
        "node_outputs": torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),
        "edge_outputs": torch.zeros(E),
    }
    y_std = {
        "node_outputs": torch.tensor([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
        "edge_outputs": torch.ones(E),
    }
    model = DummyModel(node_pred, edge_pred, y_mean, y_std)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    device = torch.device("cpu")
    loss_tuple = train_sequence(
        model,
        loader,
        dataset.edge_index,
        dataset.edge_attr,
        dataset.edge_attr,
        None,
        None,
        [],
        optimizer,
        device,
        w_flow=0.0,
        progress=False,
    )
    assert abs(loss_tuple[9] - 1.0) < 1e-6

