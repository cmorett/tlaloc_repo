import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import SequenceDataset, train_sequence


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X_seq, edge_index, edge_attr, nt, et):
        B, T, N, _ = X_seq.size()
        E = edge_index.size(1)
        node_out = torch.zeros(B, T, N, 2, device=X_seq.device) + self.param
        edge_out = torch.zeros(B, T, E, 1, device=X_seq.device) + self.param
        return {"node_outputs": node_out, "edge_outputs": edge_out}


def test_train_sequence_nan_detection():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 10), dtype=torch.float32)
    T, N, E = 1, 2, 2
    X = np.ones((1, T, N, 4), dtype=np.float32)
    X[0, 0, 0, 0] = np.nan
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((T, N, 2), dtype=np.float32),
                "edge_outputs": np.zeros((T, E), dtype=np.float32),
            }
        ],
        dtype=object,
    )
    ds = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())
    loader = TorchLoader(ds, batch_size=1)
    model = DummyModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    with pytest.raises(ValueError):
        train_sequence(
            model,
            loader,
            ds.edge_index,
            ds.edge_attr,
            edge_attr,
            None,
            None,
            [(0, 1)],
            opt,
            torch.device("cpu"),
            physics_loss=False,
            pressure_loss=False,
            node_mask=None,
        )
