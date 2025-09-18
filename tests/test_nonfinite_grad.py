import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchLoader
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.train_gnn import GCNEncoder, train, train_sequence


class DummySequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, X_seq, edge_index, attr_input, node_type, edge_type):
        return self.linear(X_seq)


class DummySequenceDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx):
        X_seq = torch.ones(1, 2, 1, dtype=torch.float32)
        Y_seq = torch.zeros(1, 2, 1, dtype=torch.float32)
        return X_seq, Y_seq


@pytest.mark.parametrize("amp", [False, True])
def test_train_skips_nonfinite_gradients(monkeypatch, amp):
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = Data(
        x=torch.ones(2, 2, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.zeros(2, 1, dtype=torch.float32),
    )
    loader = DataLoader([batch], batch_size=1)
    model = GCNEncoder(2, 4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    params_before = [p.detach().clone() for p in model.parameters()]

    original_clip = torch.nn.utils.clip_grad_norm_

    def fake_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
        original_clip(
            parameters,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
        )
        return torch.tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)

    _, _, _, _, avg_grad = train(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        amp=amp,
        progress=False,
    )

    params_after = [p.detach() for p in model.parameters()]

    for before, after in zip(params_before, params_after):
        assert torch.allclose(before, after)

    assert avg_grad is None


@pytest.mark.parametrize("amp", [False, True])
def test_train_sequence_skips_nonfinite_gradients(monkeypatch, amp):
    dataset = DummySequenceDataset()
    loader = TorchLoader(dataset, batch_size=1)
    model = DummySequenceModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    params_before = [p.detach().clone() for p in model.parameters()]

    original_clip = torch.nn.utils.clip_grad_norm_

    def fake_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
        original_clip(
            parameters,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
        )
        return torch.tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr_phys = torch.zeros((edge_index.size(1), 1), dtype=torch.float32)

    results = train_sequence(
        model,
        loader,
        edge_index,
        edge_attr=None,
        edge_attr_phys=edge_attr_phys,
        node_type=None,
        edge_type=None,
        edge_pairs=[],
        optimizer=optimizer,
        device=torch.device("cpu"),
        pump_coeffs=None,
        physics_loss=False,
        pressure_loss=False,
        pump_loss=False,
        amp=amp,
        progress=False,
    )

    avg_grad = results[-1]

    params_after = [p.detach() for p in model.parameters()]

    for before, after in zip(params_before, params_after):
        assert torch.allclose(before, after)

    assert avg_grad is None
