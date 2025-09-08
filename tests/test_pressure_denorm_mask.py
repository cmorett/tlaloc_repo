import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import train_sequence


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.y_mean = {"node_outputs": torch.tensor([0.0, 0.0])}
        self.y_std = {"node_outputs": torch.tensor([1.0, 1.0])}

    def forward(self, X_seq, edge_index, edge_attr, nt, et):
        B, T, N, _ = X_seq.shape
        E = edge_index.size(1)
        node_out = torch.zeros(B, T, N, 2) * self.scale
        edge_out = torch.zeros(B, T, E) * self.scale
        return {"node_outputs": node_out, "edge_outputs": edge_out}


def test_global_norm_with_node_mask():
    device = torch.device("cpu")
    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    N, E = 5, 4
    X_seq = torch.zeros(1, N, 3)
    Y_nodes = torch.zeros(1, N, 2)
    Y_edges = torch.zeros(1, E)
    dataset = [(X_seq, {"node_outputs": Y_nodes, "edge_outputs": Y_edges})]
    loader = DataLoader(dataset, batch_size=1)

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.zeros(E, 1)
    edge_attr_phys = edge_attr.clone()

    node_mask = torch.tensor([True, False, True, False, True])

    loss_tuple = train_sequence(
        model,
        loader,
        edge_index,
        edge_attr,
        edge_attr_phys,
        node_type=None,
        edge_type=None,
        edge_pairs=[],
        optimizer=opt,
        device=device,
        node_mask=node_mask,
    )

    assert torch.isfinite(torch.tensor(loss_tuple[0]))
    assert loss_tuple[10] == pytest.approx(0.0)
