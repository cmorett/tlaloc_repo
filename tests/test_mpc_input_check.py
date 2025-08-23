import sys
from pathlib import Path
import torch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import simulate_closed_loop, load_network, GNNSurrogate

class DummyConv(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
    def forward(self, x, edge_index, edge_attr=None):
        return x

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyConv(4)])  # only basic features
    def forward(self, x, edge_index, edge_attr=None):
        return torch.zeros(x.size(0), 2)


def test_simulate_closed_loop_requires_pump_inputs():
    device = torch.device("cpu")
    (
        wn,
        node_to_index,
        pump_names,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        template,
    ) = load_network("CTown.inp", return_edge_attr=True, return_features=True)
    model = DummyModel().to(device)
    with pytest.raises(ValueError):
        simulate_closed_loop(
            wn,
            model,
            edge_index.to(device),
            edge_attr.to(device),
            template.to(device),
            torch.tensor(node_types, dtype=torch.long, device=device),
            torch.tensor(edge_types, dtype=torch.long, device=device),
            horizon=2,
            iterations=1,
            node_to_index=node_to_index,
            pump_names=pump_names,
            device=device,
            Pmin=20.0,
            feedback_interval=0,
        )


def test_simulate_closed_loop_checks_edge_dim():
    device = torch.device("cpu")
    (
        wn,
        node_to_index,
        pump_names,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        template,
    ) = load_network("CTown.inp", return_edge_attr=True, return_features=True)

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([DummyConv(4 + len(pump_names))])
            self.edge_dim = 2  # expect fewer edge attributes than provided

        def forward(self, x, edge_index, edge_attr=None, node_types=None, edge_types=None):
            return torch.zeros(x.size(0), 2)

    model = EdgeModel().to(device)
    with pytest.raises(ValueError):
        simulate_closed_loop(
            wn,
            model,
            edge_index.to(device),
            edge_attr.to(device),
            template.to(device),
            torch.tensor(node_types, dtype=torch.long, device=device),
            torch.tensor(edge_types, dtype=torch.long, device=device),
            horizon=2,
            iterations=1,
            node_to_index=node_to_index,
            pump_names=pump_names,
            device=device,
            Pmin=20.0,
            feedback_interval=0,
        )
