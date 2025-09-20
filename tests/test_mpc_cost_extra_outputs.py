import sys
from pathlib import Path

import torch
import wntr
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.mpc_control import compute_mpc_cost


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_mean = torch.zeros(5)
        self.x_std = torch.ones(5)
        self.y_mean = torch.zeros(2)
        self.y_std = torch.ones(2)

    def forward(self, x, edge_index, edge_attr=None, node_types=None, edge_types=None):
        n = x.size(0)
        base = torch.stack(
            [
                torch.full((n,), 30.0, device=x.device),
                torch.full((n,), 1.0, device=x.device),
            ],
            dim=1,
        )
        extra = torch.full((n, 1), 123.0, device=x.device)
        return torch.cat([base, extra], dim=1)


def test_compute_mpc_cost_handles_extra_outputs():
    device = torch.device("cpu")
    model = DummyModel()
    wn = wntr.network.WaterNetworkModel()

    pump_speeds = torch.zeros(1, 1, dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 0))
    num_nodes = 2
    node_types = torch.zeros(num_nodes, dtype=torch.long)
    edge_types = torch.zeros(0, dtype=torch.long)
    template = torch.zeros(num_nodes, 5)
    template[:, 4] = torch.tensor([-1.0, 1.0])
    pressures = torch.tensor([10.0, 10.0])
    chlorine = torch.tensor([0.0, 0.0])

    cost, _ = compute_mpc_cost(
        pump_speeds,
        wn,
        model,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        template,
        pressures,
        chlorine,
        horizon=1,
        device=device,
        Pmin=5.0,
        Cmin=0.0,
        skip_normalization=True,
    )
    assert torch.isfinite(cost)
