import sys
from pathlib import Path

import torch
import wntr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.mpc_control import compute_mpc_cost


class DummyModel(torch.nn.Module):
    def __init__(self, num_nodes: int):
        super().__init__()
        # Five node features (pressure, chlorine, etc.)
        self.x_mean = torch.zeros(5)
        self.x_std = torch.ones(5)
        # Per-node normalisation stats for two outputs
        self.y_mean = torch.zeros(num_nodes, 2)
        self.y_std = torch.ones(num_nodes, 2)

    def forward(self, x, edge_index, edge_attr=None, node_types=None, edge_types=None):
        # Return zeros in normalised space
        return torch.zeros(x.size(0), 2, device=x.device)


def test_compute_mpc_cost_handles_per_node_norm():
    device = torch.device("cpu")
    num_nodes = 3
    num_pumps = 1
    model = DummyModel(num_nodes).eval()
    wn = wntr.network.WaterNetworkModel()

    pump_speeds = torch.zeros(1, num_pumps, dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 0))
    node_types = torch.zeros(num_nodes, dtype=torch.long)
    edge_types = torch.zeros(0, dtype=torch.long)
    template = torch.zeros(num_nodes, 4 + num_pumps)
    pressures = torch.full((num_nodes,), 50.0)
    chlorine = torch.zeros(num_nodes)

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
        Pmin=20.0,
        Cmin=0.2,
    )

    assert torch.isfinite(cost)
