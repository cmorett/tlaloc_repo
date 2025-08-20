import sys
from pathlib import Path

import torch
import wntr


sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import MAX_PUMP_SPEED, run_mpc_step


def _setup():
    wn = wntr.network.WaterNetworkModel("CTown.inp")
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.zeros((1, 3))
    node_types = torch.zeros(2, dtype=torch.long)
    edge_types = torch.zeros(1, dtype=torch.long)
    feature_template = torch.zeros((2, 5))
    pressures = torch.zeros(2)
    chlorine = torch.zeros(2)

    class DummyModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_attr, node_types, edge_types):
            node_outputs = torch.zeros((x.size(0), 2))
            edge_outputs = torch.zeros((1, 1))
            return {"node_outputs": node_outputs, "edge_outputs": edge_outputs}

    return (
        wn,
        DummyModel(),
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
        pressures,
        chlorine,
    )


def test_run_mpc_step_respects_max_speed():
    (
        wn,
        model,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        template,
        pressures,
        chlorine,
    ) = _setup()
    device = torch.device("cpu")
    u_warm = torch.full((1, 1), MAX_PUMP_SPEED + 0.5)
    speeds, _, _ = run_mpc_step(
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
        iterations=1,
        device=device,
        Pmin=1.0,
        Cmin=0.1,
        u_warm=u_warm,
    )

    assert torch.all(speeds <= MAX_PUMP_SPEED + 1e-6)
    assert torch.all(speeds >= 0.0)

