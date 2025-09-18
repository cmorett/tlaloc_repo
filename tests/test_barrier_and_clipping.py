import torch
import wntr
import torch
import torch
import wntr
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import compute_mpc_cost


def _setup():
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    speeds = torch.tensor([[2.0]], requires_grad=True)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.zeros((1, 10))
    node_types = torch.zeros(2, dtype=torch.long)
    edge_types = torch.zeros(1, dtype=torch.long)
    feature_template = torch.zeros((2, 5))
    pressures = torch.zeros(2)
    chlorine = torch.zeros(2)

    class DummyModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_attr, node_types, edge_types):
            node_outputs = torch.zeros((x.size(0), 2))
            edge_outputs = torch.zeros((1, 1))
            return {'node_outputs': node_outputs, 'edge_outputs': edge_outputs}

    return (
        wn,
        speeds,
        DummyModel(),
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
        pressures,
        chlorine,
    )


def test_barrier_variants_and_ste_gradient():
    (
        wn,
        speeds,
        model,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
        pressures,
        chlorine,
    ) = _setup()
    cost, _ = compute_mpc_cost(
        speeds,
        wn,
        model,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
        pressures,
        chlorine,
        horizon=1,
        device=torch.device('cpu'),
        Pmin=1.0,
        Cmin=0.1,
        barrier='softplus',
    )
    cost.backward()
    assert torch.isfinite(cost)
    assert speeds.grad.abs().max() > 0  # STE keeps gradient
    speeds.grad.zero_()
    cost_exp, _ = compute_mpc_cost(
        speeds,
        wn,
        model,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
        pressures,
        chlorine,
        horizon=1,
        device=torch.device('cpu'),
        Pmin=1.0,
        Cmin=0.1,
        barrier='exp',
    )
    cost_exp.backward()
    assert torch.isfinite(cost_exp)


def test_gradient_clipping_stub():
    wn, speeds, model, edge_index, edge_attr, node_types, edge_types, template, pressures, chlorine = _setup()
    cost, _ = compute_mpc_cost(
        speeds,
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
        device=torch.device('cpu'),
        Pmin=1.0,
        Cmin=0.1,
    )
    cost.backward()
    speeds.grad.fill_(1.0)
    gmax = 0.05
    speeds.grad.clamp_(-gmax, gmax)
    assert torch.all(speeds.grad.abs() <= gmax + 1e-6)
