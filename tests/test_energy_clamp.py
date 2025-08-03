import torch
import pytest
import wntr

from scripts.mpc_control import compute_mpc_cost


def test_negative_flow_headloss_clamped():
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    speeds = torch.zeros((1, 1))
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.zeros((1, 3))
    node_types = torch.zeros(2, dtype=torch.long)
    edge_types = torch.zeros(1, dtype=torch.long)
    feature_template = torch.zeros((2, 5))
    feature_template[:, 3] = torch.tensor([10.0, 0.0])
    pressures = torch.zeros(2)
    chlorine = torch.zeros(2)

    class DummyModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_attr, node_types, edge_types):
            node_outputs = torch.zeros((x.size(0), 2))
            edge_outputs = torch.tensor([[-5.0]])
            return {'node_outputs': node_outputs, 'edge_outputs': edge_outputs}

    model = DummyModel()

    pump_info = [(0, 0, 1)]
    with pytest.warns(RuntimeWarning):
        cost, energy = compute_mpc_cost(
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
            Pmin=0.0,
            Cmin=0.0,
            pump_info=pump_info,
            return_energy=True,
        )
    assert energy.item() == 0.0
