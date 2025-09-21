import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gnn_surrogate import HydroConv

def test_mass_conservation():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    conv = HydroConv(1, 1, edge_dim=10, num_node_types=1, num_edge_types=1)
    with torch.no_grad():
        conv.lin[0].weight.fill_(1.0)
        conv.lin[0].bias.zero_()
        conv.edge_mlps[0].weight.fill_(1.0)
        conv.edge_mlps[0].bias.zero_()
    x = torch.tensor([[1.0],[2.0]])
    out = conv(x, edge_index, edge_attr)
    assert torch.allclose(out.sum(), torch.tensor(0.0), atol=1e-6)


def test_pump_speed_scales_messages():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor([1, 1], dtype=torch.long)
    base_attr = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    edge_attr_low = base_attr.clone()
    edge_attr_high = base_attr.clone()
    edge_attr_high[:, -1] = 1.5
    conv = HydroConv(1, 1, edge_dim=10, num_node_types=1, num_edge_types=2)
    with torch.no_grad():
        conv.lin[0].weight.fill_(1.0)
        conv.lin[0].bias.zero_()
        conv.edge_mlps[1].weight.zero_()
        conv.edge_mlps[1].bias.zero_()
    x = torch.tensor([[1.0], [3.0]])
    out_low = conv(x, edge_index, edge_attr_low, edge_type=edge_type)
    out_high = conv(x, edge_index, edge_attr_high, edge_type=edge_type)
    assert not torch.allclose(out_low, out_high)
    assert torch.allclose(out_high.abs(), out_low.abs() * 1.5 / 0.5, atol=1e-5)


def test_pump_bias_adds_head_offset():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor([1, 1], dtype=torch.long)
    edge_attr = torch.zeros((2, 10), dtype=torch.float32)
    edge_attr[0, -2] = 1.0
    edge_attr[0, -1] = 0.5
    edge_attr[1, -2] = 0.0
    edge_attr[1, -1] = 1.5
    conv = HydroConv(1, 1, edge_dim=10, num_node_types=1, num_edge_types=2)
    with torch.no_grad():
        conv.lin[0].weight.fill_(1.0)
        conv.lin[0].bias.zero_()
        conv.edge_mlps[1].weight.zero_()
        conv.edge_mlps[1].bias.zero_()
    x = torch.tensor([[2.0], [5.0]])
    baseline = conv(x, edge_index, edge_attr, edge_type=edge_type)
    bias_raw = torch.tensor(0.7, dtype=edge_attr.dtype)
    with torch.no_grad():
        conv.edge_mlps[1].bias[1] = bias_raw.item()
    biased = conv(x, edge_index, edge_attr, edge_type=edge_type)
    delta = biased - baseline
    direction = edge_attr[:, -2:-1]
    sign = direction * 2.0 - 1.0
    pump_speed = edge_attr[:, -1:]
    expected_per_edge = sign * pump_speed * bias_raw
    expected = torch.zeros_like(delta)
    for eid in range(edge_index.size(1)):
        target = edge_index[1, eid]
        expected[target] += expected_per_edge[eid]
    assert torch.allclose(delta, expected, atol=1e-6)
