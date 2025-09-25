import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.loss_utils import pressure_headloss_consistency_loss

def test_headloss_consistency_zero():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1000.0, 0.5, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    flow = torch.tensor([0.1], dtype=torch.float32)
    const = 10.67
    q_m3 = flow * 0.001
    hl = const * edge_attr[0,0] * q_m3.abs().pow(1.852) / (
        edge_attr[0,2].pow(1.852) * edge_attr[0,1].pow(4.87)
    )
    pressures = torch.tensor([50.0 + hl.item(), 50.0], dtype=torch.float32)
    elevation = torch.zeros(2, dtype=torch.float32)
    node_type = torch.zeros(2, dtype=torch.long)
    loss = pressure_headloss_consistency_loss(
        pressures,
        flow,
        edge_index,
        edge_attr,
        elevation=elevation,
        node_type=node_type,
    )
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_headloss_uses_head():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1000.0, 0.5, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    flow = torch.tensor([0.1], dtype=torch.float32)
    pressures = torch.tensor([50.0, 49.0], dtype=torch.float32)
    elev_equal = torch.tensor([10.0, 10.0], dtype=torch.float32)
    elev_diff = torch.tensor([10.0, 20.0], dtype=torch.float32)
    node_type = torch.zeros(2, dtype=torch.long)
    loss_pressure = pressure_headloss_consistency_loss(
        pressures, flow, edge_index, edge_attr, use_head=False
    )
    loss_equal = pressure_headloss_consistency_loss(
        pressures,
        flow,
        edge_index,
        edge_attr,
        elevation=elev_equal,
        node_type=node_type,
    )
    loss_diff = pressure_headloss_consistency_loss(
        pressures,
        flow,
        edge_index,
        edge_attr,
        elevation=elev_diff,
        node_type=node_type,
    )
    assert torch.allclose(loss_equal, loss_pressure, atol=1e-6)
    assert not torch.allclose(loss_diff, loss_pressure, atol=1e-6)




def test_headloss_log_roughness():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1000.0, 0.5, 130.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    flow = torch.tensor([0.2], dtype=torch.float32)
    pressures = torch.tensor([75.0, 70.0], dtype=torch.float32)
    elev = torch.tensor([5.0, 5.0], dtype=torch.float32)
    node_type = torch.zeros(2, dtype=torch.long)
    loss_raw = pressure_headloss_consistency_loss(
        pressures, flow, edge_index, edge_attr, elevation=elev, node_type=node_type
    )
    edge_attr_log = edge_attr.clone()
    edge_attr_log[:, 2] = torch.log1p(edge_attr_log[:, 2])
    loss_log = pressure_headloss_consistency_loss(
        pressures,
        flow,
        edge_index,
        edge_attr_log,
        elevation=elev,
        node_type=node_type,
        log_roughness=True,
    )
    assert torch.allclose(loss_raw, loss_log, atol=1e-6)
