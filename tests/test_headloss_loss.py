import torch
from models.loss_utils import pressure_headloss_consistency_loss

def test_headloss_consistency_zero():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.tensor([[1000.0, 0.5, 100.0]], dtype=torch.float32)
    flow = torch.tensor([0.1], dtype=torch.float32)
    const = 10.67
    q_m3 = flow  # flows already in m^3/s
    hl = const * edge_attr[0,0] * q_m3.abs().pow(1.852) / (
        edge_attr[0,2].pow(1.852) * edge_attr[0,1].pow(4.87)
    )
    pressures = torch.tensor([50.0 + hl.item(), 50.0], dtype=torch.float32)
    loss = pressure_headloss_consistency_loss(pressures, flow, edge_index, edge_attr)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

