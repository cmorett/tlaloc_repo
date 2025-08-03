import torch
from models.loss_utils import pump_curve_loss


def test_pump_curve_loss_zero():
    # coefficients from CTown pump curve
    coeffs = torch.tensor([[70.0, 909.8618175228993, 1.3569154488567239]], dtype=torch.float32)
    flow = torch.tensor([0.06], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_type = torch.tensor([1], dtype=torch.long)
    loss = pump_curve_loss(flow, coeffs, edge_index, edge_type)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
