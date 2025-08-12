import torch
import torch.nn.functional as F
from models.loss_utils import pump_curve_loss


def test_pump_curve_loss_zero():
    # coefficients from CTown pump curve
    coeffs = torch.tensor([[70.0, 909.8618175228993, 1.3569154488567239]], dtype=torch.float32)
    flow = torch.tensor([0.06], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_type = torch.tensor([1], dtype=torch.long)
    loss = pump_curve_loss(flow, coeffs, edge_index, edge_type)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_pump_curve_loss_clamps_flow():
    coeffs = torch.tensor([[70.0, 909.8618175228993, 1.3569154488567239]], dtype=torch.float32)
    flow = torch.tensor([1.0], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_type = torch.tensor([1], dtype=torch.long)
    loss = pump_curve_loss(flow, coeffs, edge_index, edge_type)
    a, b, c = coeffs[0]
    q_max = (a / b) ** (1.0 / c) * 1.2
    head = a - b * q_max.abs() ** c
    violation = torch.clamp(-head, min=0.0).view(1, 1)
    expected = F.smooth_l1_loss(violation, torch.zeros_like(violation))
    assert torch.allclose(loss, expected)
