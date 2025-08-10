import torch
from typing import Optional

from .losses import compute_mass_balance_loss, pressure_headloss_consistency_loss


def pump_curve_loss(
    pred_flows: torch.Tensor,
    pump_coeffs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return penalty for flows violating pump head--flow curves."""
    if edge_type is None:
        return torch.tensor(0.0, device=pred_flows.device)

    flows = pred_flows.reshape(-1, pred_flows.shape[-1])
    mask = edge_type.flatten() == 1
    if not torch.any(mask):
        return torch.tensor(0.0, device=pred_flows.device)

    coeff = pump_coeffs[mask].to(pred_flows.device)
    q = flows[:, mask]
    a = coeff[:, 0]
    b = coeff[:, 1]
    c = coeff[:, 2]
    head = a - b * q.abs().pow(c)
    violation = torch.clamp(-head, min=0.0)
    return torch.mean(violation ** 2)
