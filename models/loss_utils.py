import torch
import torch.nn.functional as F
from typing import Optional

from .losses import compute_mass_balance_loss, pressure_headloss_consistency_loss


def pump_curve_loss(
    pred_flows: torch.Tensor,
    pump_coeffs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: Optional[torch.Tensor] = None,
    pump_speeds: Optional[torch.Tensor] = None,
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
    pump_count = int(mask.sum().item())

    speeds = None
    if pump_speeds is not None:
        speeds = pump_speeds.to(pred_flows.device)
        if speeds.ndim == 0:
            speeds = speeds.view(1, 1)
        elif speeds.ndim == 1:
            speeds = speeds.view(1, -1)
        else:
            speeds = speeds.reshape(-1, speeds.shape[-1])
        if speeds.shape[-1] == flows.shape[-1]:
            speeds = speeds[:, mask]
        elif speeds.shape[-1] != pump_count:
            raise ValueError(
                "pump_speeds must align with the pump edge mask"
            )

    a = coeff[:, 0]
    b = coeff[:, 1]
    c = coeff[:, 2]
    # Clamp flows to a realistic range based on pump curve limits to avoid
    # excessively large gradients when the network predicts out-of-distribution
    # values.
    q_max = (a / b).pow(1.0 / c) * 1.2  # 20% above zero-head flow
    q = torch.clamp(q, -q_max, q_max)
    if speeds is not None:
        eps = torch.finfo(q.dtype).eps
        s = speeds
        s_safe = torch.clamp_min(s, eps)
        head = s.pow(2) * (a - b * (q.abs() / s_safe).pow(c))
    else:
        head = a - b * q.abs().pow(c)
    violation = torch.clamp(-head, min=0.0)
    return F.smooth_l1_loss(violation, torch.zeros_like(violation))
