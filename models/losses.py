"""Loss helpers for weighted multi-task training and physics terms."""
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def _apply_loss(pred: torch.Tensor, target: torch.Tensor, loss_fn: str) -> torch.Tensor:
    """Apply the selected pointwise loss between ``pred`` and ``target``."""
    if loss_fn == "mse":
        return F.mse_loss(pred, target)
    if loss_fn == "huber":
        return F.smooth_l1_loss(pred, target, beta=1.0)
    return F.l1_loss(pred, target)


def weighted_mtl_loss(
    pred_nodes: torch.Tensor,
    target_nodes: torch.Tensor,
    edge_preds: torch.Tensor,
    edge_target: torch.Tensor,
    *,
    loss_fn: str = "mae",
    w_press: float = 5.0,
    w_cl: float = 0.0,
    w_flow: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total and component losses for pressure, chlorine and flow.

    Parameters
    ----------
    pred_nodes: ``[..., num_nodes, 2]``
        Predicted node outputs where channel ``0`` is pressure and ``1`` is
        chlorine.
    target_nodes: same shape as ``pred_nodes``
        Ground truth node outputs.
    edge_preds: ``[..., num_edges, 1]``
        Predicted edge flows.
    edge_target: same shape as ``edge_preds``
        Ground truth edge flows.
    loss_fn: {"mae", "mse", "huber"}
        Base loss applied per component.
    w_press, w_cl, w_flow: float
        Weights for pressure, chlorine and flow losses respectively. The
        defaults emphasise pressure and flow (``5.0`` and ``3.0``) while
        chlorine is disabled (``0.0``).
    """
    press_loss = _apply_loss(pred_nodes[..., 0], target_nodes[..., 0], loss_fn)
    if pred_nodes.size(-1) > 1:
        cl_loss = _apply_loss(pred_nodes[..., 1], target_nodes[..., 1], loss_fn)
    else:
        cl_loss = pred_nodes.new_tensor(0.0)
    flow_loss = _apply_loss(edge_preds, edge_target, loss_fn)
    total = w_press * press_loss + w_cl * cl_loss + w_flow * flow_loss
    return total, press_loss, cl_loss, flow_loss


def compute_mass_balance_loss(
    pred_flows: torch.Tensor,
    edge_index: torch.Tensor,
    node_count: int,
    demand: Optional[torch.Tensor] = None,
    node_type: Optional[torch.Tensor] = None,
    *,
    return_imbalance: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return mean squared node imbalance for predicted flows.

    When ``return_imbalance`` is ``True`` this function also returns the
    average absolute mass imbalance which can be logged as a metric.
    """
    if pred_flows.dim() == 1:
        flows = pred_flows.unsqueeze(1)
    else:
        flows = pred_flows.reshape(pred_flows.shape[0], -1)

    node_balance = pred_flows.new_zeros((node_count, flows.shape[1]))
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i]
        v = edge_index[1, i]
        f = flows[i]
        node_balance[u] -= f
        node_balance[v] += f

    # Each physical link appears twice (forward and reverse). Without
    # compensation this double-counts the contribution of every pipe which
    # inflates the loss for perfectly conserved flows. Halving the imbalance
    # restores a correct zero-loss for flows ``(+Q, -Q)`` on paired edges.
    node_balance = node_balance / 2.0

    if demand is not None:
        dem = demand.reshape(node_count, -1).to(node_balance.device)
        node_balance[:, : dem.shape[1]] -= dem

    if node_type is not None:
        node_type = node_type.reshape(node_count).to(node_balance.device)
        mask = (node_type == 1) | (node_type == 2)
        node_balance[mask] = 0

    loss = torch.mean(node_balance ** 2)
    if return_imbalance:
        return loss, node_balance.abs().mean()
    return loss


def pressure_headloss_consistency_loss(
    pred_pressures: torch.Tensor,
    pred_flows: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    elevation: Optional[torch.Tensor] = None,
    edge_attr_mean: Optional[torch.Tensor] = None,
    edge_attr_std: Optional[torch.Tensor] = None,
    edge_type: Optional[torch.Tensor] = None,
    node_type: Optional[torch.Tensor] = None,
    *,
    return_violation: bool = False,
    epsilon: float = 1e-6,
    sign_weight: float = 0.5,
    use_head: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return MSE between predicted and Hazen--Williams head losses.

    The loss compares predicted hydraulic head drops against the
    Hazen--Williams formula.  When ``use_head`` is ``True`` the function
    expects an ``elevation`` tensor and adds it to pressures before forming
    edge drops.  Edges incident to tanks or reservoirs (``node_type`` 1 or 2)
    are excluded.  When ``return_violation`` is ``True`` the percentage of
    edges where the predicted head loss has the wrong sign is also returned.
    """
    # Un-normalise edge attributes if statistics are available
    if edge_attr_mean is not None and edge_attr_std is not None:
        attr = edge_attr * edge_attr_std + edge_attr_mean
    else:
        attr = edge_attr

    length = attr[:, 0]
    diam = attr[:, 1].clamp(min=epsilon)
    rough = attr[:, 2].clamp(min=epsilon)

    # Flatten prediction tensors so the first dimension represents the batch
    p = pred_pressures.reshape(-1, pred_pressures.shape[-1])
    q = pred_flows.reshape(-1, pred_flows.shape[-1])

    if use_head:
        if elevation is None:
            raise ValueError("elevation tensor required when use_head=True")
        elev = elevation.reshape(-1, elevation.shape[-1]).to(p.device)
        head = p + elev
    else:
        head = p

    if edge_type is not None:
        edge_type = edge_type.flatten()
        pipe_mask = edge_type == 0
    else:
        pipe_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=p.device)

    if node_type is not None:
        nt = node_type.flatten().to(p.device)
        src_nt = nt[edge_index[0]]
        tgt_nt = nt[edge_index[1]]
        mask_nodes = (src_nt == 1) | (src_nt == 2) | (tgt_nt == 1) | (tgt_nt == 2)
        pipe_mask = pipe_mask & ~mask_nodes

    src = edge_index[0, pipe_mask]
    tgt = edge_index[1, pipe_mask]
    h_src = head[:, src]
    h_tgt = head[:, tgt]
    pred_hl = h_src - h_tgt

    const = 10.67
    length = length[pipe_mask]
    diam = diam[pipe_mask]
    rough = rough[pipe_mask]
    q_pipe = q[:, pipe_mask]
    q_m3 = q_pipe * 0.001
    flow_sign = torch.sign(q_pipe)
    denom = (rough.pow(1.852) * diam.pow(4.87)).clamp(min=epsilon)
    hw_hl = const * length * q_m3.abs().pow(1.852) / denom

    base = (
        torch.mean((pred_hl - flow_sign * hw_hl) ** 2)
        if pred_hl.numel() > 0
        else torch.tensor(0.0, device=pred_pressures.device)
    )

    # Encourage correct sign of head loss relative to flow direction.
    # For positive flow the head should drop (p_src > p_tgt) and the opposite
    # for negative flow. Wrong sign contributes a positive hinge penalty.
    if sign_weight > 0 and pred_hl.numel() > 0:
        valid = q_pipe.abs() > epsilon
        if valid.any():
            sign_term = torch.relu(-(pred_hl * flow_sign))[..., valid]
            sign_pen = sign_term.mean()
        else:
            sign_pen = torch.tensor(0.0, device=pred_pressures.device)
        loss = base + sign_weight * sign_pen
    else:
        loss = base

    if return_violation:
        with torch.no_grad():
            valid = q_pipe.abs() > epsilon
            viol = (pred_hl * flow_sign) < 0
            violation_pct = (
                (viol & valid).float().mean() if valid.any() else torch.tensor(0.0)
            )
        return loss, violation_pct

    return loss


def scale_physics_losses(
    mass_loss: torch.Tensor,
    head_loss: torch.Tensor,
    pump_loss: torch.Tensor,
    *,
    mass_scale: float = 1.0,
    head_scale: float = 1.0,
    pump_scale: float = 1.0,
    return_denominators: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float],
]:
    """Normalise physics-based losses by baseline magnitudes.

    Parameters
    ----------
    mass_loss, head_loss, pump_loss: torch.Tensor
        Raw physics loss values.
    mass_scale, head_scale, pump_scale: float, optional
        Baseline magnitudes for each loss. Values ``<= 0`` disable scaling.
    return_denominators: bool, optional
        When ``True`` the denominators applied to each loss are also
        returned so callers can reuse the safeguarded scaling factors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Scaled ``mass_loss``, ``head_loss`` and ``pump_loss`` when
        ``return_denominators`` is ``False``.
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]
        Scaled losses followed by the denominators ``(mass, head, pump)``
        when ``return_denominators`` is ``True``.
    """
    scale_floor = 1e-3

    def _scale(loss: torch.Tensor, scale: float) -> Tuple[torch.Tensor, float]:
        if scale > 0:
            denom = max(scale, scale_floor)
            return loss / denom, denom
        return loss, 1.0

    mass_loss, mass_denom = _scale(mass_loss, mass_scale)
    head_loss, head_denom = _scale(head_loss, head_scale)
    pump_loss, pump_denom = _scale(pump_loss, pump_scale)

    if return_denominators:
        return (
            mass_loss,
            head_loss,
            pump_loss,
            mass_denom,
            head_denom,
            pump_denom,
        )

    return mass_loss, head_loss, pump_loss

