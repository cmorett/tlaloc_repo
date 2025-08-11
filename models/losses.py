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
    cl_loss = _apply_loss(pred_nodes[..., 1], target_nodes[..., 1], loss_fn)
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
        dem = demand.reshape(node_count, -1)
        node_balance[:, : dem.shape[1]] -= dem

    if node_type is not None:
        node_type = node_type.reshape(node_count)
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
    edge_attr_mean: Optional[torch.Tensor] = None,
    edge_attr_std: Optional[torch.Tensor] = None,
    edge_type: Optional[torch.Tensor] = None,
    *,
    return_violation: bool = False,
    epsilon: float = 1e-6,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return MSE between predicted and Hazen--Williams head losses.

    When ``return_violation`` is ``True`` the percentage of edges where the
    predicted head loss has the wrong sign is also returned.
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

    if edge_type is not None:
        edge_type = edge_type.flatten()
        pipe_mask = edge_type == 0
    else:
        pipe_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=p.device)

    src = edge_index[0, pipe_mask]
    tgt = edge_index[1, pipe_mask]
    p_src = p[:, src]
    p_tgt = p[:, tgt]
    pred_hl = p_src - p_tgt

    const = 10.67
    length = length[pipe_mask]
    diam = diam[pipe_mask]
    rough = rough[pipe_mask]
    q_pipe = q[:, pipe_mask]
    q_m3 = q_pipe * 0.001
    flow_sign = torch.sign(q_pipe)
    denom = (rough.pow(1.852) * diam.pow(4.87)).clamp(min=epsilon)
    hw_hl = const * length * q_m3.abs().pow(1.852) / denom

    loss = (
        torch.mean((pred_hl - flow_sign * hw_hl) ** 2)
        if pred_hl.numel() > 0
        else torch.tensor(0.0, device=pred_pressures.device)
    )

    if return_violation:
        with torch.no_grad():
            valid = q_pipe.abs() > epsilon
            viol = (pred_hl * flow_sign) < 0
            violation_pct = (
                (viol & valid).float().mean() if valid.any() else torch.tensor(0.0)
            )
        return loss, violation_pct

    return loss

