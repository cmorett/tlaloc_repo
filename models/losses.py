"""Loss helpers for weighted multi-task training."""
from typing import Tuple
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
    w_press: float = 3.0,
    w_cl: float = 1.0,
    w_flow: float = 1.0,
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
        Weights for pressure, chlorine and flow losses respectively.
    """
    press_loss = _apply_loss(pred_nodes[..., 0], target_nodes[..., 0], loss_fn)
    cl_loss = _apply_loss(pred_nodes[..., 1], target_nodes[..., 1], loss_fn)
    flow_loss = _apply_loss(edge_preds, edge_target, loss_fn)
    total = w_press * press_loss + w_cl * cl_loss + w_flow * flow_loss
    return total, press_loss, cl_loss, flow_loss

