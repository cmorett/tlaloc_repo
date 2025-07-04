import torch
from typing import Optional

# Inspired by Ashraf et al. (AAAI 2024): Physics-Informed Graph Neural Networks for Water Distribution Systems
# See: https://arxiv.org/pdf/2403.18570

def compute_mass_balance_loss(
    pred_flows: torch.Tensor,
    edge_index: torch.Tensor,
    node_count: int,
    demand: Optional[torch.Tensor] = None,
    node_type: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return mean squared node imbalance for predicted flows.

    Parameters
    ----------
    pred_flows : torch.Tensor
        Flow rate predictions per edge. The first dimension must match the
        number of edges ``E``. Additional trailing dimensions are allowed and
        will be averaged over.
    edge_index : torch.Tensor
        Edge index tensor of shape ``[2, E]`` defining flow directions.
    node_count : int
        Total number of nodes in the graph.
    node_type : torch.Tensor, optional
        Integer node type array identifying tanks (value ``1``) and reservoirs
        (value ``2``). These nodes are ignored in the imbalance calculation
        when provided.
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

    return torch.mean(node_balance ** 2)


def pressure_headloss_consistency_loss(
    pred_pressures: torch.Tensor,
    pred_flows: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_attr_mean: Optional[torch.Tensor] = None,
    edge_attr_std: Optional[torch.Tensor] = None,
    edge_type: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return MSE between predicted and Hazen-Williams head losses.

    Parameters
    ----------
    pred_pressures : torch.Tensor
        Predicted pressures per node with shape ``[..., N]`` where ``N`` is the
        number of nodes. The tensor is flattened across all leading dimensions.
    pred_flows : torch.Tensor
        Predicted flow rates per edge with shape ``[..., E]``.
    edge_index : torch.Tensor
        Edge index tensor of shape ``[2, E]`` defining flow directions.
    edge_attr : torch.Tensor
        Edge attribute matrix ``[E, 3]`` providing pipe length ``[m]``, diameter
        ``[m]`` and Hazen--Williams roughness coefficient ``C``. Values may be
        normalized; ``edge_attr_mean`` and ``edge_attr_std`` will be used to
        restore physical units when provided.
    edge_attr_mean : torch.Tensor, optional
        Mean used during normalization of ``edge_attr``.
    edge_attr_std : torch.Tensor, optional
        Standard deviation used during normalization of ``edge_attr``.
    """

    # Un-normalise edge attributes if statistics are available
    if edge_attr_mean is not None and edge_attr_std is not None:
        attr = edge_attr * edge_attr_std + edge_attr_mean
    else:
        attr = edge_attr

    length = attr[:, 0]
    diam = attr[:, 1].clamp(min=1e-6)
    rough = attr[:, 2].clamp(min=1e-6)

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

    # Hazen--Williams head loss formula (SI units). Flows are stored in L/s
    # so convert to m^3/s before applying the equation.
    const = 10.67
    length = length[pipe_mask]
    diam = diam[pipe_mask]
    rough = rough[pipe_mask]
    # convert flow from L/s to m^3/s before applying Hazen--Williams
    q_m3 = q[:, pipe_mask] * 0.001
    flow_sign = torch.sign(q[:, pipe_mask])
    hw_hl = const * length * q_m3.abs().pow(1.852) / (
        rough.pow(1.852) * diam.pow(4.87)
    )

    return torch.mean((pred_hl - flow_sign * hw_hl) ** 2) if pred_hl.numel() > 0 else torch.tensor(0.0)
