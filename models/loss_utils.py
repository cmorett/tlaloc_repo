import torch

# Inspired by Ashraf et al. (AAAI 2024): Physics-Informed Graph Neural Networks for Water Distribution Systems
# See: https://arxiv.org/pdf/2403.18570

def compute_mass_balance_loss(pred_flows: torch.Tensor, edge_index: torch.Tensor, node_count: int) -> torch.Tensor:
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

    return torch.mean(node_balance ** 2)


def pressure_headloss_consistency_loss(
    pred_pressures: torch.Tensor,
    pred_flows: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_attr_mean: torch.Tensor | None = None,
    edge_attr_std: torch.Tensor | None = None,
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
    diam = attr[:, 1]
    rough = attr[:, 2].clamp(min=1e-6)

    # Flatten prediction tensors so the first dimension represents the batch
    p = pred_pressures.reshape(-1, pred_pressures.shape[-1])
    q = pred_flows.reshape(-1, pred_flows.shape[-1])

    src = edge_index[0]
    tgt = edge_index[1]
    p_src = p[:, src]
    p_tgt = p[:, tgt]
    pred_hl = (p_src - p_tgt).abs()

    # Hazen--Williams head loss formula (SI units)
    const = 10.67
    hw_hl = const * length * q.abs().pow(1.852) / (
        rough.pow(1.852) * diam.pow(4.87)
    )

    return torch.mean((pred_hl - hw_hl) ** 2)
