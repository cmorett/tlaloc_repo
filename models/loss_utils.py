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
