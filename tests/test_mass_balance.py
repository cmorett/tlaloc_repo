import torch
from models.loss_utils import compute_mass_balance_loss


def test_mass_balance_reduced_for_opposite_flows():
    """Opposite flows on paired edges should not be over-penalised."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flows = torch.tensor([1.0, -1.0])
    loss = compute_mass_balance_loss(flows, edge_index, 2)
    # without halving this would be 4.0; the fix yields 1.0
    assert torch.allclose(loss, torch.tensor(1.0))


def test_mass_balance_zero_same_direction():
    """Equal flows in both directions cancel exactly."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flows = torch.tensor([1.0, 1.0])
    loss = compute_mass_balance_loss(flows, edge_index, 2)
    assert torch.allclose(loss, torch.tensor(0.0))
