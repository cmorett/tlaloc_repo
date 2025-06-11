import torch
from models.loss_utils import compute_mass_balance_loss


def test_mass_balance_zero():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flows = torch.tensor([1.0, 1.0])
    loss = compute_mass_balance_loss(flows, edge_index, 2)
    assert torch.allclose(loss, torch.tensor(0.0))
