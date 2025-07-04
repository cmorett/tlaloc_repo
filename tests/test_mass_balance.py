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


def test_mass_balance_with_demand():
    """Demand creates imbalance when flows are symmetric."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flows = torch.tensor([1.0, -1.0])
    demand = torch.tensor([1.0, 0.0])
    loss = compute_mass_balance_loss(flows, edge_index, 2, demand=demand)
    assert torch.allclose(loss, torch.tensor(2.5))


def test_mass_balance_ignore_tank_nodes():
    """Tank nodes should not contribute to mass loss."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flows = torch.tensor([1.0, -1.0])
    node_type = torch.tensor([0, 1])
    loss = compute_mass_balance_loss(
        flows, edge_index, 2, node_type=node_type
    )
    assert torch.allclose(loss, torch.tensor(0.5))


def test_mass_balance_ignore_reservoir_nodes():
    """Reservoir nodes should also be excluded from mass loss."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    flows = torch.tensor([1.0, -1.0])
    node_type = torch.tensor([0, 2])
    loss = compute_mass_balance_loss(
        flows, edge_index, 2, node_type=node_type
    )
    assert torch.allclose(loss, torch.tensor(0.5))
