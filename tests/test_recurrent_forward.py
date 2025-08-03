import torch
from scripts.train_gnn import RecurrentGNNSurrogate, MultiTaskGNNSurrogate

def test_recurrent_gnn_forward_shape_with_pump_edges():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones(2, 3)
    edge_type = torch.tensor([1, 1], dtype=torch.long)  # pump edges
    model = RecurrentGNNSurrogate(
        in_channels=2,
        hidden_channels=4,
        edge_dim=3,
        output_dim=1,
        num_layers=2,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=5,
        num_edge_types=2,
    )
    X_seq = torch.ones(1, 3, 2, 2)
    out = model(X_seq, edge_index, edge_attr, edge_type=edge_type)
    assert out.shape == (1, 3, 2, 1)


def test_multitask_gnn_forward_shapes_with_pump_edges():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones(2, 3)
    edge_type = torch.tensor([1, 1], dtype=torch.long)
    model = MultiTaskGNNSurrogate(
        in_channels=2,
        hidden_channels=4,
        edge_dim=3,
        node_output_dim=2,
        edge_output_dim=1,
        num_layers=2,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=8,
        num_edge_types=2,
    )
    X_seq = torch.ones(1, 3, 2, 2)
    out = model(X_seq, edge_index, edge_attr, edge_type=edge_type)
    assert out['node_outputs'].shape == (1, 3, 2, 2)
    assert out['edge_outputs'].shape == (1, 3, 2, 1)
