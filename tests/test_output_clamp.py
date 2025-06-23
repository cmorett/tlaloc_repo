import torch
from scripts.train_gnn import RecurrentGNNSurrogate, MultiTaskGNNSurrogate


def test_recurrent_output_clamp():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.ones(1, 3)
    model = RecurrentGNNSurrogate(
        in_channels=2,
        hidden_channels=4,
        edge_dim=3,
        output_dim=2,
        num_layers=1,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=4,
    )
    model.decoder.weight.data.zero_()
    model.decoder.bias.data.fill_(-1.0)
    X = torch.zeros(1, 1, 2, 2)
    out = model(X, edge_index, edge_attr)
    assert torch.all(out >= 0)


def test_multitask_output_clamp_and_tank_level():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = torch.ones(1, 3)
    model = MultiTaskGNNSurrogate(
        in_channels=2,
        hidden_channels=4,
        edge_dim=3,
        node_output_dim=2,
        edge_output_dim=1,
        num_layers=1,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=4,
    )
    model.node_decoder.weight.data.zero_()
    model.node_decoder.bias.data.fill_(0.0)
    model.edge_decoder.weight.data.zero_()
    model.edge_decoder.bias.data.fill_(-1.0)
    model.tank_indices = torch.tensor([0])
    model.tank_areas = torch.tensor([1.0])
    model.tank_edges = [torch.tensor([0])]
    model.tank_signs = [torch.tensor([1.0])]
    X = torch.zeros(1, 1, 2, 2)
    out = model(X, edge_index, edge_attr)
    assert torch.all(out['node_outputs'] >= 0)
    assert torch.all(model.tank_levels >= 0)
