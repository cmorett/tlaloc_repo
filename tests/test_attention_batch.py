import torch
from models.gnn_surrogate import EnhancedGNNEncoder

def test_attention_handles_batched_graphs():
    model = EnhancedGNNEncoder(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_layers=1,
        edge_dim=1,
        use_attention=True,
        gat_heads=1,
    )
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    edge_attr = torch.ones(4, 1)
    batch = torch.tensor([0, 0, 1, 1])
    out = model(x, edge_index, edge_attr, batch=batch)
    assert out.shape == (4, 2)
