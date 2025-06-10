import torch
from scripts.train_gnn import HydroConv

def test_mass_conservation():
    edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    edge_attr = torch.ones(2,3)
    conv = HydroConv(1,1,edge_dim=3)
    with torch.no_grad():
        conv.lin.weight.fill_(1.0)
        conv.lin.bias.zero_()
        conv.edge_mlp[0].weight.fill_(1.0)
        conv.edge_mlp[0].bias.zero_()
    x = torch.tensor([[1.0],[2.0]])
    out = conv(x, edge_index, edge_attr)
    assert torch.allclose(out.sum(), torch.tensor(0.0), atol=1e-6)
