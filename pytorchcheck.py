import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
# Quick device check:
x = torch.randn(4, 16).cuda()            # Move a toy tensor to GPU
edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long).cuda()
data = Data(x=x, edge_index=edge_index)
conv = GCNConv(16, 32).cuda()
out = conv(data.x, data.edge_index)
print("PyG forward OK, output shape:", out.shape)