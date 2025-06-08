import torch
import pytest
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import train, GCNEncoder

def test_nan_detection():
    edge = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    d = Data(x=torch.tensor([[float('nan'),0.]], dtype=torch.float32), edge_index=edge,
             y=torch.tensor([[0.0]], dtype=torch.float32))
    loader = DataLoader([d], batch_size=1)
    model = GCNEncoder(2,4,1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    with pytest.raises(ValueError):
        train(model, loader, opt, torch.device('cpu'))
