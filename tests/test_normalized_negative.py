import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import train, GCNEncoder, compute_norm_stats, apply_normalization


def test_train_with_normalization_skips_negative_check():
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d = Data(x=torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32),
             edge_index=edge,
             y=torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    data = [d]
    x_mean, x_std, y_mean, y_std = compute_norm_stats(data)
    apply_normalization(data, x_mean, x_std, y_mean, y_std)
    loader = DataLoader(data, batch_size=1)
    model = GCNEncoder(2, 4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    # Should not raise error even though normalized pressures may be negative
    train(model, loader, opt, torch.device('cpu'), check_negative=False)
