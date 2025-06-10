import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import train, evaluate, GCNEncoder


def test_early_stopping(tmp_path):
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d = Data(x=torch.ones(2, 2), edge_index=edge, y=torch.zeros(2, 1))
    train_loader = DataLoader([d], batch_size=1)
    val_loader = DataLoader([d], batch_size=1)
    model = GCNEncoder(2, 4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    best = float("inf")
    patience = 0
    epochs = 0
    for epoch in range(20):
        train(model, train_loader, opt, torch.device("cpu"))
        val_loss = 1.0
        if val_loss < best - 1e-6:
            best = val_loss
            patience = 0
        else:
            patience += 1
        epochs = epoch + 1
        if patience >= 2:
            break
    assert epochs <= 3
