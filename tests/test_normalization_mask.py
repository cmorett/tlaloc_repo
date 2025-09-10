import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import compute_norm_stats, compute_sequence_norm_stats

def test_node_mask_excludes_nodes():
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d = Data(
        x=torch.tensor([[0.0], [10.0]], dtype=torch.float32),
        edge_index=edge,
        y=torch.tensor([[0.0], [10.0]], dtype=torch.float32),
    )
    mask = torch.tensor([True, False])
    x_mean, x_std, y_mean, y_std = compute_norm_stats([d], node_mask=mask)
    assert torch.allclose(x_mean, torch.tensor([0.0]))
    assert torch.allclose(y_mean, torch.tensor([0.0]))
    x_mean_all, _, y_mean_all, _ = compute_norm_stats([d])
    assert torch.allclose(x_mean_all, torch.tensor([5.0]))
    assert torch.allclose(y_mean_all, torch.tensor([5.0]))


def test_sequence_node_mask_excludes_nodes():
    X = np.array([[[[0.0], [10.0]]]], dtype=np.float32)
    Y = np.array([[[[0.0], [10.0]]]], dtype=np.float32)
    mask = [True, False]
    x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(X, Y, node_mask=mask)
    assert torch.allclose(x_mean, torch.tensor([0.0]))
    assert torch.allclose(y_mean, torch.tensor([0.0]))
    x_mean_all, _, y_mean_all, _ = compute_sequence_norm_stats(X, Y)
    assert torch.allclose(x_mean_all, torch.tensor([5.0]))
    assert torch.allclose(y_mean_all, torch.tensor([5.0]))
