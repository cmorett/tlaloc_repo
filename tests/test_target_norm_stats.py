import torch
from torch_geometric.data import Data
import sys
from pathlib import Path
import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import compute_norm_stats, summarize_target_norm_stats


def test_summarize_target_norm_stats_per_node():
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d1 = Data(
        x=torch.randn(2, 3),
        edge_index=edge,
        y=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    d2 = Data(
        x=torch.randn(2, 3),
        edge_index=edge,
        y=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )
    _, _, y_mean, y_std = compute_norm_stats([d1, d2], per_node=True)
    pressure = summarize_target_norm_stats(y_mean, y_std)
    expected_std = 2.8284271
    assert pressure == pytest.approx((4.0, expected_std))

