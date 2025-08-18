import torch
from torch_geometric.data import Data
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import compute_norm_stats, apply_normalization
import pytest


@pytest.mark.parametrize("per_node", [False, True])
def test_normalization(per_node):
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d1 = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=edge,
        y=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    d2 = Data(
        x=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        edge_index=edge,
        y=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
    )
    data = [d1, d2]
    x_mean, x_std, y_mean, y_std = compute_norm_stats(data, per_node=per_node)
    apply_normalization(data, x_mean, x_std, y_mean, y_std, per_node=per_node)
    if per_node:
        all_x = torch.stack([d.x for d in data], dim=0)
        all_y = torch.stack([d.y for d in data], dim=0)
        assert torch.allclose(all_x.mean(dim=0), torch.zeros_like(x_mean), atol=1e-6)
        assert torch.allclose(all_x.std(dim=0), torch.ones_like(x_std), atol=1e-6)
        assert torch.allclose(all_y.mean(dim=0), torch.zeros_like(y_mean), atol=1e-6)
        assert torch.allclose(all_y.std(dim=0), torch.ones_like(y_std), atol=1e-6)
    else:
        all_x = torch.cat([d.x for d in data], dim=0)
        all_y = torch.cat([d.y for d in data], dim=0)
        assert torch.allclose(all_x.mean(dim=0), torch.zeros_like(x_mean), atol=1e-6)
        assert torch.allclose(all_x.std(dim=0), torch.ones_like(x_std), atol=1e-6)
        assert torch.allclose(all_y.mean(dim=0), torch.zeros_like(y_mean), atol=1e-6)
        assert torch.allclose(all_y.std(dim=0), torch.ones_like(y_std), atol=1e-6)
