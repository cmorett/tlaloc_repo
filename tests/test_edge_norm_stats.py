import torch
from torch_geometric.data import Data
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import compute_norm_stats, apply_normalization


@pytest.mark.parametrize("per_node", [False, True])
def test_edge_norm_stats(per_node):
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d1 = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=edge,
        y=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_y=torch.tensor([[1.0], [2.0]]),
    )
    d2 = Data(
        x=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        edge_index=edge,
        y=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        edge_y=torch.tensor([[3.0], [4.0]]),
    )
    data = [d1, d2]
    x_mean, x_std, y_mean, y_std = compute_norm_stats(data, per_node=per_node)
    apply_normalization(data, x_mean, x_std, y_mean, y_std, per_node=per_node)

    if per_node:
        all_edge = torch.stack([d.edge_y for d in data], dim=0)
        assert y_mean["edge_outputs"].shape == (edge.shape[1], 1)
    else:
        all_edge = torch.cat([d.edge_y for d in data], dim=0)
        assert y_mean["edge_outputs"].shape == (1,)

    assert torch.allclose(all_edge.mean(dim=0), torch.zeros_like(y_mean["edge_outputs"]), atol=1e-6)
    assert torch.allclose(all_edge.std(dim=0), torch.ones_like(y_std["edge_outputs"]), atol=1e-6)
