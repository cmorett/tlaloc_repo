import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import (
    compute_norm_stats,
    apply_normalization,
    SequenceDataset,
    compute_sequence_norm_stats,
    apply_sequence_normalization,
)
import pytest


@pytest.mark.parametrize("per_node", [False, True])
def test_node_mask_normalization(per_node):
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d1 = Data(
        x=torch.tensor([[100.0], [1.0]]),
        edge_index=edge,
        y=torch.tensor([[100.0], [1.0]]),
    )
    d2 = Data(
        x=torch.tensor([[100.0], [3.0]]),
        edge_index=edge,
        y=torch.tensor([[100.0], [3.0]]),
    )
    data = [d1, d2]
    mask = torch.tensor([False, True])
    x_mean, x_std, y_mean, y_std = compute_norm_stats(
        data, per_node=per_node, node_mask=mask
    )
    apply_normalization(
        data, x_mean, x_std, y_mean, y_std, per_node=per_node, node_mask=mask
    )
    if per_node:
        all_x = torch.stack([d.x[mask] for d in data], dim=0)
        all_y = torch.stack([d.y[mask] for d in data], dim=0)
    else:
        all_x = torch.cat([d.x[mask] for d in data], dim=0)
        all_y = torch.cat([d.y[mask] for d in data], dim=0)
    assert torch.allclose(all_x.mean(dim=0), torch.zeros_like(x_mean), atol=1e-6)
    assert torch.allclose(all_x.std(dim=0), torch.ones_like(x_std), atol=1e-6)
    assert torch.allclose(all_y.mean(dim=0), torch.zeros_like(y_mean), atol=1e-6)
    assert torch.allclose(all_y.std(dim=0), torch.ones_like(y_std), atol=1e-6)
    # Unmasked node remains unmodified
    for d in data:
        assert torch.allclose(d.x[0], torch.tensor([100.0]))
        assert torch.allclose(d.y[0], torch.tensor([100.0]))


@pytest.mark.parametrize("per_node", [False, True])
def test_sequence_node_mask_normalization(per_node):
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_attr = np.zeros((edge_index.shape[1], 3), dtype=np.float32)
    X = np.array([
        [[[100.0], [1.0]]],
        [[[100.0], [3.0]]],
    ], dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.array([[[100.0], [1.0]]], dtype=np.float32),
                "edge_outputs": np.zeros((1, edge_index.shape[1]), dtype=np.float32),
            },
            {
                "node_outputs": np.array([[[100.0], [3.0]]], dtype=np.float32),
                "edge_outputs": np.zeros((1, edge_index.shape[1]), dtype=np.float32),
            },
        ],
        dtype=object,
    )
    mask = torch.tensor([False, True])
    x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(
        X, Y, per_node=per_node, node_mask=mask
    )
    ds = SequenceDataset(X, Y, edge_index, edge_attr)
    apply_sequence_normalization(
        ds,
        x_mean,
        x_std,
        y_mean,
        y_std,
        per_node=per_node,
        node_mask=mask,
    )
    masked_x = ds.X[:, :, mask, :].reshape(-1)
    masked_y = ds.Y["node_outputs"][:, :, mask, :].reshape(-1)
    assert abs(float(masked_x.mean())) < 1e-6
    assert abs(float(masked_x.std(unbiased=False)) - 1) < 1e-6
    assert abs(float(masked_y.mean())) < 1e-6
    assert abs(float(masked_y.std(unbiased=False)) - 1) < 1e-6
    assert torch.allclose(ds.X[:, :, 0, :], torch.tensor(100.0))
    assert torch.allclose(ds.Y["node_outputs"][..., 0, :], torch.tensor(100.0))
