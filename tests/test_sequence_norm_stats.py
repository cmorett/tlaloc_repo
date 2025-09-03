import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import (
    compute_sequence_norm_stats,
    SequenceDataset,
    apply_sequence_normalization,
)
import pytest


@pytest.mark.parametrize("per_node", [False, True])
def test_sequence_norm_stats_edge_vector(per_node):
    X = np.zeros((1, 1, 2, 4), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((1, 2, 2), dtype=np.float32),
                "edge_outputs": np.array([[1.0, -1.0]], dtype=np.float32),
            }
        ],
        dtype=object,
    )
    x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(X, Y, per_node=per_node)
    assert isinstance(y_mean, dict)
    assert y_mean["edge_outputs"].shape == torch.Size([2])
    assert y_std["edge_outputs"].shape == torch.Size([2])
    if per_node:
        assert x_mean.shape == torch.Size([2, 4])
        assert y_mean["node_outputs"].shape == torch.Size([2, 2])
    else:
        assert x_mean.shape == torch.Size([4])
        assert y_mean["node_outputs"].shape == torch.Size([2])
    # check normalization broadcast
    Y_tensor = torch.tensor(Y[0]["edge_outputs"], dtype=torch.float32)
    Y_norm = (Y_tensor - y_mean["edge_outputs"]) / y_std["edge_outputs"]
    assert abs(float(Y_norm.mean())) < 1e-6
    assert abs(float(Y_norm.std(unbiased=False))) < 1e-6


def test_static_cols_global_stats():
    X = np.array([[[[0, 0, 0, 10], [0, 0, 0, 20]]]], dtype=np.float32)
    Y = np.zeros((1, 1, 2, 1), dtype=np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)

    x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(
        X, Y, per_node=True, static_cols=[3]
    )
    assert torch.allclose(x_mean[:, 3], torch.tensor([15.0, 15.0]))
    assert torch.allclose(x_std[:, 3], torch.tensor([5.0, 5.0]))

    ds = SequenceDataset(X, Y, edge_index, None)
    apply_sequence_normalization(
        ds, x_mean, x_std, y_mean, y_std, per_node=True, static_cols=[3]
    )
    col = ds.X[..., 3]
    assert torch.allclose(col.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(col.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
