import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import compute_sequence_norm_stats
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
