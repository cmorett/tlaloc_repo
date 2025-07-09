import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import compute_sequence_norm_stats


def test_sequence_norm_stats_edge_vector():
    X = np.zeros((1, 1, 2, 4), dtype=np.float32)
    Y = np.array([
        {
            "node_outputs": np.zeros((1, 2, 2), dtype=np.float32),
            "edge_outputs": np.array([[1.0, -1.0]], dtype=np.float32),
        }
    ], dtype=object)
    x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(X, Y)
    assert isinstance(y_mean, dict)
    assert y_mean["edge_outputs"].shape == torch.Size([2])
    assert y_std["edge_outputs"].shape == torch.Size([2])
    # check normalization broadcast
    Y_tensor = torch.tensor(Y[0]["edge_outputs"], dtype=torch.float32)
    Y_norm = (Y_tensor - y_mean["edge_outputs"]) / y_std["edge_outputs"]
    assert abs(float(Y_norm.mean())) < 1e-6
    assert abs(float(Y_norm.std(unbiased=False))) < 1e-6
