import numpy as np
import torch
from torch_geometric.data import Data
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import (
    compute_sequence_norm_stats,
    SequenceDataset,
    apply_sequence_normalization,
    compute_norm_stats,
    apply_normalization,
)


def _sample_inputs():
    X = np.array(
        [
            [
                [[0, 0, 0, 1, 10], [0, 0, 0, 4, 40]],
                [[0, 0, 0, 2, 20], [0, 0, 0, 5, 50]],
            ]
        ],
        dtype=np.float32,
    )
    Y = np.zeros((1, 2, 2, 1), dtype=np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)
    pump_cols = [3, 4]
    return X, Y, edge_index, pump_cols


def test_pump_speed_global_stats_sequence():
    X, Y, edge_index, pump_cols = _sample_inputs()
    x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(
        X, Y, per_node=True, static_cols=pump_cols
    )
    for col in pump_cols:
        assert torch.allclose(
            x_mean[:, col], x_mean[0, col].repeat(x_mean.size(0))
        )
        assert torch.allclose(
            x_std[:, col], x_std[0, col].repeat(x_std.size(0))
        )
    ds = SequenceDataset(X, Y, edge_index, None)
    apply_sequence_normalization(
        ds, x_mean, x_std, y_mean, y_std, per_node=True, static_cols=pump_cols
    )
    col = ds.X[..., pump_cols]
    assert torch.allclose(col.mean(dim=(0, 1, 2)), torch.zeros(len(pump_cols)), atol=1e-6)
    assert torch.allclose(
        col.std(dim=(0, 1, 2), unbiased=False), torch.ones(len(pump_cols)), atol=1e-6
    )


def test_pump_speed_global_stats_data_list():
    X, Y, _, pump_cols = _sample_inputs()
    edge = torch.empty((2, 0), dtype=torch.long)
    data_list = [
        Data(x=torch.tensor(X[0, 0], dtype=torch.float32), edge_index=edge, y=torch.zeros((2, 1))),
        Data(x=torch.tensor(X[0, 1], dtype=torch.float32), edge_index=edge, y=torch.zeros((2, 1))),
    ]
    x_mean, x_std, y_mean, y_std = compute_norm_stats(
        data_list, per_node=True, static_cols=pump_cols
    )
    for col in pump_cols:
        assert torch.allclose(
            x_mean[:, col], x_mean[0, col].repeat(x_mean.size(0))
        )
        assert torch.allclose(
            x_std[:, col], x_std[0, col].repeat(x_std.size(0))
        )
    apply_normalization(data_list, x_mean, x_std, y_mean, y_std, per_node=True)
    cols = torch.stack([d.x for d in data_list], dim=0)[..., pump_cols]
    assert torch.allclose(cols.mean(dim=(0, 1)), torch.zeros(len(pump_cols)), atol=1e-6)
    assert torch.allclose(
        cols.std(dim=(0, 1), unbiased=True), torch.ones(len(pump_cols)), atol=1e-6
    )
