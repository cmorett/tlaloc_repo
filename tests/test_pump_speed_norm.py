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
from models.gnn_surrogate import HydroConv


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


def test_direction_and_pump_speed_skip_normalization():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr_values = torch.tensor(
        [
            [[0.2, 1.5, 1.0, 1.2], [0.4, 0.5, 0.0, 0.0]],
            [[0.6, 1.2, 1.0, 0.8], [0.3, 0.7, 0.0, 0.4]],
        ],
        dtype=torch.float32,
    )
    data_list = [
        Data(
            x=torch.zeros((2, 1), dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr_values[i].clone(),
            y=torch.zeros((2, 1), dtype=torch.float32),
        )
        for i in range(edge_attr_values.size(0))
    ]
    flat = edge_attr_values.reshape(-1, edge_attr_values.size(-1)).numpy()
    edge_mean = torch.tensor(flat.mean(axis=0), dtype=torch.float32)
    edge_std = torch.tensor(flat.std(axis=0) + 1e-8, dtype=torch.float32)
    skip_cols = [edge_attr_values.size(-1) - 2, edge_attr_values.size(-1) - 1]
    x_mean = torch.zeros(1)
    x_std = torch.ones(1)
    y_mean = torch.zeros(1)
    y_std = torch.ones(1)
    apply_normalization(
        data_list,
        x_mean,
        x_std,
        y_mean,
        y_std,
        edge_mean,
        edge_std,
        skip_edge_attr_cols=skip_cols,
    )
    normalized_attrs = torch.stack([d.edge_attr for d in data_list], dim=0)
    direction_vals = normalized_attrs[..., skip_cols[0]]
    pump_vals = normalized_attrs[..., skip_cols[1]]
    expected_direction = edge_attr_values[..., skip_cols[0]]
    expected_pump = edge_attr_values[..., skip_cols[1]]
    assert set(direction_vals.flatten().tolist()) == {0.0, 1.0}
    assert torch.allclose(direction_vals, expected_direction)
    assert torch.all(pump_vals >= 0)
    assert torch.allclose(pump_vals, expected_pump)

    # Sequence dataset should honour the same skip columns
    X_seq = np.zeros((1, 1, 2, 1), dtype=np.float32)
    Y_seq = np.zeros((1, 1, 2, 1), dtype=np.float32)
    edge_attr_seq = edge_attr_values.unsqueeze(0).numpy()
    seq_dataset = SequenceDataset(
        X_seq,
        Y_seq,
        edge_index.numpy(),
        edge_attr=edge_attr_values[0].numpy(),
        edge_attr_seq=edge_attr_seq,
    )
    apply_sequence_normalization(
        seq_dataset,
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        edge_mean,
        edge_std,
        skip_edge_attr_cols=skip_cols,
    )
    assert torch.allclose(seq_dataset.edge_attr[:, skip_cols[0]], expected_direction[0])
    assert torch.all(seq_dataset.edge_attr[:, skip_cols[1]] >= 0)
    assert torch.allclose(seq_dataset.edge_attr[:, skip_cols[1]], expected_pump[0])
    seq_direction = seq_dataset.edge_attr_seq[..., skip_cols[0]]
    seq_pump = seq_dataset.edge_attr_seq[..., skip_cols[1]]
    assert set(seq_direction.flatten().tolist()) == {0.0, 1.0}
    assert torch.all(seq_pump >= 0)

    # HydroConv should still compute ±1 signs from the raw direction column
    conv = HydroConv(in_channels=1, out_channels=1, edge_dim=normalized_attrs.size(-1))
    x = torch.tensor([[1.0], [4.0]], dtype=torch.float32)
    x_i = x.index_select(0, edge_index[1])
    x_j = x.index_select(0, edge_index[0])
    edge_type = torch.zeros(normalized_attrs.size(1), dtype=torch.long)
    messages = conv.message(x_i, x_j, normalized_attrs[0], edge_type)
    diff = x_j - x_i
    ratio = messages.squeeze() / diff.squeeze()
    expected_sign = normalized_attrs[0][:, skip_cols[0]] * 2.0 - 1.0
    assert torch.allclose(torch.sign(ratio), expected_sign)
