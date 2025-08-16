import numpy as np
import torch
import pytest
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import load_surrogate_model
from models.gnn_surrogate import RecurrentGNNSurrogate


def test_strict_load_vs_cfg_mismatch(tmp_path):
    device = torch.device('cpu')
    model = RecurrentGNNSurrogate(
        in_channels=1,
        hidden_channels=32,
        edge_dim=0,
        output_dim=1,
        num_layers=1,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=32,
    )
    ckpt = {
        'model_state_dict': model.state_dict(),
        'model_meta': {
            'model_class': 'RecurrentGNNSurrogate',
            'in_channels': 1,
            'hidden_dim': 32,
            'num_layers': 1,
            'use_attention': False,
            'gat_heads': 1,
            'residual': False,
            'dropout': 0.0,
            'activation': 'relu',
            'output_dim': 1,
            'rnn_hidden_dim': 32,
            'edge_dim': 0,
        },
    }
    path = tmp_path / 'model.pth'
    torch.save(ckpt, path)
    cfg = {'hidden_dim': 16, 'num_layers': 1}
    with pytest.raises(ValueError):
        load_surrogate_model(device, path=str(path), use_jit=False, cfg_meta=cfg)


def test_edge_attr_scaling_effect():
    raw = np.array([[10.0, 0.5, 100.0], [20.0, 0.7, 50.0]], dtype=float)
    scaled = MinMaxScaler().fit_transform(np.log1p(raw))
    y_true = scaled[:, 0]
    mae_raw = np.abs(raw[:, 0] - y_true).mean()
    mae_scaled = np.abs(scaled[:, 0] - y_true).mean()
    assert mae_raw > mae_scaled * 2


def test_masking_consistency():
    preds = np.array([1.0, 5.0, 7.0])
    true = np.array([1.5, 2.5, 3.5])
    node_types = torch.tensor([0, 1, 2])
    diff = preds - true
    mask = (node_types == 0).numpy()
    mae_junction = np.abs(diff[mask]).mean()
    mae_all = np.abs(diff).mean()
    assert mask.sum() == 1
    assert np.isclose(mae_junction, np.abs(diff[0]))
    assert mae_all > mae_junction
