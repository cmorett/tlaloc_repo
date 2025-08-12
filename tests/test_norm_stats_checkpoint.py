import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import load_surrogate_model


def _make_norm_stats():
    x_mean = np.zeros(1, dtype=np.float32)
    x_std = np.ones(1, dtype=np.float32)
    y_mean = np.zeros(1, dtype=np.float32)
    y_std = np.ones(1, dtype=np.float32)
    edge_mean = np.zeros(1, dtype=np.float32)
    edge_std = np.ones(1, dtype=np.float32)
    md5 = hashlib.md5()
    for arr in [x_mean, x_std, y_mean, y_std, edge_mean, edge_std]:
        md5.update(arr.tobytes())
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "hash": md5.hexdigest(),
    }


def test_norm_stats_from_checkpoint(tmp_path):
    state = {
        'layers.0.weight': torch.zeros(1, 1),
        'layers.0.bias': torch.zeros(1),
    }
    norm_stats = _make_norm_stats()
    ckpt = {"model_state_dict": state, "norm_stats": norm_stats}
    path = tmp_path / 'model.pth'
    torch.save(ckpt, path)
    np.savez(
        tmp_path / 'model_norm.npz',
        x_mean=norm_stats['x_mean'],
        x_std=norm_stats['x_std'],
        y_mean=norm_stats['y_mean'],
        y_std=norm_stats['y_std'],
        edge_mean=norm_stats['edge_mean'],
        edge_std=norm_stats['edge_std'],
    )
    model = load_surrogate_model(torch.device('cpu'), path=str(path), use_jit=False)
    assert torch.allclose(model.x_mean.cpu(), torch.tensor([0.0]))
    assert model.norm_hash == norm_stats['hash']


def test_norm_stats_npz_mismatch(tmp_path):
    state = {
        'layers.0.weight': torch.zeros(1, 1),
        'layers.0.bias': torch.zeros(1),
    }
    norm_stats = _make_norm_stats()
    path = tmp_path / 'model.pth'
    torch.save({"model_state_dict": state, "norm_stats": norm_stats}, path)
    np.savez(
        tmp_path / 'model_norm.npz',
        x_mean=np.ones(1),
        x_std=np.ones(1),
        y_mean=np.zeros(1),
        y_std=np.ones(1),
        edge_mean=np.zeros(1),
        edge_std=np.ones(1),
    )
    with pytest.raises(ValueError):
        load_surrogate_model(torch.device('cpu'), path=str(path), use_jit=False)
