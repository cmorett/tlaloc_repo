import torch
import pytest
import os
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import load_surrogate_model


def test_load_surrogate_renames_old_keys(tmp_path):
    state = {
        'conv1.bias': torch.zeros(4),
        'conv1.lin.weight': torch.zeros(4, 2),
        'conv2.bias': torch.zeros(2),
        'conv2.lin.weight': torch.zeros(2, 4)
    }
    path = tmp_path / 'model_old.pth'
    torch.save(state, path)
    model = load_surrogate_model(torch.device('cpu'), path=str(path), use_jit=False)
    assert model.layers[0].out_channels == 4
    assert model.layers[-1].out_channels == 2


def test_load_surrogate_detects_nan(tmp_path):
    state = {
        'layers.0.weight': torch.full((3, 2), float('nan')),
        'layers.0.bias': torch.zeros(3),
        'layers.1.weight': torch.zeros(1, 3),
        'layers.1.bias': torch.zeros(1)
    }
    path = tmp_path / 'model_nan.pth'
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_surrogate_model(torch.device('cpu'), path=str(path), use_jit=False)


def test_load_surrogate_selects_latest(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    older = {
        'conv1.lin.weight': torch.zeros(1, 1),
        'conv1.bias': torch.zeros(1),
        'conv2.lin.weight': torch.zeros(1, 1),
        'conv2.bias': torch.zeros(1)
    }
    newer = {
        'conv1.lin.weight': torch.zeros(2, 2),
        'conv1.bias': torch.zeros(2),
        'conv2.lin.weight': torch.zeros(1, 2),
        'conv2.bias': torch.zeros(1)
    }
    old_path = models_dir / 'old.pth'
    new_path = models_dir / 'new.pth'
    torch.save(older, old_path)
    torch.save(newer, new_path)
    os.utime(old_path, (os.path.getmtime(old_path) - 10, os.path.getmtime(old_path) - 10))

    # Ensure function searches within our temporary repository
    monkeypatch.setattr('scripts.mpc_control.REPO_ROOT', tmp_path)

    model = load_surrogate_model(torch.device('cpu'), use_jit=False)
    assert model.layers[0].out_channels == 2


def test_load_surrogate_handles_multitask_norm(tmp_path):
    state = {
        'conv1.lin.weight': torch.zeros(1, 1),
        'conv1.bias': torch.zeros(1),
        'conv2.lin.weight': torch.zeros(1, 1),
        'conv2.bias': torch.zeros(1),
    }
    path = tmp_path / 'model.pth'
    torch.save(state, path)
    np.savez(
        tmp_path / 'model_norm.npz',
        x_mean=np.zeros(1),
        x_std=np.ones(1),
        y_mean_node=np.zeros(1),
        y_std_node=np.ones(1),
        y_mean_edge=np.zeros(1),
        y_std_edge=np.ones(1),
    )
    model = load_surrogate_model(torch.device('cpu'), path=str(path), use_jit=False)
    assert model.y_mean is not None
    assert model.y_mean_energy is None
    assert model.y_std_energy is None
