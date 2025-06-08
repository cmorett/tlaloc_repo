import torch
import pytest
from pathlib import Path
import sys

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
    model = load_surrogate_model(torch.device('cpu'), path=str(path))
    assert model.conv1.out_channels == 4
    assert model.conv2.out_channels == 2


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
        load_surrogate_model(torch.device('cpu'), path=str(path))
