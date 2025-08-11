import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import load_surrogate_model


def test_missing_decoder_bias_raises(tmp_path):
    state = {
        'layers.0.lin.weight': torch.zeros(1, 1),
        'layers.0.bias': torch.zeros(1),
        'rnn.weight_ih_l0': torch.zeros(4, 1),
        'rnn.weight_hh_l0': torch.zeros(4, 1),
        'rnn.bias_ih_l0': torch.zeros(4),
        'rnn.bias_hh_l0': torch.zeros(4),
        'decoder.weight': torch.zeros(1, 1),
        # 'decoder.bias' intentionally missing
    }
    path = tmp_path / 'model.pth'
    torch.save(state, path)
    with pytest.raises(KeyError):
        load_surrogate_model(torch.device('cpu'), path=str(path), use_jit=False)


def test_missing_tank_params_raises(tmp_path):
    state = {
        'layers.0.lin.weight': torch.zeros(1, 1),
        'layers.0.bias': torch.zeros(1),
        'decoder.weight': torch.zeros(1, 1),
        'decoder.bias': torch.zeros(1),
        'tank_indices': torch.tensor([0]),
        'tank_edges': torch.tensor([0]),
        'tank_signs': torch.tensor([1.0]),
        # 'tank_areas' intentionally missing
    }
    path = tmp_path / 'model.pth'
    torch.save(state, path)
    with pytest.raises(KeyError):
        load_surrogate_model(torch.device('cpu'), path=str(path), use_jit=False)
