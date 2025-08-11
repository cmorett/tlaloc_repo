import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import load_surrogate_model, EPS


def test_checkpoint_norm_io(tmp_path):
    torch.manual_seed(0)
    state = {
        'layers.0.lin.weight': torch.randn(2, 5),
        'layers.0.bias': torch.zeros(2),
    }
    ckpt = tmp_path / 'model.pth'
    torch.save(state, ckpt)
    np.savez(
        tmp_path / 'model_norm.npz',
        x_mean=np.zeros(5),
        x_std=np.ones(5),
        y_mean=np.zeros(2),
        y_std=np.ones(2),
    )
    model = load_surrogate_model(torch.device('cpu'), path=str(ckpt), use_jit=False)
    assert model.x_mean.shape == (5,)
    assert model.x_std.shape == (5,)
    assert model.y_mean.shape == (2,)
    assert model.y_std.shape == (2,)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x_norm = torch.randn(2, 5)
    with torch.no_grad():
        out_norm = model(x_norm, edge_index)
        x_raw = x_norm * (model.x_std + EPS) + model.x_mean
        out_manual = model((x_raw - model.x_mean) / (model.x_std + EPS), edge_index)
    assert torch.allclose(out_norm, out_manual, atol=1e-6)
