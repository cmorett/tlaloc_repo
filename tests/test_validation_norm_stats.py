from pathlib import Path
import subprocess
import torch
import numpy as np

repo = Path(__file__).resolve().parents[1]

def test_experiments_validation_requires_output_norm(tmp_path):
    state = {
        'layers.0.weight': torch.zeros(1, 1),
        'layers.0.bias': torch.zeros(1),
    }
    model_path = tmp_path / 'model.pth'
    torch.save(state, model_path)

    stats_path = tmp_path / 'stats.npz'
    np.savez(stats_path, x_mean=np.zeros(1), x_std=np.ones(1))

    cmd = [
        'python', str(repo / 'scripts/experiments_validation.py'),
        '--model', str(model_path),
        '--norm-stats', str(stats_path),
        '--inp', str(repo / 'CTown.inp'),
        '--test-pkl', str(tmp_path / 'missing.pkl'),
        '--no-jit',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'output normalization statistics' in result.stderr.lower()
