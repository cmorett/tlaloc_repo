from pathlib import Path
import subprocess
import torch
import numpy as np
import hashlib

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
    assert 'normalization' in result.stderr.lower()


def test_experiments_validation_hash_mismatch(tmp_path):
    state = {
        'layers.0.weight': torch.zeros(1, 1),
        'layers.0.bias': torch.zeros(1),
    }
    model_path = tmp_path / 'model.pth'
    # Expected hash from zero stats
    x_mean = np.zeros(1, dtype=np.float32)
    x_std = np.ones(1, dtype=np.float32)
    y_mean = np.zeros(1, dtype=np.float32)
    y_std = np.ones(1, dtype=np.float32)
    md5 = hashlib.md5()
    for arr in [x_mean, x_std, y_mean, y_std]:
        md5.update(arr.tobytes())
    expected_hash = md5.hexdigest()
    torch.save({'model_state_dict': state, 'model_meta': {'norm_stats_md5': expected_hash}}, model_path)

    stats_path = tmp_path / 'stats.npz'
    # Different stats to trigger mismatch
    np.savez(stats_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean + 1, y_std=y_std)

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
    assert 'normalization' in result.stderr.lower()
