import pathlib
import sys

import torch

# Ensure repository root is on the Python path so ``scripts`` can be imported
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.train_gnn import _trim_norm_stats


def test_trim_norm_stats_pads_when_shorter():
    mean = torch.tensor([[1.0], [2.0]])
    std = torch.tensor([[0.5], [0.25]])
    mean2, std2 = _trim_norm_stats(mean, std, 4)
    assert mean2.shape == (4, 1)
    assert std2.shape == (4, 1)
    # existing values are preserved
    assert torch.allclose(mean2[:2], mean)
    assert torch.allclose(std2[:2], std)
    # new rows default to 0 mean and 1 std so denormalisation leaves them unchanged
    assert torch.allclose(mean2[2:], torch.zeros(2, 1))
    assert torch.allclose(std2[2:], torch.ones(2, 1))

