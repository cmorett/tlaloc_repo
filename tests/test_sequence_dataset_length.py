import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.feature_utils import SequenceDataset

def test_sequence_dataset_truncates_to_shortest_length():
    X = np.zeros((10, 2, 3), dtype=np.float32)
    Y = np.zeros((7, 2), dtype=np.float32)
    edge_index = np.zeros((2,0), dtype=np.int64)
    dataset = SequenceDataset(X, Y, edge_index, None)
    assert len(dataset) == 7
    # last index should be accessible without error
    x, y = dataset[6]
    assert x.shape[0] == 2 and x.shape[1] == 3
    assert y.shape[0] == 2
