import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.data_generation import split_results


def test_split_results_deterministic():
    results = [(i, {}, {}) for i in range(20)]
    train1, val1, test1, counts1 = split_results(results, seed=123)
    train2, val2, test2, counts2 = split_results(results, seed=123)
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2
    assert counts1 == counts2


def test_split_results_different_seed():
    results = [(i, {}, {}) for i in range(20)]
    split1 = split_results(results, seed=1)
    split2 = split_results(results, seed=2)
    assert split1 != split2
