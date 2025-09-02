import sys
from pathlib import Path
import logging
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import log_array_stats  # noqa: E402


def test_log_array_stats_handles_strings(caplog):
    caplog.set_level(logging.INFO)
    arr = np.array(["normal", "fire_flow"], dtype=object)
    log_array_stats("scenarios", arr)
    assert "scenarios: shape" in caplog.text


def test_log_array_stats_detects_invalid():
    arr = np.array([{"α": np.array([1.0, np.nan])}], dtype=object)
    with pytest.raises(ValueError):
        log_array_stats("bad", arr)
