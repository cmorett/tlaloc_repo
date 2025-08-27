import sys
from pathlib import Path
import random
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import _run_single_scenario


def _run_scenario():
    inp = Path(__file__).resolve().parents[1] / 'CTown.inp'
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        return _run_single_scenario((0, str(inp), 42))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def _run_fixed_speed():
    inp = Path(__file__).resolve().parents[1] / 'CTown.inp'
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        return _run_single_scenario(
            (0, str(inp), 42),
            extreme_rate=0.0,
            pump_outage_rate=0.0,
            local_surge_rate=0.0,
            fixed_pump_speed=1.0,
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def test_at_least_one_pump_active_per_hour():
    res, scale, controls = _run_scenario()
    hours = len(next(iter(controls.values())))
    for h in range(hours):
        assert any(controls[p][h] > 0.05 for p in controls)


def test_pump_speeds_continuous_and_correlated():
    res, scale, controls = _run_scenario()
    for speeds in controls.values():
        arr = np.array(speeds, dtype=float)
        # Speeds span more than the discrete {0.0, 0.5, 1.0} set
        assert len(np.unique(np.round(arr, 2))) > 3
        # Adjacent hours should be positively correlated with limited jumps
        if len(arr) > 1:
            corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            assert corr > 0.3
            assert np.max(np.abs(np.diff(arr))) < 0.25


def test_fixed_pump_speed_constant():
    res, scale, controls = _run_fixed_speed()
    for speeds in controls.values():
        assert np.allclose(speeds, 1.0)

