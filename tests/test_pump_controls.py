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


def test_at_least_one_pump_active_per_hour():
    res, scale, controls = _run_scenario()
    hours = len(next(iter(controls.values())))
    for h in range(hours):
        assert any(controls[p][h] > 0 for p in controls)


def test_pump_min_dwell_time():
    res, scale, controls = _run_scenario()
    for speeds in controls.values():
        prev_state = speeds[0] > 0
        dwell = 1
        for spd in speeds[1:]:
            state = spd > 0
            if state != prev_state:
                assert dwell >= 2
                dwell = 1
                prev_state = state
            else:
                dwell += 1

