import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import _run_single_scenario


def test_at_least_one_pump_active_per_hour():
    inp = Path(__file__).resolve().parents[1] / 'CTown.inp'
    res, scale, controls = _run_single_scenario((0, str(inp), 42))
    hours = len(next(iter(controls.values())))
    for h in range(hours):
        assert any(controls[p][h] > 0 for p in controls)

