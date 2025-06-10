import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import _run_single_scenario


def test_extreme_event_label():
    inp = Path(__file__).resolve().parents[1] / 'CTown.inp'
    res, _, _ = _run_single_scenario((0, str(inp), 123), extreme_event_prob=1.0)
    assert hasattr(res, 'scenario_type')
    assert res.scenario_type in {'fire_flow', 'pump_failure', 'quality_variation'}
