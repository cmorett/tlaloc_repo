import sys
from pathlib import Path
import numpy as np
import wntr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import _run_single_scenario, build_dataset


def test_reservoir_pressure_equals_head():
    inp = Path(__file__).resolve().parents[1] / "CTown.inp"
    results = _run_single_scenario((0, str(inp), 42))
    assert results is not None
    sim_res, scale_dict, pump_ctrl = results
    wn = wntr.network.WaterNetworkModel(str(inp))
    X, Y = build_dataset([(sim_res, scale_dict, pump_ctrl)], wn)
    res_name = wn.reservoir_name_list[0]
    idx = wn.node_name_list.index(res_name)
    head = wn.get_node(res_name).base_head
    assert np.allclose(X[:, idx, 1], head)

