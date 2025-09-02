import sys
from pathlib import Path
import random
import numpy as np
import wntr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import _build_randomized_network


def test_demand_multiplier_range():
    inp = Path(__file__).resolve().parents[1] / "CTown.inp"

    base_wn = wntr.network.WaterNetworkModel(str(inp))
    base_patterns = {}
    for jname in base_wn.junction_name_list:
        ts = base_wn.get_node(jname).demand_timeseries_list[0]
        if ts.pattern is None:
            base_mult = np.ones(168, dtype=float)
        else:
            base_mult = np.array(ts.pattern.multipliers, dtype=float)
        base_patterns[jname] = base_mult

    random.seed(42)
    np.random.seed(42)
    _, scale_dict, _ = _build_randomized_network(str(inp), idx=0)

    for jname, scaled in scale_dict.items():
        base_mult = base_patterns[jname][: len(scaled)]
        ratio = scaled / base_mult
        assert np.all(ratio >= 0.8)
        assert np.all(ratio <= 1.2)

    random.seed(123)
    np.random.seed(123)
    _, custom_scale_dict, _ = _build_randomized_network(
        str(inp), idx=0, demand_scale_min=0.5, demand_scale_max=1.5
    )

    for jname, scaled in custom_scale_dict.items():
        base_mult = base_patterns[jname][: len(scaled)]
        ratio = scaled / base_mult
        assert np.all(ratio >= 0.5)
        assert np.all(ratio <= 1.5)

