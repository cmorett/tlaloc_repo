import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import wntr
from scripts.data_generation import _build_randomized_network


def test_random_initial_tanks():
    inp = Path(__file__).resolve().parents[1] / "CTown.inp"
    wn_ref = wntr.network.WaterNetworkModel(str(inp))
    base_levels = {t: wn_ref.get_node(t).init_level for t in wn_ref.tank_name_list}
    wn_rand, _, _ = _build_randomized_network(str(inp), 0)
    changed = False
    for tname in wn_rand.tank_name_list:
        tank = wn_rand.get_node(tname)
        assert tank.min_level <= tank.init_level <= tank.max_level
        if abs(tank.init_level - base_levels[tname]) > 1e-6:
            changed = True
    assert changed, "tank levels should be randomized"
