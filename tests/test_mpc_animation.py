import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import wntr

from scripts.experiments_validation import animate_mpc_network


def test_animate_mpc_network(tmp_path: Path):
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    pump_names = wn.pump_name_list
    node_names = wn.node_name_list
    pressures = {n: 50.0 for n in node_names}
    df = pd.DataFrame([
        {
            'time': 0,
            'pressures': pressures,
            'controls': [0.5] * len(pump_names),
        }
    ])
    gif_path, html_path = animate_mpc_network(df, pump_names, 'unit', inp_path='CTown.inp', plots_dir=tmp_path)
    assert gif_path.exists()
    assert html_path.exists()
