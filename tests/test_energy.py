import os
from pathlib import Path
import wntr
from wntr.metrics.economic import pump_energy

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = REPO_ROOT / "data" / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def test_pump_energy_not_nan():
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(str(TEMP_DIR / "temp"))
    energy_df = pump_energy(results.link['flowrate'][wn.pump_name_list], results.node['head'], wn)
    assert not energy_df[wn.pump_name_list].isna().any().any(), 'energy contains NaN'
