import os
import sys
from pathlib import Path
import wntr
from wntr.metrics.economic import pump_energy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "scripts"))
TEMP_DIR = REPO_ROOT / "data" / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
from scripts.wntr_compat import make_simulator


def test_pump_energy_not_nan():
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    sim = make_simulator(wn)
    results = sim.run_sim(str(TEMP_DIR / "temp"))
    energy_df = pump_energy(results.link['flowrate'][wn.pump_name_list], results.node['head'], wn)
    assert not energy_df[wn.pump_name_list].isna().any().any(), 'energy contains NaN'


def _simulate_speed(speed: float):
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    for pn in wn.pump_name_list:
        pump = wn.get_link(pn)
        pump.initial_status = wntr.network.base.LinkStatus.Open
        pump.base_speed = speed
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    sim = make_simulator(wn)
    results = sim.run_sim(str(TEMP_DIR / f"temp_{speed}"))
    flows = results.link['flowrate'][wn.pump_name_list].iloc[-1].abs().sum()
    energy_df = pump_energy(results.link['flowrate'][wn.pump_name_list], results.node['head'], wn)
    energy = energy_df[wn.pump_name_list].iloc[-1].sum()
    return flows, energy


def test_fractional_pump_speed_affects_flow_and_energy():
    flow_half, energy_half = _simulate_speed(0.5)
    flow_full, energy_full = _simulate_speed(1.0)
    assert flow_half < flow_full
    assert energy_half < energy_full
