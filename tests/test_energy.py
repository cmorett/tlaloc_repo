import wntr
from wntr.metrics.economic import pump_energy


def test_pump_energy_not_nan():
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    energy_df = pump_energy(results.link['flowrate'][wn.pump_name_list], results.node['head'], wn)
    assert not energy_df[wn.pump_name_list].isna().any().any(), 'energy contains NaN'
