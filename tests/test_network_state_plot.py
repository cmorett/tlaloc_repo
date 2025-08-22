import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.mpc_control import plot_network_state_epyt


def test_plot_network_state_epyt_creates_png(tmp_path):
    pressures = {"J511": 30.0, "J411": 25.0}
    pump_controls = {"P1": 0.7}
    out_path = plot_network_state_epyt(pressures, pump_controls, "test", 0, plots_dir=tmp_path)
    assert out_path.exists()
    assert out_path.suffix == ".png"
