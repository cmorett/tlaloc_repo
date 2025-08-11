import pandas as pd
from scripts.sweep_training import generate_configs
from scripts.plot_sweep import plot_pressure_mae_vs_config

def test_generate_configs():
    cfgs = generate_configs([1.0, 2.0], [0.0, 1.0], [0.0, 1.0], [3, 5], [64, 128], [0, 1])
    assert len(cfgs) >= 8

def test_plot_sweep(tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"pressure_mae": [0.1, 0.2]}).to_csv(csv_path, index=False)
    out = tmp_path / "plot.png"
    plot_pressure_mae_vs_config(csv_path, out)
    assert out.exists()
