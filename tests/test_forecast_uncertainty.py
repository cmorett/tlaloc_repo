from pathlib import Path
import numpy as np
import pandas as pd

from scripts.forecast_uncertainty import plot_forecast_uncertainty


def test_plot_forecast_uncertainty(tmp_path: Path):
    ts = pd.date_range("2021-01-01", periods=48, freq="H")
    actual = np.tile(np.arange(24), 2)
    forecast = actual + np.tile([1, -1], 24)
    output = tmp_path / "fu.png"
    stats = plot_forecast_uncertainty(actual, forecast, ts, output)
    assert output.exists()
    assert "error_ci" in stats.columns
