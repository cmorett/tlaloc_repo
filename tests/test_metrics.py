import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "scripts"))

from metrics import accuracy_metrics, control_metrics, computational_metrics


def test_accuracy_metrics_basic():
    true_p = [10.0, 20.0]
    pred_p = [9.0, 21.0]
    true_c = [0.5, 0.4]
    pred_c = [0.45, 0.6]
    df = accuracy_metrics(true_p, pred_p, true_c, pred_c)
    assert np.isclose(df.loc["Mean Absolute Error (MAE)", "Pressure (m)"], 1.0)
    assert np.isclose(df.loc["Mean Absolute Error (MAE)", "Chlorine (mg/L)"], 0.125)
    assert np.isclose(df.loc["Maximum Error", "Pressure (m)"], 1.0)
    assert np.isclose(df.loc["Maximum Error", "Chlorine (mg/L)"], 0.2)


def test_control_metrics_basic():
    min_p = [30.0, 10.0, 25.0]
    min_c = [0.3, 0.1, 0.4]
    energy = [1.0, 2.0, 3.0]
    df = control_metrics(min_p, min_c, energy, p_min=20.0, c_min=0.2)
    assert df.loc["Pressure Constraint Violations (hrs)", "Value"] == 1
    assert df.loc["Chlorine Constraint Violations (hrs)", "Value"] == 1
    assert np.isclose(df.loc["Total Pump Energy (kWh)", "Value"], 6.0)


def test_computational_metrics_basic():
    inf = [0.01, 0.02]
    opt = [0.1, 0.12]
    df = computational_metrics(inf, opt)
    assert np.isclose(df.loc["Inference Time per Simulation Step", "Average Time (ms)"], 15.0)
    assert np.isclose(df.loc["Optimization Time per Control Step", "Average Time (ms)"], 110.0)
