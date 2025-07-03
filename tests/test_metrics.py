import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Add the repository root so the `scripts` package can be imported when
# running the tests directly or through CI.
sys.path.append(str(REPO_ROOT))

from scripts.metrics import (
    accuracy_metrics,
    control_metrics,
    computational_metrics,
)

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
    assert np.isclose(df.loc["Total Pump Energy (J)", "Value"], 6.0)


def test_computational_metrics_basic():
    inf = [0.01, 0.02]
    opt = [0.1, 0.12]
    df = computational_metrics(inf, opt)
    assert np.isclose(df.loc["Inference Time per Simulation Step", "Average Time (ms)"], 15.0)
    assert np.isclose(df.loc["Optimization Time per Control Step", "Average Time (ms)"], 110.0)

def test_accuracy_metrics_consistent_after_normalization():
    true_p = np.array([15.0, 25.0, 30.0])
    pred_p = np.array([14.0, 26.0, 31.0])
    true_c = np.array([0.6, 0.4, 0.5])
    pred_c = np.array([0.55, 0.38, 0.52])

    base_df = accuracy_metrics(true_p, pred_p, true_c, pred_c)

    p_mean, p_std = true_p.mean(), true_p.std() + 1e-8
    c_mean, c_std = true_c.mean(), true_c.std() + 1e-8

    tp_norm = (true_p - p_mean) / p_std
    pp_norm = (pred_p - p_mean) / p_std
    tc_norm = (true_c - c_mean) / c_std
    pc_norm = (pred_c - c_mean) / c_std

    tp_denorm = tp_norm * p_std + p_mean
    pp_denorm = pp_norm * p_std + p_mean
    tc_denorm = tc_norm * c_std + c_mean
    pc_denorm = pc_norm * c_std + c_mean

    norm_df = accuracy_metrics(tp_denorm, pp_denorm, tc_denorm, pc_denorm)

    assert np.allclose(base_df.values, norm_df.values)