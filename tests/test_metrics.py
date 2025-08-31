import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Add the repository root so the `scripts` package can be imported when
# running the tests directly or through CI.
sys.path.append(str(REPO_ROOT))

from scripts.metrics import (
    accuracy_metrics,
    constraint_metrics,
    computational_metrics,
    per_node_mae,
)

def test_accuracy_metrics_basic():
    true_p = [10.0, 20.0]
    pred_p = [9.0, 21.0]
    df = accuracy_metrics(true_p, pred_p)
    assert np.isclose(df.loc["Mean Absolute Error (MAE)", "Pressure (m)"], 1.0)
    assert np.isclose(df.loc["Maximum Error", "Pressure (m)"], 1.0)


def test_constraint_metrics_basic():
    min_p = [30.0, 10.0, 25.0]
    energy = [1.0, 2.0, 3.0]
    df = constraint_metrics(min_p, energy, p_min=20.0)
    assert df.loc["Pressure Constraint Violations (hrs)", "Value"] == 1
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

    base_df = accuracy_metrics(true_p, pred_p)

    p_mean, p_std = true_p.mean(), true_p.std() + 1e-8
    tp_norm = (true_p - p_mean) / p_std
    pp_norm = (pred_p - p_mean) / p_std

    tp_denorm = tp_norm * p_std + p_mean
    pp_denorm = pp_norm * p_std + p_mean

    norm_df = accuracy_metrics(tp_denorm, pp_denorm)

    assert np.allclose(base_df.values, norm_df.values)


def test_per_node_mae_basic():
    true = [[1.0, 2.0], [3.0, 4.0]]
    pred = [[1.5, 1.0], [2.5, 5.0]]
    nodes = ["A", "B"]
    df = per_node_mae(true, pred, nodes)
    assert list(df["Junction"]) == nodes
    assert np.allclose(df["MAE"].values, [0.5, 1.0])