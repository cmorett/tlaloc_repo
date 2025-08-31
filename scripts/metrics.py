"""Utility functions for reporting surrogate and MPC performance metrics."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _to_numpy(seq: Sequence[float]) -> np.ndarray:
    """Convert a sequence to a NumPy array of floats."""
    return np.asarray(seq, dtype=float)


def accuracy_metrics(
    true_pressure: Sequence[float],
    pred_pressure: Sequence[float],
) -> pd.DataFrame:
    """Return accuracy metrics for pressure predictions.

    Parameters
    ----------
    true_pressure, pred_pressure : sequence of floats
        Ground truth and predicted pressures in meters.

    Returns
    -------
    pd.DataFrame
        Table with MAE, RMSE, MAPE and maximum error for pressure.
    """
    tp = _to_numpy(true_pressure)
    pp = _to_numpy(pred_pressure)

    abs_p = np.abs(tp - pp)
    mae_p = abs_p.mean()
    rmse_p = np.sqrt(((tp - pp) ** 2).mean())
    mape_p = (abs_p / np.maximum(np.abs(tp), 1e-8)).mean() * 100.0
    max_err_p = abs_p.max()

    data = {
        "Pressure (m)": [mae_p, rmse_p, mape_p, max_err_p]
    }

    index = [
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Percentage Error",
        "Maximum Error",
    ]
    return pd.DataFrame(data, index=index)


def constraint_metrics(
    min_pressures: Sequence[float],
    energy_joules: Sequence[float],
    p_min: float,
) -> pd.DataFrame:
    """Compute pressure constraint violations and total pump energy.

    Parameters
    ----------
    min_pressures : sequence of floats
        Minimum pressure recorded for each simulation step in meters.
    energy_joules : sequence of floats
        Pump energy usage per step in Joules.
    p_min : float
        Operational lower bound for pressure.
    """
    p = _to_numpy(min_pressures)
    e = _to_numpy(energy_joules)

    pressure_violations = int(np.sum(p < p_min))
    total_energy = float(e.sum())

    data = {"Value": [pressure_violations, total_energy]}
    index = [
        "Pressure Constraint Violations (hrs)",
        "Total Pump Energy (J)",
    ]
    return pd.DataFrame(data, index=index)


def computational_metrics(
    inference_times: Sequence[float],
    optimisation_times: Sequence[float],
) -> pd.DataFrame:
    """Return average inference and optimisation runtimes in milliseconds."""
    inf = _to_numpy(inference_times) * 1000.0
    opt = _to_numpy(optimisation_times) * 1000.0

    data = {
        "Average Time (ms)": [inf.mean(), opt.mean()]
    }
    index = [
        "Inference Time per Simulation Step",
        "Optimization Time per Control Step",
    ]
    return pd.DataFrame(data, index=index)


def per_node_mae(
    true_pressure: Sequence[Sequence[float]],
    pred_pressure: Sequence[Sequence[float]],
    node_names: Sequence[str],
) -> pd.DataFrame:
    """Return mean absolute error for each junction.

    Parameters
    ----------
    true_pressure, pred_pressure : sequence of sequence of floats
        Ground truth and predicted pressures with shape ``(N, num_nodes)``.
    node_names : sequence of str
        Names of the junctions corresponding to the columns.

    Returns
    -------
    pd.DataFrame
        Table with columns ``Junction`` and ``MAE`` listing the error per node.
    """

    tp = _to_numpy(true_pressure)
    pp = _to_numpy(pred_pressure)
    if tp.shape != pp.shape:
        raise ValueError("Shape mismatch between true and predicted pressure arrays")
    mae = np.abs(tp - pp).mean(axis=0)
    return pd.DataFrame({"Junction": list(node_names), "MAE": mae})


def export_table(df: pd.DataFrame, path: str) -> None:
    """Export a metric table to CSV, Excel or JSON based on file suffix."""
    p = Path(path)
    if p.suffix == ".csv":
        df.to_csv(p)
    elif p.suffix in {".xls", ".xlsx"}:
        df.to_excel(p)
    elif p.suffix == ".json":
        df.to_json(p, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported export format: {p.suffix}")


