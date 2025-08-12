"""Utility functions for reporting surrogate and MPC performance metrics."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _to_numpy(seq: Sequence[float]) -> np.ndarray:
    """Convert a sequence to a NumPy array of floats."""
    return np.asarray(seq, dtype=float)


def accuracy_metrics(
    true_pressure: Sequence[float],
    pred_pressure: Sequence[float],
    true_chlorine: Optional[Sequence[float]] = None,
    pred_chlorine: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Return accuracy metrics for pressure predictions and optionally chlorine.

    Parameters
    ----------
    true_pressure, pred_pressure : sequence of floats
        Ground truth and predicted pressures in meters.
    true_chlorine, pred_chlorine : sequence of floats, optional
        Ground truth and predicted chlorine levels in mg/L.

    Returns
    -------
    pd.DataFrame
        Table with MAE, RMSE, MAPE and maximum error for pressure and, when
        provided, chlorine.
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

    if true_chlorine is not None and pred_chlorine is not None:
        tc = _to_numpy(true_chlorine)
        pc = _to_numpy(pred_chlorine)
        abs_c = np.abs(tc - pc)
        mae_c = abs_c.mean()
        rmse_c = np.sqrt(((tc - pc) ** 2).mean())
        mape_c = (abs_c / np.maximum(np.abs(tc), 1e-8)).mean() * 100.0
        max_err_c = abs_c.max()
        data["Chlorine (mg/L)"] = [mae_c, rmse_c, mape_c, max_err_c]

    index = [
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Percentage Error",
        "Maximum Error",
    ]
    return pd.DataFrame(data, index=index)


def control_metrics(
    min_pressures: Sequence[float],
    min_chlorine: Sequence[float],
    energy_joules: Sequence[float],
    p_min: float,
    c_min: float,
) -> pd.DataFrame:
    """Compute constraint violations and total pump energy.

    Parameters
    ----------
    min_pressures : sequence of floats
        Minimum pressure recorded for each simulation step in meters.
    min_chlorine : sequence of floats
        Minimum chlorine concentration for each step in mg/L.
    energy_joules : sequence of floats
        Pump energy usage per step in Joules.
    p_min, c_min : float
        Operational lower bounds for pressure and chlorine.
    """
    p = _to_numpy(min_pressures)
    c = _to_numpy(min_chlorine)
    e = _to_numpy(energy_joules)

    pressure_violations = int(np.sum(p < p_min))
    chlorine_violations = int(np.sum(c < c_min))
    total_energy = float(e.sum())

    data = {
        "Value": [pressure_violations, chlorine_violations, total_energy]
    }
    index = [
        "Pressure Constraint Violations (hrs)",
        "Chlorine Constraint Violations (hrs)",
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


