from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"


def plot_forecast_uncertainty(
    actual: Sequence[float],
    forecast: Sequence[float],
    timestamps: Sequence[Union[str, pd.Timestamp]],
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Plot forecast vs. actual demand with hourly error bands.

    Parameters
    ----------
    actual, forecast : sequence of floats
        Actual and forecast demand values.
    timestamps : sequence of datetime-like
        Timestamps corresponding to each demand observation.
    output_path : str or Path, optional
        Where to save the plot. Defaults to
        ``figures/forecast_uncertainty.png`` relative to the repository root.

    Returns
    -------
    pd.DataFrame
        Hourly statistics with columns ``actual``, ``forecast``, ``error`` and
        ``error_ci``.
    """
    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({"actual": actual, "forecast": forecast}, index=ts)
    df["hour"] = df.index.hour

    def _stats(group: pd.DataFrame) -> pd.Series:
        err = group["forecast"] - group["actual"]
        mean_err = err.mean()
        std_err = err.std(ddof=1)
        n = len(err)
        ci = 1.96 * std_err / np.sqrt(n) if n > 0 else 0.0
        return pd.Series(
            {
                "actual": group["actual"].mean(),
                "forecast": group["forecast"].mean(),
                "error": mean_err,
                "error_ci": ci,
            }
        )

    stats = df.groupby("hour").apply(_stats)
    hours = stats.index.to_numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(hours, stats["actual"], label="Actual")
    plt.plot(hours, stats["forecast"], label="Forecast")
    plt.fill_between(
        hours,
        stats["forecast"] - stats["error_ci"],
        stats["forecast"] + stats["error_ci"],
        color="gray",
        alpha=0.3,
        label="95% CI",
    )
    plt.xlabel("Hour of Day")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        output_path = FIGURES_DIR / "forecast_uncertainty.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return stats
