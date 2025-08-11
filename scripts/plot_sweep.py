import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "plots"


def plot_pressure_mae_vs_config(csv_path, output_path=None):
    """Create bar plot of pressure MAE for each configuration."""
    df = pd.read_csv(csv_path)
    if output_path is None:
        output_path = PLOTS_DIR / "pressure_mae_vs_config.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(df)), df["pressure_mae"])
    plt.xlabel("Configuration")
    plt.ylabel("Pressure MAE (m)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def main():
    p = argparse.ArgumentParser(description="Plot sweep results")
    p.add_argument("csv_path", help="CSV file produced by sweep_training.py")
    p.add_argument("--output", help="Output image path")
    args = p.parse_args()
    plot_pressure_mae_vs_config(args.csv_path, args.output)


if __name__ == "__main__":
    main()
