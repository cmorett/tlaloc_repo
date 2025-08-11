import argparse
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "logs"


def generate_configs(w_press, w_mass, w_head, depths, hiddens, residuals):
    """Create list of configuration dictionaries for the sweep."""
    configs = []
    for wp, wm, wh, d, h, r in product(w_press, w_mass, w_head, depths, hiddens, residuals):
        configs.append(
            {
                "w_press": wp,
                "w_mass": wm,
                "w_head": wh,
                "depth": d,
                "hidden": h,
                "residual": bool(r),
            }
        )
    return configs


def run_config(cmd, run_name):
    """Execute training command and collect metrics."""
    subprocess.run(cmd, check=True)
    acc_path = LOG_DIR / f"accuracy_{run_name}.csv"
    acc_df = pd.read_csv(acc_path, index_col=0)
    press_mae = float(acc_df.loc["Mean Absolute Error (MAE)", "Pressure (m)"])
    cl_mae = float(acc_df.loc["Mean Absolute Error (MAE)", "Chlorine (mg/L)"])
    # mass imbalance stored in training log
    log_path = DATA_DIR / f"training_{run_name}.log"
    mass_imb = None
    if log_path.exists():
        with open(log_path) as f:
            lines = [ln for ln in f.readlines() if ln and ln[0].isdigit()]
        if lines:
            parts = lines[-1].strip().split(",")
            if len(parts) >= 12:
                mass_imb = float(parts[11])
    return press_mae, cl_mae, mass_imb


def parse_args():
    p = argparse.ArgumentParser(description="Run hyperparameter sweep for GNN training")
    p.add_argument("--x-path", required=True)
    p.add_argument("--y-path", required=True)
    p.add_argument("--edge-index-path", required=True)
    p.add_argument("--inp-path", required=True)
    p.add_argument("--edge-attr-path")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--w-press", nargs="+", type=float, default=[1.0, 2.0])
    p.add_argument("--w-mass", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--w-head", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--depths", nargs="+", type=int, default=[3, 5])
    p.add_argument("--hiddens", nargs="+", type=int, default=[64, 128])
    p.add_argument(
        "--residuals",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Residual options: 0 disable, 1 enable",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output-csv", default=str(DATA_DIR / "sweep_results.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    configs = generate_configs(
        args.w_press, args.w_mass, args.w_head, args.depths, args.hiddens, args.residuals
    )
    results = []
    base_cmd = [
        "python",
        str(REPO_ROOT / "scripts" / "train_gnn.py"),
        "--x-path",
        args.x_path,
        "--y-path",
        args.y_path,
        "--edge-index-path",
        args.edge_index_path,
        "--inp-path",
        args.inp_path,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    if args.edge_attr_path:
        base_cmd += ["--edge-attr-path", args.edge_attr_path]
    if args.deterministic:
        base_cmd.append("--deterministic")

    for i, cfg in enumerate(configs):
        run_name = f"sweep_{i}"
        cmd = base_cmd + [
            "--w-press",
            str(cfg["w_press"]),
            "--w-mass",
            str(cfg["w_mass"]),
            "--w-head",
            str(cfg["w_head"]),
            "--num-layers",
            str(cfg["depth"]),
            "--hidden-dim",
            str(cfg["hidden"]),
            "--run-name",
            run_name,
        ]
        if cfg["residual"]:
            cmd.append("--residual")
        press_mae, cl_mae, mass_imb = run_config(cmd, run_name)
        results.append(
            {
                "run_name": run_name,
                **cfg,
                "pressure_mae": press_mae,
                "chlorine_mae": cl_mae,
                "mass_imbalance": mass_imb,
            }
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    baseline = df.iloc[0]
    valid = df[df["chlorine_mae"] < 0.02]
    if not valid.empty:
        best = valid.loc[valid["pressure_mae"].idxmin()]
        improvement = (baseline["pressure_mae"] - best["pressure_mae"]) / baseline["pressure_mae"]
        print(
            f"Best config {best['run_name']} improves pressure MAE by {improvement*100:.1f}%"
        )
    else:
        print("No configuration met the chlorine MAE constraint.")


if __name__ == "__main__":
    main()
