import argparse
import subprocess
import time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"
MODEL_DIR = REPO_ROOT / "models"


def run_config(base_cmd, name):
    cmd = base_cmd + ["--run-name", name, "--output", str(MODEL_DIR / f"{name}.pth")]
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    df = pd.read_csv(LOG_DIR / f"accuracy_{name}.csv", index_col=0)
    mae = float(df.loc["Mean Absolute Error (MAE)", "Pressure (m)"])
    return mae, elapsed


def main():
    p = argparse.ArgumentParser(description="Run ablation studies for the GNN surrogate")
    p.add_argument("--x-path", required=True)
    p.add_argument("--y-path", required=True)
    p.add_argument("--edge-index-path", required=True)
    p.add_argument("--edge-attr-path")
    p.add_argument("--inp-path", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    base_cmd = [
        "python", str(REPO_ROOT / "scripts" / "train_gnn.py"),
        "--x-path", args.x_path,
        "--y-path", args.y_path,
        "--edge-index-path", args.edge_index_path,
        "--inp-path", args.inp_path,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.edge_attr_path:
        base_cmd += ["--edge-attr-path", args.edge_attr_path]

    configs = [
        ("baseline", []),
        ("residual", ["--residual"]),
        ("deep", ["--num-layers", "8", "--hidden-dim", "256", "--residual"]),
        ("attention", ["--use-attention"]),
    ]

    results = []
    for name, extra in configs:
        mae, t = run_config(base_cmd + extra, name)
        results.append((name, mae, t))

    baseline_mae, baseline_time = results[0][1], results[0][2]
    for name, mae, t in results:
        impr = (baseline_mae - mae) / baseline_mae if baseline_mae else 0.0
        speed = t / baseline_time if baseline_time else 0.0
        print(f"{name}: MAE={mae:.4f} m, improvement={impr*100:.1f}%, time x{speed:.2f}")


if __name__ == "__main__":
    main()
