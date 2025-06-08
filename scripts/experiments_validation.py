"""Experiment runner for C-Town GNN surrogate and MPC control.

This script validates the trained GNN surrogate against EPANET results,
performs closed-loop MPC simulations, evaluates two baseline control
strategies, and aggregates all metrics into CSV files and plots.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wntr

# Compute absolute path to the repository's data directory so that results are
# always written inside the project regardless of the current working
# directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

from mpc_control import (
    load_network,
    load_surrogate_model,
    simulate_closed_loop,
)


def _prepare_features(
    wn: wntr.network.WaterNetworkModel,
    pressures: Dict[str, float],
    chlorine: Dict[str, float],
    pump_controls: np.ndarray,
) -> torch.Tensor:
    """Build node features including pump controls.

    Parameters
    ----------
    wn : WaterNetworkModel
        EPANET network instance.
    pressures, chlorine : dict
        Node pressure and chlorine dictionaries at the current time step.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[num_nodes, 4 + num_pumps]`` with
        [base_demand, pressure, chlorine, elevation, pump1, ...].
    """

    num_pumps = len(pump_controls)
    feats = []
    for name in wn.node_name_list:
        node = wn.get_node(name)
        if name in wn.junction_name_list:
            demand = node.demand_timeseries_list[0].base_value
        else:
            demand = 0.0

        if name in wn.junction_name_list or name in wn.tank_name_list:
            elev = node.elevation
        elif name in wn.reservoir_name_list:
            # ``Reservoir`` objects store their hydraulic head in ``base_head``
            # and expose ``head`` as ``None`` which previously caused a
            # ``TypeError`` when converting features to a tensor.
            elev = node.base_head
        else:
            elev = node.head

        if elev is None:
            elev = 0.0

        base = [demand, pressures.get(name, 0.0), chlorine.get(name, 0.0), elev]
        base.extend(pump_controls.tolist())
        feats.append(base)
    return torch.tensor(feats, dtype=torch.float32)


def validate_surrogate(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    wn: wntr.network.WaterNetworkModel,
    test_results: List[wntr.network.model.NetworkResults],
    device: torch.device,
) -> Dict[str, float]:
    """Compute RMSE of surrogate predictions against EPANET results."""

    rmse_p = 0.0
    rmse_c = 0.0
    count = 0
    model.eval()
    edge_index = edge_index.to(device)

    with torch.no_grad():
        for res in test_results:
            pressures_df = res.node["pressure"]
            chlorine_df = res.node["quality"]
            pump_df = res.link["status"][wn.pump_name_list]
            times = pressures_df.index
            pump_array = pump_df.values
            for i in range(len(times) - 1):
                p = pressures_df.iloc[i].to_dict()
                c = chlorine_df.iloc[i].to_dict()
                controls = pump_array[i]
                x = _prepare_features(wn, p, c, controls).to(device)
                pred = model(x, edge_index)
                y_true_p = pressures_df.iloc[i + 1].to_numpy()
                y_true_c = chlorine_df.iloc[i + 1].to_numpy()
                diff_p = pred[:, 0].cpu().numpy() - y_true_p
                diff_c = pred[:, 1].cpu().numpy() - y_true_c
                rmse_p += float((diff_p ** 2).sum())
                rmse_c += float((diff_c ** 2).sum())
                count += len(y_true_p)

    rmse_p = (rmse_p / count) ** 0.5
    rmse_c = (rmse_c / count) ** 0.5
    return {"pressure_rmse": rmse_p, "chlorine_rmse": rmse_c}


def run_all_pumps_on(
    wn: wntr.network.WaterNetworkModel,
    pump_names: List[str],
) -> pd.DataFrame:
    """Simulate with every pump on at full speed."""

    log = []
    for hour in range(24):
        for pn in pump_names:
            link = wn.get_link(pn)
            link.initial_status = wntr.network.base.LinkStatus.Open
            link.base_speed = 1.0
        sim = wntr.sim.EpanetSimulator(wn)
        wn.options.time.duration = 3600
        wn.options.time.report_timestep = 3600
        results = sim.run_sim()
        pressures = results.node["pressure"].iloc[-1].to_dict()
        chlorine = results.node["quality"].iloc[-1].to_dict()
        log.append(
            {
                "time": hour,
                "min_pressure": min(pressures.values()),
                "min_chlorine": min(chlorine.values()),
                "energy": float(len(pump_names)),
            }
        )
    df = pd.DataFrame(log)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "baseline_all_pumps_on.csv"), index=False)
    return df


def run_heuristic_baseline(
    wn: wntr.network.WaterNetworkModel,
    pump_names: List[str],
    threshold_p: float = 20.0,
    threshold_c: float = 0.2,
) -> pd.DataFrame:
    """Simple rule-based control using pressure/quality thresholds."""

    log = []
    sim = wntr.sim.EpanetSimulator(wn)
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    results = sim.run_sim()
    pressures = results.node["pressure"].iloc[-1].to_dict()
    chlorine = results.node["quality"].iloc[-1].to_dict()

    for hour in range(24):
        if min(pressures.values()) < threshold_p or min(chlorine.values()) < threshold_c:
            status = wntr.network.base.LinkStatus.Open
        else:
            status = wntr.network.base.LinkStatus.Closed
        for pn in pump_names:
            link = wn.get_link(pn)
            link.initial_status = status
            link.base_speed = 1.0
        sim = wntr.sim.EpanetSimulator(wn)
        wn.options.time.duration = 3600
        wn.options.time.report_timestep = 3600
        results = sim.run_sim()
        pressures = results.node["pressure"].iloc[-1].to_dict()
        chlorine = results.node["quality"].iloc[-1].to_dict()
        log.append(
            {
                "time": hour,
                "min_pressure": min(pressures.values()),
                "min_chlorine": min(chlorine.values()),
                "energy": float(status * len(pump_names)),
            }
        )
    df = pd.DataFrame(log)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "baseline_heuristic.csv"), index=False)
    return df


def aggregate_and_plot(results: Dict[str, pd.DataFrame]) -> None:
    """Save combined CSV and generate simple plots."""

    combined = pd.concat(results, names=["method", None])
    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(os.path.join(DATA_DIR, "all_results.csv"))

    plt.figure(figsize=(10, 4))
    for name, df in results.items():
        plt.plot(df["time"], df["min_pressure"], label=name)
    plt.xlabel("Hour")
    plt.ylabel("Minimum Pressure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "pressure_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    for name, df in results.items():
        plt.plot(df["time"], df["min_chlorine"], label=name)
    plt.xlabel("Hour")
    plt.ylabel("Minimum Chlorine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "chlorine_comparison.png"))
    plt.close()


def main() -> None:
    """Entry point for running all experiments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default="CTown.inp", help="EPANET network file")
    parser.add_argument(
        "--model", default="models/gnn_surrogate.pth", help="Trained surrogate weights"
    )
    parser.add_argument(
        "--test-pkl",
        default=os.path.join(DATA_DIR, "test_results_list.pkl"),
        help="Pickle file with test scenarios",
    )
    parser.add_argument("--horizon", type=int, default=6, help="MPC horizon")
    parser.add_argument("--iterations", type=int, default=50, help="GD iterations")
    parser.add_argument("--Pmin", type=float, default=20.0, help="Pressure threshold")
    parser.add_argument("--Cmin", type=float, default=0.2, help="Chlorine threshold")
    parser.add_argument(
        "--feedback-interval",
        type=int,
        default=1,
        help="Hours between EPANET synchronizations",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wn, node_to_index, pump_names, edge_index = load_network(args.inp)
    edge_index = edge_index.to(device)
    model = load_surrogate_model(device, path=args.model)

    if os.path.exists(args.test_pkl):
        with open(args.test_pkl, "rb") as f:
            test_res = pickle.load(f)
        rmse = validate_surrogate(model, edge_index, wn, test_res, device)
        print(
            f"Surrogate RMSE - Pressure: {rmse['pressure_rmse']:.2f}, "
            f"Chlorine: {rmse['chlorine_rmse']:.3f}"
        )
    else:
        print(f"{args.test_pkl} not found. Skipping surrogate validation.")

    mpc_df = simulate_closed_loop(
        wntr.network.WaterNetworkModel(args.inp),
        model,
        edge_index,
        args.horizon,
        args.iterations,
        node_to_index,
        pump_names,
        device,
        args.Pmin,
        args.Cmin,
        args.feedback_interval,
    )

    heur_df = run_heuristic_baseline(
        wntr.network.WaterNetworkModel(args.inp), pump_names
    )
    all_on_df = run_all_pumps_on(
        wntr.network.WaterNetworkModel(args.inp), pump_names
    )

    aggregate_and_plot({"mpc": mpc_df, "heuristic": heur_df, "all_on": all_on_df})


if __name__ == "__main__":
    main()
