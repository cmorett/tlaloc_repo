"""Experiment runner for C-Town GNN surrogate and MPC control.

This script validates the trained GNN surrogate against EPANET results,
performs closed-loop MPC simulations, evaluates two baseline control
strategies, and aggregates all metrics into CSV files and plots.
"""

from __future__ import annotations

import argparse
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wntr
from wntr.metrics.economic import pump_energy

# Ensure the repository root is importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loss_utils import compute_mass_balance_loss

# Compute absolute path to the repository's data directory so that results are
# always written inside the project regardless of the current working
# directory.
DATA_DIR = REPO_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

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
    model: torch.nn.Module,
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
    feats = torch.tensor(feats, dtype=torch.float32)
    if hasattr(model, "x_mean") and model.x_mean is not None:
        feats = (feats - model.x_mean.cpu()) / model.x_std.cpu()
    return feats


def validate_surrogate(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    wn: wntr.network.WaterNetworkModel,
    test_results: List,
    device: torch.device,
) -> Dict[str, float]:
    """Compute RMSE of surrogate predictions.

    ``test_results`` may either contain ``wntr`` ``NetworkResults`` objects or
    tuples of ``(NetworkResults, demand_scale)`` produced by
    ``data_generation.py``.  Only the results object is needed here, so tuples
    are automatically unpacked.
    """

    rmse_p = 0.0
    rmse_c = 0.0
    mae_p = 0.0
    mae_c = 0.0
    max_err_p = 0.0
    max_err_c = 0.0
    mass_total = 0.0
    mass_count = 0
    count = 0
    model.eval()
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    with torch.no_grad():
        for res in test_results:
            # ``data_generation.py`` stores tuples of ``(results, demand_scale)``.
            # Older pickle files may therefore provide the result object as the
            # first element of a tuple.  Support both formats here.
            if isinstance(res, tuple):
                res = res[0]

            pressures_df = res.node["pressure"]
            chlorine_df = res.node["quality"]
            pump_df = res.link["status"][wn.pump_name_list]
            times = pressures_df.index
            pump_array = pump_df.values
            for i in range(len(times) - 1):
                p = pressures_df.iloc[i].to_dict()
                c = chlorine_df.iloc[i].to_dict()
                controls = pump_array[i]
                x = _prepare_features(wn, p, c, controls, model).to(device)
                if hasattr(model, "rnn"):
                    seq_in = x.unsqueeze(0).unsqueeze(0)
                    pred = model(seq_in, edge_index, edge_attr)
                    if isinstance(pred, dict):
                        node_pred = pred.get("node_outputs")[0, 0]
                        flow_pred = pred.get("edge_outputs")[0, 0].squeeze()
                    else:
                        node_pred = pred
                        flow_pred = None
                else:
                    pred = model(x, edge_index, edge_attr)
                    if isinstance(pred, dict):
                        node_pred = pred.get("node_outputs")
                        flow_pred = pred.get("edge_outputs").squeeze()
                    else:
                        node_pred = pred
                        flow_pred = None
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    node_pred = node_pred * model.y_std + model.y_mean
                y_true_p = pressures_df.iloc[i + 1].to_numpy()
                y_true_c = chlorine_df.iloc[i + 1].to_numpy()
                diff_p = node_pred[:, 0].cpu().numpy() - y_true_p
                diff_c = node_pred[:, 1].cpu().numpy() - y_true_c
                rmse_p += float((diff_p ** 2).sum())
                rmse_c += float((diff_c ** 2).sum())
                mae_p += float(np.abs(diff_p).sum())
                mae_c += float(np.abs(diff_c).sum())
                max_err_p = max(max_err_p, float(np.max(np.abs(diff_p))))
                max_err_c = max(max_err_c, float(np.max(np.abs(diff_c))))
                count += len(y_true_p)
                if flow_pred is not None:
                    mass_loss = compute_mass_balance_loss(
                        flow_pred,
                        edge_index,
                        len(wn.node_name_list),
                    )
                    mass_total += mass_loss.item()
                    mass_count += 1

    rmse_p = (rmse_p / count) ** 0.5
    rmse_c = (rmse_c / count) ** 0.5
    mae_p = mae_p / count
    mae_c = mae_c / count

    print(
        f"[Metrics] RMSE (Pressure): {rmse_p:.4f} | MAE: {mae_p:.4f} | Max Err: {max_err_p:.4f}"
    )
    print(
        f"[Metrics] RMSE (Chlorine): {rmse_c:.4f} | MAE: {mae_c:.4f} | Max Err: {max_err_c:.4f}"
    )

    metrics = {
        "pressure_rmse": rmse_p,
        "chlorine_rmse": rmse_c,
        "pressure_mae": mae_p,
        "chlorine_mae": mae_c,
        "pressure_max_error": max_err_p,
        "chlorine_max_error": max_err_c,
    }
    if mass_count > 0:
        avg_mass = mass_total / mass_count
        print(f"[Validation] Avg node imbalance (kg/s): {avg_mass:.3e}")
        metrics["avg_mass_imbalance"] = avg_mass

    os.makedirs(REPO_ROOT / "logs", exist_ok=True)
    with open(REPO_ROOT / "logs" / "surrogate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


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
        wn.options.time.start_clocktime = hour * 3600
        wn.options.time.duration = 3600
        wn.options.time.report_timestep = 3600
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim(str(TEMP_DIR / "temp"))
        pressures = results.node["pressure"].iloc[-1].to_dict()
        chlorine = results.node["quality"].iloc[-1].to_dict()
        energy_df = pump_energy(
            results.link["flowrate"][pump_names], results.node["head"], wn
        )
        energy = energy_df[pump_names].iloc[-1].sum()
        log.append(
            {
                "time": hour,
                "min_pressure": max(
                    min(
                        pressures[n]
                        for n in wn.junction_name_list + wn.tank_name_list
                    ),
                    0.0,
                ),
                "min_chlorine": max(
                    min(
                        chlorine[n]
                        for n in wn.junction_name_list + wn.tank_name_list
                    ),
                    0.0,
                ),
                "energy": float(energy),
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
    wn.options.time.start_clocktime = 0
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(str(TEMP_DIR / "temp"))
    pressures = results.node["pressure"].iloc[-1].to_dict()
    chlorine = results.node["quality"].iloc[-1].to_dict()

    for hour in range(24):
        if min(
            pressures[n] for n in wn.junction_name_list + wn.tank_name_list
        ) < threshold_p or min(
            chlorine[n] for n in wn.junction_name_list + wn.tank_name_list
        ) < threshold_c:
            status = wntr.network.base.LinkStatus.Open
        else:
            status = wntr.network.base.LinkStatus.Closed
        for pn in pump_names:
            link = wn.get_link(pn)
            link.initial_status = status
            link.base_speed = 1.0
        wn.options.time.start_clocktime = (hour + 1) * 3600
        wn.options.time.duration = 3600
        wn.options.time.report_timestep = 3600
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim(str(TEMP_DIR / "temp"))
        pressures = results.node["pressure"].iloc[-1].to_dict()
        chlorine = results.node["quality"].iloc[-1].to_dict()
        energy_df = pump_energy(
            results.link["flowrate"][pump_names], results.node["head"], wn
        )
        energy = energy_df[pump_names].iloc[-1].sum()
        log.append(
            {
                "time": hour,
                "min_pressure": max(
                    min(
                        pressures[n]
                        for n in wn.junction_name_list + wn.tank_name_list
                    ),
                    0.0,
                ),
                "min_chlorine": max(
                    min(
                        chlorine[n]
                        for n in wn.junction_name_list + wn.tank_name_list
                    ),
                    0.0,
                ),
                "energy": float(energy),
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
    wn, node_to_index, pump_names, edge_index, edge_attr = load_network(
        args.inp, return_edge_attr=True
    )
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    model = load_surrogate_model(device, path=args.model)

    if os.path.exists(args.test_pkl):
        with open(args.test_pkl, "rb") as f:
            test_res = pickle.load(f)
        metrics = validate_surrogate(model, edge_index, edge_attr, wn, test_res, device)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(DATA_DIR, "surrogate_validation.csv"), index=False
        )
    else:
        print(f"{args.test_pkl} not found. Skipping surrogate validation.")

    mpc_df = simulate_closed_loop(
        wntr.network.WaterNetworkModel(args.inp),
        model,
        edge_index,
        edge_attr,
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
