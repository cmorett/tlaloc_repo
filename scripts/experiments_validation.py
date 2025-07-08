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
from typing import Dict, List, Optional, Sequence
from datetime import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wntr
from wntr.metrics.economic import pump_energy
import epyt

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
PLOTS_DIR = REPO_ROOT / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def energy_pressure_tradeoff(
    strategy: Sequence[str],
    energy: Sequence[float],
    violations: Sequence[int],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Bar chart comparing energy usage and pressure violations."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(strategy))
    width = 0.4

    ax1.bar(x - width / 2, energy, width, color="tab:blue", label="Energy")
    ax1.set_ylabel("Energy Consumption (J)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, violations, width, color="tab:orange", label="Violations")
    ax2.set_ylabel("Number of Pressure Violations", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy)
    ax1.set_xlabel("Control Strategy")
    ax1.set_title("Energy vs Pressure Constraint Violations")

    fig.tight_layout()
    fig.savefig(plots_dir / f"energy_pressure_tradeoff_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def pressure_error_heatmap(
    errors: np.ndarray,
    times: Sequence[int],
    node_names: Sequence[str],
    inp_path: str,
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Heatmaps showing pressure prediction errors across nodes and time."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(errors, aspect="auto", cmap="magma")
    axes[0].set_xlabel("Node")
    axes[0].set_ylabel("Time (h)")
    axes[0].set_title("Heatmap of Pressure Prediction Errors")
    axes[0].set_yticks(np.arange(len(times)))
    axes[0].set_yticklabels(times)
    axes[0].set_xticks(np.arange(len(node_names)))
    axes[0].set_xticklabels(node_names, rotation=90, fontsize=6)
    fig.colorbar(im, ax=axes[0], label="Error (m)")

    net = epyt.epanet(str(inp_path))
    coords = net.getNodeCoordinates()
    node_index = {name: i + 1 for i, name in enumerate(net.NodeNameID)}
    avg_err = errors.mean(axis=0)
    x = [coords["x"][node_index[n]] for n in node_names]
    y = [coords["y"][node_index[n]] for n in node_names]
    sc = axes[1].scatter(x, y, c=avg_err, cmap="magma")
    axes[1].set_title("Average Node Error")
    fig.colorbar(sc, ax=axes[1], label="Error (m)")

    fig.tight_layout()
    fig.savefig(plots_dir / f"pressure_error_heatmap_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None

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
    demands: Optional[Dict[str, float]] = None,
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
        [dynamic demand, pressure, chlorine, elevation, pump1, ...].
    """

    num_nodes = len(wn.node_name_list)
    num_pumps = len(pump_controls)
    feats = torch.empty(
        (num_nodes, 4 + num_pumps),
        dtype=torch.float32,
        pin_memory=torch.cuda.is_available(),
    )
    pump_t = torch.tensor(pump_controls, dtype=torch.float32)
    for idx, name in enumerate(wn.node_name_list):
        node = wn.get_node(name)
        if demands is not None and name in demands:
            demand = demands.get(name, 0.0)
        elif name in wn.junction_name_list:
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

        feats[idx, 0] = float(demand)
        feats[idx, 1] = pressures.get(name, 0.0)
        feats[idx, 2] = chlorine.get(name, 0.0)
        feats[idx, 3] = float(elev)
        feats[idx, 4:] = pump_t
    return feats


def validate_surrogate(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    wn: wntr.network.WaterNetworkModel,
    test_results: List,
    device: torch.device,
    run_name: str,
    node_types_tensor: Optional[torch.Tensor] = None,
    edge_types_tensor: Optional[torch.Tensor] = None,
) -> tuple[Dict[str, float], np.ndarray, List[int]]:
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
    err_p_all: List[float] = []
    err_c_all: List[float] = []
    err_matrix: List[np.ndarray] = []
    err_times: List[int] = []
    node_types = {
        n: (
            "junction"
            if n in wn.junction_name_list
            else ("tank" if n in wn.tank_name_list else "reservoir")
        )
        for n in wn.node_name_list
    }
    err_p_by_type = {t: [] for t in set(node_types.values())}
    err_c_by_type = {t: [] for t in set(node_types.values())}
    model.eval()
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    with torch.no_grad():
        first = True
        for res in test_results:
            # ``data_generation.py`` stores tuples of ``(results, demand_scale)``.
            # Older pickle files may therefore provide the result object as the
            # first element of a tuple.  Support both formats here.
            if isinstance(res, tuple):
                res = res[0]

            pressures_df = res.node["pressure"].clip(lower=5.0)
            chlorine_df = res.node["quality"]
            demand_df = res.node.get("demand")
            pump_df = res.link["setting"][wn.pump_name_list]
            times = pressures_df.index
            pump_array = pump_df.values
            for i in range(len(times) - 1):
                p = pressures_df.iloc[i].to_dict()
                c = chlorine_df.iloc[i].to_dict()
                dem = demand_df.iloc[i].to_dict() if demand_df is not None else None
                controls = pump_array[i]
                feats = _prepare_features(wn, p, c, controls, model, dem)
                x = feats.to(device, non_blocking=True)
                if hasattr(model, "x_mean") and model.x_mean is not None:
                    x = (x - model.x_mean) / model.x_std
                if hasattr(model, "rnn"):
                    seq_in = x.unsqueeze(0).unsqueeze(0)
                    pred = model(
                        seq_in,
                        edge_index,
                        edge_attr,
                        node_types_tensor,
                        edge_types_tensor,
                    )
                    if isinstance(pred, dict):
                        node_pred = pred.get("node_outputs")[0, 0]
                        flow_pred = pred.get("edge_outputs")[0, 0].squeeze()
                    else:
                        node_pred = pred
                        flow_pred = None
                else:
                    pred = model(
                        x,
                        edge_index,
                        edge_attr,
                        node_types_tensor,
                        edge_types_tensor,
                    )
                    if isinstance(pred, dict):
                        node_pred = pred.get("node_outputs")
                        flow_pred = pred.get("edge_outputs").squeeze()
                    else:
                        node_pred = pred
                        flow_pred = None
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        y_mean_node = model.y_mean["node_outputs"].to(node_pred.device)
                        y_std_node = model.y_std["node_outputs"].to(node_pred.device)
                        node_pred = node_pred * y_std_node + y_mean_node
                    else:
                        node_pred = (
                            node_pred * model.y_std.to(node_pred.device)
                            + model.y_mean.to(node_pred.device)
                        )
                pred_p = node_pred[:, 0].cpu().numpy()
                pred_c = node_pred[:, 1].cpu().numpy()
                y_true_p = pressures_df.iloc[i + 1].to_numpy()
                y_true_c = chlorine_df.iloc[i + 1].to_numpy()
                # chlorine predictions were trained in log space so convert
                # predictions back to mg/L before computing errors
                pred_c = np.expm1(pred_c) * 1000.0

                diff_p = pred_p - y_true_p
                diff_c = pred_c - y_true_c
                if node_types_tensor is not None:
                    mask = (node_types_tensor == 0).cpu().numpy()
                    diff_p_masked = diff_p[mask]
                    diff_c_masked = diff_c[mask]
                else:
                    diff_p_masked = diff_p
                    diff_c_masked = diff_c
                err_p_all.extend(diff_p.tolist())
                err_c_all.extend(diff_c.tolist())
                for idx, name in enumerate(wn.node_name_list):
                    t = node_types[name]
                    err_p_by_type[t].append(float(diff_p[idx]))
                    err_c_by_type[t].append(float(diff_c[idx]))
                if first:
                    err_matrix.append(diff_p)
                    err_times.append(int(times[i + 1]))
                rmse_p += float((diff_p_masked ** 2).sum())
                rmse_c += float((diff_c_masked ** 2).sum())
                mae_p += float(np.abs(diff_p_masked).sum())
                mae_c += float(np.abs(diff_c_masked).sum())
                if diff_p_masked.size > 0:
                    max_err_p = max(max_err_p, float(np.max(np.abs(diff_p_masked))))
                if diff_c_masked.size > 0:
                    max_err_c = max(max_err_c, float(np.max(np.abs(diff_c_masked))))
                count += len(diff_p_masked)
                if flow_pred is not None:
                    mass_loss = compute_mass_balance_loss(
                        flow_pred,
                        edge_index,
                        len(wn.node_name_list),
                        node_type=node_types_tensor,
                    )
                    mass_total += mass_loss.item()
                    mass_count += 1
            if first:
                first = False

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

    if err_p_all:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].hist(err_p_all, bins=50, color="tab:blue", alpha=0.7)
        axes[0, 0].set_title("Pressure Error")
        axes[0, 1].hist(err_c_all, bins=50, color="tab:orange", alpha=0.7)
        axes[0, 1].set_title("Chlorine Error")
        types = list(err_p_by_type.keys())
        axes[1, 0].boxplot([err_p_by_type[t] for t in types], labels=types)
        axes[1, 0].set_title("Pressure Error by Node Type")
        axes[1, 1].boxplot([err_c_by_type[t] for t in types], labels=types)
        axes[1, 1].set_title("Chlorine Error by Node Type")
        for ax in axes.ravel():
            ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"error_histograms_{run_name}.png"))
        plt.close()

    err_arr = np.stack(err_matrix) if err_matrix else np.empty((0, len(wn.node_name_list)))
    return metrics, err_arr, err_times


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
                "controls": [1.0 for _ in pump_names],
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
                "controls": [1.0 if status == wntr.network.base.LinkStatus.Open else 0.0 for _ in pump_names],
            }
        )
    df = pd.DataFrame(log)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "baseline_heuristic.csv"), index=False)
    return df


def aggregate_and_plot(results: Dict[str, pd.DataFrame], run_name: str, Pmin: float) -> None:
    """Save combined CSV and generate simple plots."""

    combined = pd.concat(results, names=["method", None])
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    combined.to_csv(os.path.join(DATA_DIR, "all_results.csv"))

    plt.figure(figsize=(10, 4))
    for name, df in results.items():
        plt.plot(df["time"], df["min_pressure"], label=name)
    plt.xlabel("Hour")
    plt.ylabel("Minimum Pressure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_min_pressure_{run_name}.png"))
    plt.close()

    strategies = list(results.keys())
    energies = [df["energy"].sum() for df in results.values()]
    violations = [int((df["min_pressure"] < Pmin).sum()) for df in results.values()]
    energy_pressure_tradeoff(strategies, energies, violations, run_name)

    plt.figure(figsize=(10, 4))
    for name, df in results.items():
        plt.plot(df["time"], df["min_chlorine"], label=name)
    plt.xlabel("Hour")
    plt.ylabel("Minimum Chlorine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_chlorine_{run_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    for name, df in results.items():
        plt.plot(df["time"], df["energy"], label=name)
    plt.xlabel("Hour")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_energy_{run_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    for name, df in results.items():
        ctrl = np.stack(df["controls"].to_list())
        avg = ctrl.mean(axis=1)
        plt.plot(df["time"], avg, label=name)
    plt.xlabel("Hour")
    plt.ylabel("Average Pump Speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_controls_{run_name}.png"))
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
    parser.add_argument("--run-name", default="", help="Optional run name")
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable TorchScript compilation of the surrogate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        wn,
        node_to_index,
        pump_names,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
    ) = load_network(args.inp, return_edge_attr=True, return_features=True)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    model = load_surrogate_model(device, path=args.model, use_jit=not args.no_jit)

    if os.path.exists(args.test_pkl):
        with open(args.test_pkl, "rb") as f:
            test_res = pickle.load(f)
        metrics, err_arr, err_times = validate_surrogate(
            model,
            edge_index,
            edge_attr,
            wn,
            test_res,
            device,
            args.run_name or "",
            torch.tensor(node_types, dtype=torch.long, device=device),
            torch.tensor(edge_types, dtype=torch.long, device=device),
        )
        pd.DataFrame([metrics]).to_csv(
            os.path.join(DATA_DIR, "surrogate_validation.csv"), index=False
        )
        pressure_error_heatmap(
            err_arr,
            err_times,
            wn.node_name_list,
            args.inp,
            args.run_name or "",
        )
    else:
        print(f"{args.test_pkl} not found. Skipping surrogate validation.")

    mpc_df = simulate_closed_loop(
        wntr.network.WaterNetworkModel(args.inp),
        model,
        edge_index,
        edge_attr,
        feature_template.to(device),
        torch.tensor(node_types, dtype=torch.long, device=device),
        torch.tensor(edge_types, dtype=torch.long, device=device),
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

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_and_plot({"mpc": mpc_df, "heuristic": heur_df, "all_on": all_on_df}, run_name, args.Pmin)


if __name__ == "__main__":
    main()
