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
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wntr
from wntr.metrics.economic import pump_energy
import epyt
from tqdm import tqdm
import imageio.v2 as imageio

try:
    from .reproducibility import configure_seeds, save_config
except ImportError:  # pragma: no cover
    from reproducibility import configure_seeds, save_config

# Minimum allowed pressure [m] applied during preprocessing.
MIN_PRESSURE = 5.0
# Ensure the repository root is importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loss_utils import compute_mass_balance_loss
try:
    from .feature_utils import build_static_node_features, prepare_node_features
except ImportError:  # pragma: no cover
    from feature_utils import build_static_node_features, prepare_node_features

# Compute absolute path to the repository's data directory so that results are
# always written inside the project regardless of the current working
# directory.
DATA_DIR = REPO_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
PLOTS_DIR = REPO_ROOT / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
RUNS_DIR = REPO_ROOT / "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


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


def animate_mpc_network(
    mpc_df: pd.DataFrame,
    pump_names: Sequence[str],
    run_name: str,
    inp_path: Optional[str] = None,
    plots_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Create GIF and HTML animations of MPC network pressures and pump speeds."""
    from epyt import epanet as _epanet  # local import to keep startup fast

    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    if inp_path is None:
        inp_path = REPO_ROOT / "CTown.inp"

    net = _epanet(str(inp_path))
    coords = net.getNodeCoordinates()
    node_names = net.getNodeNameID()
    xs = [coords["x"][i + 1] for i in range(len(node_names))]
    ys = [coords["y"][i + 1] for i in range(len(node_names))]
    link_names = net.getLinkNameID()

    frames = []
    for _, row in mpc_df.iterrows():
        pressures = row.get("pressures", {})
        speeds = row.get("controls", [])
        pump_controls = {pump_names[i]: speeds[i] for i in range(len(pump_names))}

        fig, ax = plt.subplots(figsize=(5, 5))
        vals = [pressures.get(n, 0.0) for n in node_names]
        sc = ax.scatter(xs, ys, c=vals, cmap="coolwarm", s=25)
        plt.colorbar(sc, ax=ax, label="Pressure (m)")
        for lname in link_names:
            s_idx, e_idx = net.getLinkNodesIndex(lname)
            x1, y1 = coords["x"][s_idx], coords["y"][s_idx]
            x2, y2 = coords["x"][e_idx], coords["y"][e_idx]
            width = 1.0
            color = "lightgray"
            if lname in pump_controls:
                width = 0.5 + pump_controls[lname]
                color = "tab:green"
            ax.plot([x1, x2], [y1, y2], linewidth=width, color=color)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Pressure (m)")
        fig.suptitle(f"Hour {int(row['time'])}")
        frame_path = plots_dir / f"mpc_frame_{run_name}_t{int(row['time'])}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    net.closeNetwork()

    gif_path = plots_dir / f"mpc_animation_{run_name}.gif"
    html_path = plots_dir / f"mpc_animation_{run_name}.html"
    if frames:
        imageio.mimsave(gif_path, frames, duration=0.8)
        with open(html_path, "w") as f:
            f.write(f"<img src='{gif_path.name}' />")
    return gif_path, html_path

try:
    from .mpc_control import (
        load_network,
        load_surrogate_model,
        simulate_closed_loop,
    )
except ImportError:  # pragma: no cover
    from mpc_control import (
        load_network,
        load_surrogate_model,
        simulate_closed_loop,
    )



def _prepare_features(
    wn: wntr.network.WaterNetworkModel,
    pressures: Dict[str, float],
    pump_controls: np.ndarray,
    model: torch.nn.Module,
    demands: Optional[Dict[str, float]] = None,
    skip_normalization: bool = False,
) -> torch.Tensor:
    """Assemble and optionally normalize node features."""

    num_pumps = len(pump_controls)
    if not hasattr(_prepare_features, "template"):
        _prepare_features.template = build_static_node_features(wn, num_pumps)
    template = _prepare_features.template

    pressures_t = torch.tensor(
        [
            wn.get_node(n).base_head if n in wn.reservoir_name_list else pressures.get(n, 0.0)
            for n in wn.node_name_list
        ],
        dtype=torch.float32,
    )
    pump_t = torch.tensor(pump_controls, dtype=torch.float32)
    demands_t = None
    if demands is not None:
        demands_t = torch.tensor(
            [demands.get(n, 0.0) for n in wn.node_name_list], dtype=torch.float32
        )
    return prepare_node_features(
        template,
        pressures_t,
        pump_t,
        model,
        demands_t,
        skip_normalization=skip_normalization,
    )

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
    debug: bool = False,
    debug_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], np.ndarray, List[int]] | Tuple[Dict[str, float], np.ndarray, List[int], Dict[str, Any]]:
    """Compute RMSE of surrogate predictions.

    ``test_results`` may either contain ``wntr`` ``NetworkResults`` objects or
    tuples of ``(NetworkResults, demand_scale)`` produced by
    ``data_generation.py``.  Only the results object is needed here, so tuples
    are automatically unpacked.
    """

    if edge_attr is not None and hasattr(model, "edge_dim"):
        if edge_attr.size(1) != model.edge_dim:
            raise ValueError(
                f"Edge attribute dimension mismatch: model expects {model.edge_dim}, "
                f"but received {edge_attr.size(1)}."
            )

    rmse_p = 0.0
    mae_p = 0.0
    rmse_p_all = 0.0
    mae_p_all = 0.0
    max_err_p = 0.0
    mass_total = 0.0
    mass_count = 0
    count = 0
    count_all = 0
    err_p_all: List[float] = []
    err_matrix: List[np.ndarray] = []
    err_times: List[int] = []
    if debug and debug_info is None:
        debug_info = {}
    node_types = {
        n: (
            "junction"
            if n in wn.junction_name_list
            else ("tank" if n in wn.tank_name_list else "reservoir")
        )
        for n in wn.node_name_list
    }
    err_p_by_type = {t: [] for t in set(node_types.values())}
    model.eval()
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
        if getattr(model, "edge_mean", None) is not None:
            edge_attr = (edge_attr - model.edge_mean) / (model.edge_std + 1e-8)

    with torch.no_grad():
        first = True
        for res in tqdm(
            test_results,
            desc="Validating scenarios",
            disable=__name__ != "__main__",
        ):
            # ``data_generation.py`` stores tuples of ``(results, demand_scale)``.
            # Older pickle files may therefore provide the result object as the
            # first element of a tuple.  Support both formats here.
            if isinstance(res, tuple):
                res = res[0]

            # Clip pressures to match preprocessing and align all DataFrame
            # columns with the network's node ordering. This avoids
            # misaligned ground-truth vectors when converting to numpy arrays.
            pressures_df = (
                res.node["pressure"].clip(lower=MIN_PRESSURE).reindex(columns=wn.node_name_list)
            )
            demand_df = res.node.get("demand")
            if demand_df is not None:
                demand_df = demand_df.reindex(columns=wn.node_name_list)
            assert list(pressures_df.columns) == wn.node_name_list
            pump_df = res.link["setting"][wn.pump_name_list]
            times = pressures_df.index
            pump_array = np.clip(pump_df.values, 0.0, 1.0)
            for i in tqdm(
                range(len(times) - 1),
                desc="Sim steps",
                leave=False,
                disable=__name__ != "__main__",
            ):
                p = pressures_df.iloc[i].to_dict()
                dem = demand_df.iloc[i].to_dict() if demand_df is not None else None
                controls = pump_array[i]
                feats = _prepare_features(wn, p, controls, model, dem)
                if debug and first and i == 0:
                    pre = _prepare_features(
                        wn, p, controls, model, dem, skip_normalization=True
                    )
                    feats_np = pre.cpu().numpy()
                    debug_info["node_pre_norm_stats"] = {
                        "min": feats_np.min(axis=0).tolist(),
                        "max": feats_np.max(axis=0).tolist(),
                        "mean": feats_np.mean(axis=0).tolist(),
                        "std": feats_np.std(axis=0).tolist(),
                    }
                    x_np = feats.cpu().numpy()
                    debug_info["node_post_norm_stats"] = {
                        "min": x_np.min(axis=0).tolist(),
                        "max": x_np.max(axis=0).tolist(),
                        "mean": x_np.mean(axis=0).tolist(),
                        "std": x_np.std(axis=0).tolist(),
                    }
                x = feats.to(device, non_blocking=True)
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
                        node_pred = pred[0, 0]
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
                        y_mean = model.y_mean["node_outputs"].to(node_pred.device)
                        y_std = model.y_std["node_outputs"].to(node_pred.device)
                    else:
                        y_mean = model.y_mean.to(node_pred.device)
                        y_std = model.y_std.to(node_pred.device)
                    target_dim = y_mean.shape[0]
                    node_pred = node_pred[:, :target_dim]
                    node_pred = node_pred * y_std + y_mean
                pred_p = node_pred[:, 0].cpu().numpy()
                y_true_p = pressures_df.iloc[i + 1].to_numpy()
                for j, name in enumerate(wn.node_name_list):
                    if name in wn.reservoir_name_list:
                        y_true_p[j] = wn.get_node(name).base_head

                diff_p = pred_p - y_true_p
                if node_types_tensor is not None:
                    mask = (node_types_tensor == 0).cpu().numpy()
                    diff_p_masked = diff_p[mask]
                else:
                    diff_p_masked = diff_p
                err_p_all.extend(diff_p.tolist())
                for idx, name in enumerate(wn.node_name_list):
                    t = node_types[name]
                    err_p_by_type[t].append(float(diff_p[idx]))
                if first:
                    err_matrix.append(diff_p)
                    err_times.append(int(times[i + 1]))
                rmse_p += float((diff_p_masked ** 2).sum())
                mae_p += float(np.abs(diff_p_masked).sum())
                rmse_p_all += float((diff_p ** 2).sum())
                mae_p_all += float(np.abs(diff_p).sum())
                if diff_p_masked.size > 0:
                    max_err_p = max(max_err_p, float(np.max(np.abs(diff_p_masked))))
                count += len(diff_p_masked)
                count_all += len(diff_p)
                if flow_pred is not None:
                    mass_loss = compute_mass_balance_loss(
                        flow_pred,
                        edge_index,
                        len(wn.node_name_list),
                        node_type=node_types_tensor,
                    )
                    mass_total += mass_loss.item()
                    mass_count += 1
                if debug and first and i == 0:
                    mini_mae = float(np.abs(diff_p_masked).mean()) if diff_p_masked.size > 0 else float("nan")
                    debug_info["mini_batch_mae"] = mini_mae
                    mask_bool = mask if node_types_tensor is not None else np.ones_like(diff_p, dtype=bool)
                    rows = []
                    for idx, name in enumerate(wn.node_name_list):
                        rows.append(
                            {
                                "node_id": name,
                                "node_type": node_types[name],
                                "true_pressure": float(y_true_p[idx]),
                                "pred_pressure": float(pred_p[idx]),
                                "error_m": float(diff_p[idx]),
                                "included_in_mask": bool(mask_bool[idx]),
                            }
                        )
                    debug_info["sample_df"] = pd.DataFrame(rows[:10])
            if first:
                first = False

    rmse_p = (rmse_p / count) ** 0.5
    mae_p = mae_p / count
    if count_all > 0:
        rmse_p_all = (rmse_p_all / count_all) ** 0.5
        mae_p_all = mae_p_all / count_all
    if debug_info is not None:
        debug_info["mae_pressure_all"] = mae_p_all
        debug_info["rmse_pressure_all"] = rmse_p_all
        debug_info["node_type_counts"] = {k: len(v) for k, v in err_p_by_type.items()}
        if node_types_tensor is not None:
            mask_bool = (node_types_tensor == 0).cpu().numpy()
            debug_info["mask_included"] = int(mask_bool.sum())
            debug_info["mask_excluded"] = int(len(mask_bool) - mask_bool.sum())

    print(
        f"[Metrics] RMSE (Pressure): {rmse_p:.4f} | MAE: {mae_p:.4f} | Max Err: {max_err_p:.4f}"
    )

    metrics = {
        "pressure_rmse": rmse_p,
        "pressure_mae": mae_p,
        "pressure_max_error": max_err_p,
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
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(err_p_all, bins=50, color="tab:blue", alpha=0.7)
        axes[0].set_title("Pressure Error")
        types = list(err_p_by_type.keys())
        axes[1].boxplot([err_p_by_type[t] for t in types], labels=types)
        axes[1].set_title("Pressure Error by Node Type")
        for ax in axes:
            ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"error_histograms_{run_name}.png"))
        plt.close()

    err_arr = np.stack(err_matrix) if err_matrix else np.empty((0, len(wn.node_name_list)))
    if debug:
        return metrics, err_arr, err_times, (debug_info or {})
    return metrics, err_arr, err_times


def rollout_surrogate(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    wn: wntr.network.WaterNetworkModel,
    test_results: List,
    device: torch.device,
    steps: int,
    node_types_tensor: Optional[torch.Tensor] = None,
    edge_types_tensor: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Roll out surrogate predictions for multiple steps without EPANET feedback.

    Returns per-step RMSE for pressure averaged across scenarios.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")

    model.eval()
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
        if getattr(model, "edge_mean", None) is not None:
            edge_attr = (edge_attr - model.edge_mean) / (model.edge_std + 1e-8)

    sq_p = np.zeros(steps, dtype=float)
    counts = np.zeros(steps, dtype=int)

    with torch.no_grad():
        for res in test_results:
            if isinstance(res, tuple):
                res = res[0]
            pressures_df = res.node["pressure"].clip(lower=MIN_PRESSURE)
            demand_df = res.node.get("demand")
            pump_df = res.link["setting"][wn.pump_name_list]
            pump_array = np.clip(pump_df.values, 0.0, 1.0)

            current_p = {n: float(v) for n, v in pressures_df.iloc[0].to_dict().items()}

            max_steps = min(steps, len(pressures_df.index) - 1)
            for t in range(max_steps):
                dem = demand_df.iloc[t].to_dict() if demand_df is not None else None
                controls = pump_array[t]
                feats = _prepare_features(wn, current_p, controls, model, dem)
                x = feats.to(device, non_blocking=True)
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
                    else:
                        node_pred = pred
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
                    else:
                        node_pred = pred
                if hasattr(model, "y_mean") and model.y_mean is not None:
                    if isinstance(model.y_mean, dict):
                        y_mean_node = model.y_mean["node_outputs"].to(node_pred.device)
                        y_std_node = model.y_std["node_outputs"].to(node_pred.device)
                        num_targets = y_std_node.shape[-1]
                        if node_pred.shape[1] < num_targets:
                            raise ValueError(
                                "node_pred has fewer columns than model.y_std"
                            )
                        node_pred = node_pred[:, :num_targets]
                        node_pred = node_pred * y_std_node + y_mean_node
                    else:
                        y_mean = model.y_mean.to(node_pred.device)
                        y_std = model.y_std.to(node_pred.device)
                        num_targets = y_std.shape[-1]
                        if node_pred.shape[1] < num_targets:
                            raise ValueError(
                                f"node_pred has {node_pred.shape[1]} columns but y_std expects {num_targets}"
                            )
                        node_pred = node_pred[:, :num_targets]
                        node_pred = node_pred * y_std + y_mean
                pred_p = node_pred[:, 0].cpu().numpy()
                true_p = pressures_df.iloc[t + 1].to_numpy()
                for j, name in enumerate(wn.node_name_list):
                    if name in wn.reservoir_name_list:
                        true_p[j] = wn.get_node(name).base_head
                diff_p = pred_p - true_p
                if node_types_tensor is not None:
                    mask = (node_types_tensor == 0).cpu().numpy()
                    diff_p = diff_p[mask]
                sq_p[t] += float((diff_p ** 2).sum())
                counts[t] += len(diff_p)

                current_p = {n: float(pred_p[i]) for i, n in enumerate(wn.node_name_list)}

    rmse_p = np.sqrt(np.divide(sq_p, counts, out=np.zeros_like(sq_p), where=counts > 0))
    return rmse_p


def run_all_pumps_on(
    wn: wntr.network.WaterNetworkModel,
    pump_names: List[str],
) -> pd.DataFrame:
    """Simulate with every pump on at full speed."""

    log = []
    for hour in tqdm(
        range(24),
        desc="Baseline: all pumps on",
        disable=__name__ != "__main__",
    ):
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
        energy_df = pump_energy(
            results.link["flowrate"][pump_names], results.node["head"], wn
        )
        energy = energy_df[pump_names].iloc[-1].sum()
        log.append(
            {
                "time": hour,
                "min_pressure": max(
                    min(pressures[n] for n in wn.junction_name_list),
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
) -> pd.DataFrame:
    """Simple rule-based control using a pressure threshold."""

    log = []
    wn.options.time.start_clocktime = 0
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(str(TEMP_DIR / "temp"))
    pressures = results.node["pressure"].iloc[-1].to_dict()

    for hour in tqdm(
        range(24),
        desc="Baseline: heuristic",
        disable=__name__ != "__main__",
    ):
        if min(pressures[n] for n in wn.junction_name_list) < threshold_p:
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
        energy_df = pump_energy(
            results.link["flowrate"][pump_names], results.node["head"], wn
        )
        energy = energy_df[pump_names].iloc[-1].sum()
        log.append(
            {
                "time": hour,
                "min_pressure": max(
                    min(pressures[n] for n in wn.junction_name_list),
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
        "--norm-stats",
        type=Path,
        default=None,
        help="Path to .npz file with normalization statistics",
    )
    parser.add_argument(
        "--test-pkl",
        default=os.path.join(DATA_DIR, "test_results_list.pkl"),
        help="Pickle file with test scenarios",
    )
    parser.add_argument("--horizon", type=int, default=6, help="MPC horizon")
    parser.add_argument("--iterations", type=int, default=50, help="GD iterations")
    parser.add_argument("--Pmin", type=float, default=20.0, help="Pressure threshold")
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable forensic debug mode with additional diagnostics",
    )
    parser.add_argument(
        "--force-arch-mismatch",
        action="store_true",
        help="Force model loading even if architecture metadata mismatches",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=24,
        help="Number of steps for surrogate roll-out evaluation",
    )
    parser.add_argument(
        "--rollout-eval",
        action="store_true",
        help="Run N-step surrogate roll-out validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch ops",
    )
    parser.add_argument("--w_p", type=float, default=100.0, help="Weight on pressure violations")
    parser.add_argument("--w_e", type=float, default=1.0, help="Weight on energy usage")
    args = parser.parse_args()
    configure_seeds(args.seed, args.deterministic)
    if args.feedback_interval > 1:
        print(
            f"WARNING: --feedback-interval set to {args.feedback_interval}; "
            "surrogate predictions may drift without hourly EPANET feedback."
        )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.debug:
        print(f"[DEBUG] model path: {args.model}")
        if not os.path.exists(args.model):
            raise FileNotFoundError(args.model)
        if args.norm_stats:
            print(f"[DEBUG] norm stats path: {args.norm_stats}")
            if not args.norm_stats.exists():
                raise FileNotFoundError(args.norm_stats)
            arr = np.load(args.norm_stats)
            md5 = hashlib.md5()
            arrays = [arr["x_mean"], arr["x_std"]]
            if "y_mean_node" in arr:
                arrays.extend([arr["y_mean_node"], arr["y_std_node"]])
                if "y_mean_edge" in arr:
                    arrays.extend([arr["y_mean_edge"], arr["y_std_edge"]])
            elif "y_mean" in arr:
                arrays.extend([arr["y_mean"], arr["y_std"]])
            if "edge_mean" in arr:
                arrays.extend([arr["edge_mean"], arr["edge_std"]])
            for a in arrays:
                md5.update(a.tobytes())
            print(f"[DEBUG] norm_stats_md5: {md5.hexdigest()}")
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
    if args.debug and edge_attr is not None:
        ea = edge_attr.cpu().numpy()
        edge_stats = {
            "min": ea.min(axis=0).tolist(),
            "max": ea.max(axis=0).tolist(),
            "mean": ea.mean(axis=0).tolist(),
            "std": ea.std(axis=0).tolist(),
        }
        print(f"[DEBUG] edge_attr_stats: {edge_stats}")
    model = load_surrogate_model(
        device,
        path=args.model,
        use_jit=not args.no_jit,
        norm_stats_path=str(args.norm_stats) if args.norm_stats else None,
        force_arch_mismatch=args.force_arch_mismatch,
    )
    torch.set_grad_enabled(False)
    if args.debug:
        params = sum(p.numel() for p in model.parameters())
        first_w = next(model.parameters()).detach().abs().sum().item()
        print(f"[DEBUG] model params: {params}, first_layer_abs_sum: {first_w:.4f}")
    if getattr(model, "y_mean", None) is None or getattr(model, "y_std", None) is None:
        raise RuntimeError(
            "Model is missing output normalization statistics. "
            "Provide --norm-stats pointing to an .npz file with y_mean and y_std."
        )
    # Compare normalization checksum from checkpoint metadata with the provided .npz
    ckpt_hash = getattr(model, "_ckpt_meta", {}).get("norm_stats_md5")
    stats_path = Path(args.norm_stats) if args.norm_stats else Path(str(Path(args.model).with_suffix("")) + "_norm.npz")
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    if ckpt_hash is None or not stats_path.exists():
        raise RuntimeError(
            "Normalization statistics missing or checksum not found. "
            "Regenerate or supply matching normalization files."
        )
    arr = np.load(stats_path)
    md5 = hashlib.md5()
    arrays = [arr["x_mean"], arr["x_std"]]
    if "y_mean_node" in arr:
        arrays.extend([arr["y_mean_node"], arr["y_std_node"]])
        if "y_mean_edge" in arr:
            arrays.extend([arr["y_mean_edge"], arr["y_std_edge"]])
    elif "y_mean" in arr:
        arrays.extend([arr["y_mean"], arr["y_std"]])
    if "edge_mean" in arr:
        arrays.extend([arr["edge_mean"], arr["edge_std"]])
    for a in arrays:
        md5.update(a.tobytes())
    npz_hash = md5.hexdigest()
    if npz_hash != ckpt_hash:
        raise RuntimeError(
            "Normalization statistics hash mismatch. Regenerate or supply matching normalization files."
        )
    norm_md5 = ckpt_hash
    model_layers = len(getattr(model, "layers", []))
    model_hidden = getattr(getattr(model, "layers", [None])[0], "out_channels", None)

    if os.path.exists(args.test_pkl):
        with open(args.test_pkl, "rb") as f:
            test_res = pickle.load(f)
        metrics, err_arr, err_times, dbg = validate_surrogate(
            model,
            edge_index,
            edge_attr,
            wn,
            test_res,
            device,
            run_name,
            torch.tensor(node_types, dtype=torch.long, device=device),
            torch.tensor(edge_types, dtype=torch.long, device=device),
            debug=args.debug,
        )
        if args.debug:
            dbg["edge_attr_stats"] = edge_stats if edge_attr is not None else {}
            dbg["ckpt_meta"] = getattr(model, "_ckpt_meta", {})
            debug_path = RUNS_DIR / "debug_validation_summary.json"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            if "sample_df" in dbg:
                sample_path = RUNS_DIR / "debug_samples.csv"
                dbg["sample_df"].to_csv(sample_path, index=False)
                dbg["sample_df"] = sample_path.name
            def _to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            with open(debug_path, "w") as f:
                json.dump(dbg, f, indent=2, default=_to_serializable)
            print(f"[DEBUG] summary saved to {debug_path}")
        pd.DataFrame([metrics]).to_csv(
            os.path.join(DATA_DIR, "surrogate_validation.csv"), index=False
        )
        pressure_error_heatmap(
            err_arr,
            err_times,
            wn.node_name_list,
            args.inp,
            run_name,
        )
        if args.rollout_eval:
            rmse_p = rollout_surrogate(
                model,
                edge_index,
                edge_attr,
                wn,
                test_res,
                device,
                args.rollout_steps,
                torch.tensor(node_types, dtype=torch.long, device=device),
                torch.tensor(edge_types, dtype=torch.long, device=device),
            )
            run_dir = RUNS_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            rollout_df = pd.DataFrame(
                {
                    "t": np.arange(1, len(rmse_p) + 1),
                    "pressure_rmse": rmse_p,
                }
            )
            rollout_df.to_csv(run_dir / "rollout_rmse.csv", index=False)
            plt.figure()
            plt.plot(rollout_df["t"], rollout_df["pressure_rmse"], label="pressure_rmse")
            plt.xlabel("t")
            plt.ylabel("RMSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(run_dir / "rollout_rmse.png")
            plt.close()
    else:
        print(f"{args.test_pkl} not found. Skipping surrogate validation.")

    # Re-enable gradient calculations for MPC optimization
    torch.set_grad_enabled(True)
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
        0.0,
        args.feedback_interval,
        w_p=args.w_p,
        w_c=0.0,
        w_e=args.w_e,
    )

    # Export an animated view of network pressures and pump speeds
    try:
        animate_mpc_network(mpc_df, pump_names, run_name, inp_path=args.inp)
    except Exception as exc:
        print(f"[WARN] animate_mpc_network failed: {exc}")

    heur_df = run_heuristic_baseline(
        wntr.network.WaterNetworkModel(args.inp), pump_names
    )
    all_on_df = run_all_pumps_on(
        wntr.network.WaterNetworkModel(args.inp), pump_names
    )

    aggregate_and_plot({"mpc": mpc_df, "heuristic": heur_df, "all_on": all_on_df}, run_name, args.Pmin)

    cfg_extra = {
        "norm_stats_md5": norm_md5,
        "model_layers": model_layers,
        "model_hidden_dim": model_hidden,
    }
    save_config(
        REPO_ROOT / "logs" / f"config_validation_{run_name}.yaml",
        vars(args),
        cfg_extra,
    )


if __name__ == "__main__":
    main()
