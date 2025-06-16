"""Visualization utilities for GNN surrogate training and MPC runs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import epyt

# Resolve repository root so this module works when imported from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "plots"


def _to_numpy(seq: Sequence[float]) -> np.ndarray:
    """Convert sequence to NumPy array."""
    return np.asarray(seq, dtype=float)


def predicted_vs_actual_scatter(
    true_pressure: Sequence[float],
    pred_pressure: Sequence[float],
    true_chlorine: Sequence[float],
    pred_chlorine: Sequence[float],
    run_name: str,
    plots_dir: Path | None = None,
    return_fig: bool = False,
):
    """Scatter plots comparing surrogate predictions with EPANET results."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    tp = _to_numpy(true_pressure)
    pp = _to_numpy(pred_pressure)
    tc = _to_numpy(true_chlorine)
    pc = _to_numpy(pred_chlorine)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Pressure scatter plot -------------------------------------------------
    axes[0].scatter(tp, pp, label="Pressure", color="tab:blue", alpha=0.7)
    min_p, max_p = tp.min(), tp.max()
    axes[0].plot([min_p, max_p], [min_p, max_p], "k--", lw=1)
    axes[0].set_xlabel("Actual Pressure (m)")
    axes[0].set_ylabel("Predicted Pressure (m)")
    axes[0].set_title("Pressure")

    # --- Chlorine scatter plot -------------------------------------------------
    axes[1].scatter(tc, pc, label="Chlorine", color="tab:orange", alpha=0.7)
    min_c, max_c = tc.min(), tc.max()
    axes[1].plot([min_c, max_c], [min_c, max_c], "k--", lw=1)
    axes[1].set_xlabel("Actual Chlorine (mg/L)")
    axes[1].set_ylabel("Predicted Chlorine (mg/L)")
    axes[1].set_title("Chlorine")

    fig.suptitle("Surrogate Model Prediction Accuracy for Pressure and Chlorine")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    fig.savefig(plots_dir / f"pred_vs_actual_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_mpc_time_series(
    df,  # pandas.DataFrame
    Pmin: float,
    Cmin: float,
    run_name: str,
    plots_dir: Path | None = None,
    return_fig: bool = False,
):
    """Time series of minimum pressure/chlorine and pump actions."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # -- Minimum pressure -------------------------------------------------------
    axes[0].plot(df["time"], df["min_pressure"], label="Min Pressure", color="tab:blue")
    axes[0].axhline(Pmin, color="red", linestyle="--", label="P_min")
    axes[0].set_ylabel("Pressure (m)")
    axes[0].set_title("Operational Performance Under MPC Control")
    axes[0].legend()

    # -- Minimum chlorine -------------------------------------------------------
    axes[1].plot(df["time"], df["min_chlorine"], label="Min Chlorine", color="tab:orange")
    axes[1].axhline(Cmin, color="red", linestyle="--", label="C_min")
    axes[1].set_ylabel("Chlorine (mg/L)")
    axes[1].legend()

    # -- Pump controls ----------------------------------------------------------
    controls = np.stack(df["controls"].to_list())
    axes[2].plot(df["time"], controls)
    axes[2].set_ylabel("Pump Speed (%)")
    axes[2].set_xlabel("Time (hours)")

    fig.tight_layout()
    fig.savefig(plots_dir / f"mpc_timeseries_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def energy_pressure_tradeoff(
    strategy: Sequence[str],
    energy: Sequence[float],
    violations: Sequence[int],
    run_name: str,
    plots_dir: Path | None = None,
    return_fig: bool = False,
):
    """Bar chart comparing energy usage and pressure violations."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(strategy))
    width = 0.4

    bar1 = ax1.bar(x - width / 2, energy, width, color="tab:blue", label="Energy")
    ax1.set_ylabel("Energy Consumption (kWh)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width / 2, violations, width, color="tab:orange", label="Violations")
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
    plots_dir: Path | None = None,
    return_fig: bool = False,
):
    """Heatmaps showing pressure prediction errors across nodes and time."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -- Heatmap over time ------------------------------------------------------
    im = axes[0].imshow(errors, aspect="auto", cmap="magma")
    axes[0].set_xlabel("Node")
    axes[0].set_ylabel("Time (h)")
    axes[0].set_title("Heatmap of Pressure Prediction Errors")
    axes[0].set_yticks(np.arange(len(times)))
    axes[0].set_yticklabels(times)
    axes[0].set_xticks(np.arange(len(node_names)))
    axes[0].set_xticklabels(node_names, rotation=90, fontsize=6)
    fig.colorbar(im, ax=axes[0], label="Error (m)")

    # -- Geographic error map ---------------------------------------------------
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


def plot_convergence_curve(
    costs: Sequence[float],
    run_name: str,
    plots_dir: Path | None = None,
    return_fig: bool = False,
):
    """Plot optimisation cost over iterations."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(costs)), costs, marker="o")
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Cost Function Value")
    ax.set_title("Convergence of Gradient-Based MPC Optimization")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plots_dir / f"mpc_convergence_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None

