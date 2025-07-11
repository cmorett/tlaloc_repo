import argparse
import time
from typing import Dict, List, Optional, Sequence
import os
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import inspect
from torch_geometric.nn import GCNConv, GATConv
from sklearn.preprocessing import MinMaxScaler
# Import ``HydroConv`` from the training module located in the same
# directory.  Using a plain module import keeps the file executable both when
# called directly (``python scripts/mpc_control.py``) and when loaded from
# other scripts like ``experiments_validation.py`` where ``sys.path`` points to
# the ``scripts`` folder.
# ``mpc_control.py`` may be executed directly or imported as part of the
# ``scripts`` package.  Support both usages by attempting a relative import
# first and falling back to an absolute one when the module is run as a script.
try:
    from .train_gnn import (
        HydroConv,
        RecurrentGNNSurrogate,
        MultiTaskGNNSurrogate,
        build_edge_attr,
        build_edge_type,
        build_node_type,
    )  # type: ignore
except ImportError:  # pragma: no cover - executed when run as a script
    from train_gnn import (
        HydroConv,
        RecurrentGNNSurrogate,
        MultiTaskGNNSurrogate,
        build_edge_attr,
        build_edge_type,
        build_node_type,
    )
import wntr
from wntr.metrics.economic import pump_energy


# Resolve the repository root so files are written relative to the project
# instead of the current working directory.  This avoids permission errors
# when the script is executed from another location.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
PLOTS_DIR = REPO_ROOT / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_mpc_time_series(
    df,  # pandas.DataFrame
    Pmin: float,
    Cmin: float,
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Time series of minimum pressure/chlorine and pump actions."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(df["time"], df["min_pressure"], label="Min Pressure", color="tab:blue")
    axes[0].axhline(Pmin, color="red", linestyle="--", label="P_min")
    axes[0].set_ylabel("Pressure (m)")
    axes[0].set_title("Operational Performance Under MPC Control")
    axes[0].legend()

    axes[1].plot(df["time"], df["min_chlorine"], label="Min Chlorine", color="tab:orange")
    axes[1].axhline(Cmin, color="red", linestyle="--", label="C_min")
    axes[1].set_ylabel("Chlorine (mg/L)")
    axes[1].legend()

    controls = np.stack(df["controls"].to_list())
    axes[2].plot(df["time"], controls)
    axes[2].set_ylabel("Pump Speed (%)")
    axes[2].set_xlabel("Time (hours)")

    fig.tight_layout()
    fig.savefig(plots_dir / f"mpc_timeseries_{run_name}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_convergence_curve(
    costs: Sequence[float],
    run_name: str,
    plots_dir: Optional[Path] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
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


class GNNSurrogate(torch.nn.Module):
    """Flexible GCN used by the MPC controller."""

    def __init__(self, conv_layers: List[nn.Module], fc_out: Optional[nn.Linear] = None):
        super().__init__()
        self.layers = nn.ModuleList(conv_layers)
        self.fc_out = fc_out

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_types: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, conv in enumerate(self.layers):
            if hasattr(conv, "edge_mlps"):
                x = conv(x, edge_index, edge_attr, node_types, edge_types)
            else:
                x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        if self.fc_out is not None:
            x = self.fc_out(x)
        return x


def load_network(
    inp_file: str,
    return_edge_attr: bool = False,
    return_features: bool = False,
):
    """Load EPANET network and build edge index for PyG.

    Parameters
    ----------
    return_edge_attr : bool, optional
        If ``True`` also return the normalized edge attributes.
    return_features : bool, optional
        If ``True`` also return a static node feature tensor used for fast MPC
        simulation.
    """
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.quality.parameter = "CHEMICAL"
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600

    node_to_index = {n: i for i, n in enumerate(wn.node_name_list)}
    edges = []
    attrs = []
    etypes = []
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_to_index[link.start_node.name]
        j = node_to_index[link.end_node.name]
        edges.append([i, j])
        edges.append([j, i])
        length = getattr(link, "length", 0.0) or 0.0
        diam = getattr(link, "diameter", 0.0) or 0.0
        rough = getattr(link, "roughness", 0.0) or 0.0
        attrs.append([length, diam, rough])
        attrs.append([length, diam, rough])
        if link_name in wn.pipe_name_list:
            t = 0
        elif link_name in wn.pump_name_list:
            t = 1
        elif link_name in wn.valve_name_list:
            t = 2
        else:
            t = 0
        etypes.extend([t, t])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_types = build_node_type(wn)
    edge_types = torch.tensor(etypes, dtype=torch.long)

    static_feats = None
    if return_features:
        num_nodes = len(wn.node_name_list)
        num_pumps = len(wn.pump_name_list)
        static_feats = torch.zeros(num_nodes, 4 + num_pumps, dtype=torch.float32)
        for name, idx in node_to_index.items():
            node = wn.get_node(name)
            if name in wn.junction_name_list:
                demand = node.demand_timeseries_list[0].base_value
            else:
                demand = 0.0
            if name in wn.junction_name_list or name in wn.tank_name_list:
                elev = node.elevation
            elif name in wn.reservoir_name_list:
                elev = node.base_head
            else:
                elev = node.head
            static_feats[idx, 0] = float(demand)
            static_feats[idx, 3] = float(elev or 0.0)

    if return_edge_attr:
        edge_attr = build_edge_attr(wn, edge_index.numpy())
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        if return_features:
            return (
                wn,
                node_to_index,
                wn.pump_name_list,
                edge_index,
                edge_attr,
                node_types,
                edge_types,
                static_feats,
            )
        return (
            wn,
            node_to_index,
            wn.pump_name_list,
            edge_index,
            edge_attr,
            node_types,
            edge_types,
        )

    if return_features:
        return (
            wn,
            node_to_index,
            wn.pump_name_list,
            edge_index,
            node_types,
            edge_types,
            static_feats,
        )

    return wn, node_to_index, wn.pump_name_list, edge_index, node_types, edge_types



def load_surrogate_model(
    device: torch.device,
    path: Optional[str] = None,
    use_jit: bool = True,
) -> GNNSurrogate:
    """Load trained GNN surrogate weights.

    Parameters
    ----------
    device : torch.device
        Device to map the model to.
    path : Optional[str], optional
        Location of the saved state dict.  If ``None`` the newest ``.pth`` file
        in the ``models`` directory is loaded.

    Returns
    -------
    GNNSurrogate
        Loaded surrogate model set to eval mode.
    """
    # Determine which checkpoint to load.  If ``path`` is ``None`` look for the
    # most recently modified ``.pth`` file inside ``models``.
    if path is None:
        models_dir = REPO_ROOT / "models"
        candidates = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No model checkpoints found in {models_dir}")
        full_path = candidates[0]
    else:
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = REPO_ROOT / full_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"{full_path} not found. Run train_gnn.py to generate the surrogate weights."
        )
    state = torch.load(str(full_path), map_location=device)

    # Support both the current ``layers.X`` style parameter names as well as
    # older checkpoints that used ``conv1``/``conv2``.  If the latter is
    # detected, rename the keys so ``load_state_dict`` can succeed.
    if any(k.startswith("conv1") for k in state) and not any(k.startswith("layers.0") for k in state):
        renamed = {}
        for k, v in state.items():
            if k.startswith("conv1."):
                renamed["layers.0." + k.split(".", 1)[1]] = v
            elif k.startswith("conv2."):
                renamed["layers.1." + k.split(".", 1)[1]] = v
            else:
                renamed[k] = v
        state = renamed
    elif any(k.startswith("encoder.convs") for k in state):
        # Models trained with ``EnhancedGNNEncoder`` store convolution weights
        # under ``encoder.convs.X`` and the final fully connected layer under
        # ``encoder.fc_out``.  Convert these to ``layers.X`` / ``fc_out`` and
        # duplicate the first/last layer under ``conv1``/``conv2`` for backwards
        # compatibility.
        renamed = {}
        indices = sorted({int(k.split(".")[2]) for k in state if k.startswith("encoder.convs")})
        last_idx = max(indices)
        for k, v in state.items():
            if k.startswith("encoder.convs"):
                parts = k.split(".")
                idx = int(parts[2])
                rest = ".".join(parts[3:])
                if rest.startswith("lin.0."):
                    rest = "lin." + rest.split(".", 2)[2]
                base_key = f"layers.{idx}.{rest}"
                renamed[base_key] = v
                if idx == 0:
                    renamed[f"conv1.{rest}"] = v
                if idx == last_idx:
                    renamed[f"conv2.{rest}"] = v
            elif k.startswith("encoder.fc_out"):
                rest = k.split(".", 2)[2]
                renamed[f"fc_out.{rest}"] = v
            else:
                renamed[k] = v
        state = renamed
    elif any(k.startswith("convs") for k in state):
        renamed = {}
        indices = sorted({int(k.split(".")[1]) for k in state if k.startswith("convs")})
        last_idx = max(indices)
        for k, v in state.items():
            if not k.startswith("convs"):
                continue
            parts = k.split(".")
            idx = int(parts[1])
            rest = ".".join(parts[2:])
            base_key = f"layers.{idx}.{rest}"
            renamed[base_key] = v
            if idx == 0:
                renamed[f"conv1.{rest}"] = v
            if idx == last_idx:
                renamed[f"conv2.{rest}"] = v
        state = renamed

    layer_keys = [
        k
        for k in state
        if k.startswith("layers.")
        and (k.endswith("weight") or k.endswith("lin.weight"))
    ]
    fc_layer = None
    if layer_keys:
        indices = sorted({int(k.split(".")[1]) for k in layer_keys})
        conv_layers = []
        for i in indices:
            w_key = (
                f"layers.{i}.weight"
                if f"layers.{i}.weight" in state
                else f"layers.{i}.lin.weight"
            )
            w = state[w_key]
            in_dim = w.shape[1]
            out_dim = w.shape[0]

            edge_key = None
            if f"layers.{i}.edge_mlp.0.weight" in state:
                edge_key = f"layers.{i}.edge_mlp.0.weight"
            elif f"layers.{i}.edge_mlps.0.0.weight" in state:
                edge_key = f"layers.{i}.edge_mlps.0.0.weight"

            if edge_key is not None:
                e_dim = state[edge_key].shape[1]

                node_keys = [
                    k
                    for k in state
                    if k.startswith(f"layers.{i}.lin.") and k.endswith("weight")
                ]
                if node_keys:
                    node_indices = []
                    for k in node_keys:
                        parts = k.split(".")
                        if len(parts) > 3 and parts[3].isdigit():
                            node_indices.append(int(parts[3]))
                    n_node_types = max(node_indices) + 1 if node_indices else 1
                else:
                    n_node_types = 1

                edge_keys = [
                    k
                    for k in state
                    if k.startswith(f"layers.{i}.edge_mlps") and k.endswith("weight")
                ]
                if edge_keys:
                    edge_indices = []
                    for k in edge_keys:
                        parts = k.split(".")
                        if len(parts) > 3 and parts[3].isdigit():
                            edge_indices.append(int(parts[3]))
                    n_edge_types = max(edge_indices) + 1 if edge_indices else 1
                else:
                    n_edge_types = 1

                conv_layers.append(
                    HydroConv(
                        in_dim,
                        out_dim,
                        e_dim,
                        num_node_types=n_node_types,
                        num_edge_types=n_edge_types,
                    )
                )
            else:
                conv_layers.append(GCNConv(in_dim, out_dim))
    else:
        weight_key = "conv1.weight" if "conv1.weight" in state else "conv1.lin.weight"
        hidden_key = weight_key
        out_key = "conv2.weight" if "conv2.weight" in state else "conv2.lin.weight"
        in_dim = state[weight_key].shape[1]
        hidden_dim = state[hidden_key].shape[0]
        out_dim = state[out_key].shape[0]
        conv_layers = [GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, out_dim)]

    if "fc_out.weight" in state:
        out_dim, hidden_dim = state["fc_out.weight"].shape
        fc_layer = nn.Linear(hidden_dim, out_dim)

    has_rnn = any(k.startswith("rnn.") for k in state)
    multitask = "node_decoder.weight" in state

    # Fail early if the checkpoint contains invalid values which would otherwise
    # produce NaN predictions during MPC optimisation.
    for k, v in state.items():
        if torch.isnan(v).any():
            raise ValueError(f"NaN detected in model weights ({k}) – re-train the surrogate.")

    edge_dim = None
    if isinstance(conv_layers[0], HydroConv):
        # ``HydroConv`` renamed ``edge_mlp`` to ``edge_mlps`` in recent versions
        mlp_list = getattr(conv_layers[0], "edge_mlp", None)
        if mlp_list is None:
            mlp_list = conv_layers[0].edge_mlps
        first_layer = mlp_list[0]
        if isinstance(first_layer, nn.Sequential):
            edge_dim = first_layer[0].in_features
        else:
            edge_dim = first_layer.in_features

    if multitask:
        node_out_dim, rnn_hidden_dim = state["node_decoder.weight"].shape
        edge_out_dim = state["edge_decoder.weight"].shape[0]
        model = MultiTaskGNNSurrogate(
            in_channels=conv_layers[0].in_channels,
            hidden_channels=conv_layers[0].out_channels,
            edge_dim=edge_dim if edge_dim is not None else 0,
            node_output_dim=node_out_dim,
            edge_output_dim=edge_out_dim,
            num_layers=len(conv_layers),
            use_attention=False,
            gat_heads=1,
            dropout=0.0,
            residual=False,
            rnn_hidden_dim=rnn_hidden_dim,
            num_node_types=getattr(conv_layers[0], "num_node_types", 1),
            num_edge_types=getattr(conv_layers[0], "num_edge_types", 1),
        ).to(device)
    elif has_rnn:
        out_dim, rnn_hidden_dim = state["decoder.weight"].shape
        model = RecurrentGNNSurrogate(
            in_channels=conv_layers[0].in_channels,
            hidden_channels=conv_layers[0].out_channels,
            edge_dim=edge_dim if edge_dim is not None else 0,
            output_dim=out_dim,
            num_layers=len(conv_layers),
            use_attention=False,
            gat_heads=1,
            dropout=0.0,
            residual=False,
            rnn_hidden_dim=rnn_hidden_dim,
            num_node_types=getattr(conv_layers[0], "num_node_types", 1),
            num_edge_types=getattr(conv_layers[0], "num_edge_types", 1),
        ).to(device)
    else:
        model = GNNSurrogate(conv_layers, fc_out=fc_layer).to(device)

    model.load_state_dict(state, strict=False)

    # store expected edge attribute dimension for input checks
    model.edge_dim = edge_dim if edge_dim is not None else 0

    norm_path = str(full_path.with_suffix("")) + "_norm.npz"
    if os.path.exists(norm_path):
        arr = np.load(norm_path)
        # Moved normalization constants to GPU to avoid device transfer
        model.x_mean = torch.tensor(arr["x_mean"], dtype=torch.float32, device=device)
        model.x_std = torch.tensor(arr["x_std"], dtype=torch.float32, device=device)
        if "y_mean" in arr:
            model.y_mean = torch.tensor(arr["y_mean"], dtype=torch.float32, device=device)
            model.y_std = torch.tensor(arr["y_std"], dtype=torch.float32, device=device)
        elif "y_mean_node" in arr:
            model.y_mean = torch.tensor(arr["y_mean_node"], dtype=torch.float32, device=device)
            model.y_std = torch.tensor(arr["y_std_node"], dtype=torch.float32, device=device)
        else:
            model.y_mean = model.y_std = None

        model.y_mean_energy = None
        model.y_std_energy = None
    else:
        model.x_mean = model.x_std = model.y_mean = model.y_std = None

    model.eval()

    if use_jit:
        try:
            source = inspect.getsource(getattr(model, "encoder", model).__class__)
        except (OSError, TypeError):  # pragma: no cover - source unavailable
            source = ""

        if "GATConv" in source or any(isinstance(m, GATConv) for m in model.modules()):
            warnings.warn(
                "GATConv layers are not TorchScript compatible – skipping JIT compilation"
            )
        else:
            try:
                scripted = torch.jit.script(model)
                model = torch.jit.optimize_for_inference(scripted)
            except Exception as e:  # pragma: no cover - optional optimisation
                warnings.warn(f"TorchScript compilation failed: {e}")

    return model


def compute_demand_vectors(
    wn: wntr.network.WaterNetworkModel,
    node_to_index: Dict[str, int],
    base_demands: Dict[str, float],
    start_hour: int,
    horizon: int,
) -> torch.Tensor:
    """Compute per-node demand vectors for each hour of the horizon."""
    num_nodes = len(node_to_index)
    out = torch.zeros(horizon, num_nodes, dtype=torch.float32)
    for t in range(horizon):
        t_sec = (start_hour + t) * 3600
        for j in wn.junction_name_list:
            pat_name = wn.get_node(j).demand_timeseries_list[0].pattern_name
            mult = 1.0
            if pat_name:
                mult = wn.get_pattern(pat_name).at(t_sec)
            out[t, node_to_index[j]] = base_demands[j] * mult
    return out


def prepare_node_features(
    template: torch.Tensor,
    pressures: torch.Tensor,
    chlorine: torch.Tensor,
    pump_controls: torch.Tensor,
    model: GNNSurrogate,
    demands: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assemble node features using precomputed static attributes.

    Parameters
    ----------
    demands : torch.Tensor, optional
        Per-node demand values for the current time step. When ``None`` the
        base demand stored in ``template`` is used.
    """
    num_nodes = template.size(0)
    num_pumps = pump_controls.size(-1)
    pump_controls = pump_controls.to(dtype=torch.float32, device=template.device)

    if pressures.dim() == 2:
        batch_size = pressures.size(0)
        feats = template.expand(batch_size, num_nodes, template.size(1)).clone()
        if demands is not None:
            feats[:, :, 0] = demands
        feats[:, :, 1] = pressures
        feats[:, :, 2] = chlorine
        feats[:, :, 4 : 4 + num_pumps] = pump_controls.view(batch_size, 1, -1).expand(batch_size, num_nodes, num_pumps)
        in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", None)
        if in_dim is not None:
            feats = feats[:, :, :in_dim]
        if getattr(model, "x_mean", None) is not None:
            feats = (feats - model.x_mean.view(1, 1, -1)) / model.x_std.view(1, 1, -1)
        return feats.view(batch_size * num_nodes, -1)

    feats = template.clone()
    if demands is not None:
        feats[:, 0] = demands
    feats[:, 1] = pressures
    feats[:, 2] = chlorine
    feats[:, 4 : 4 + num_pumps] = pump_controls.view(1, -1).expand(num_nodes, num_pumps)
    in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", None)
    if in_dim is not None:
        feats = feats[:, :in_dim]
    if getattr(model, "x_mean", None) is not None:
        feats = (feats - model.x_mean) / model.x_std
    return feats


def compute_mpc_cost(
    u: torch.Tensor,
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    feature_template: torch.Tensor,
    pressures: torch.Tensor,
    chlorine: torch.Tensor,
    horizon: int,
    device: torch.device,
    Pmin: float,
    Cmin: float,
    demands: Optional[torch.Tensor] = None,
    pump_info: Optional[list[tuple[int, int, int]]] = None,
    return_energy: bool = False,
    init_tank_levels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the MPC cost for a sequence of pump controls.

    The cost combines pressure and chlorine constraint violations, pump
    energy use and a smoothness term on control differences. ``demands`` can be
    used to provide per-node demand values for each step which keeps the
    surrogate inputs consistent with training.
    Violations are penalized cubically to strongly discourage operating below
    the specified minimum thresholds.
    """
    cur_p = pressures.to(device)
    cur_c = chlorine.to(device)

    if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
        if init_tank_levels is None:
            init_press = cur_p[model.tank_indices].unsqueeze(0)
            init_levels = init_press * model.tank_areas
        else:
            init_levels = init_tank_levels.to(device)
        model.reset_tank_levels(init_levels)
    total_cost = torch.tensor(0.0, device=device)
    smoothness_penalty = torch.tensor(0.0, device=device)
    energy_first = torch.tensor(0.0, device=device) if return_energy else None

    for t in range(horizon):
        d = demands[t] if demands is not None else None
        ctrl = torch.clamp(u[t], 0.0, 1.0)
        x = prepare_node_features(feature_template, cur_p, cur_c, ctrl, model, d)
        if hasattr(model, "rnn"):
            seq_in = x.unsqueeze(0).unsqueeze(0)
            out = model(seq_in, edge_index, edge_attr, node_types, edge_types)
            if isinstance(out, dict):
                pred = out.get("node_outputs")[0, 0]
                flows = out.get("edge_outputs")[0, 0].squeeze(-1)
            else:
                pred = out[0, 0]
                flows = None
        else:
            out = model(x, edge_index, edge_attr, node_types, edge_types)
            if isinstance(out, dict):
                pred = out.get("node_outputs")
                flows = out.get("edge_outputs").squeeze(-1)
            else:
                pred = out
                flows = None

        if getattr(model, "y_mean", None) is not None:
            pred = pred * model.y_std + model.y_mean
        assert not torch.isnan(pred).any(), "NaN prediction"
        pred_p = pred[:, 0]
        pred_c = pred[:, 1]

        # ------------------------------------------------------------------
        # Cost terms
        # ------------------------------------------------------------------
        # We enforce pressure and chlorine constraints using a cubic penalty
        # on the amount by which the prediction falls below the minimum
        # thresholds.  Cubic growth penalizes larger violations more
        # aggressively than a simple quadratic term.

        w_p, w_c, w_e, w_s = 10.0, 5.0, 1.0, 0.01

        Pmin_safe = Pmin + 3.0
        Cmin_safe = Cmin + 0.05

        psf = torch.clamp(Pmin_safe - pred_p, min=0.0)
        csf = torch.clamp(Cmin_safe - pred_c, min=0.0)
        pressure_penalty = torch.sum(psf ** 3)
        chlorine_penalty = torch.sum(csf ** 3)

        if flows is not None and pump_info is not None:
            head = pred_p + feature_template[:, 3]
            energy_term = torch.tensor(0.0, device=device)
            eff = wn.options.energy.global_efficiency / 100.0
            for idx, s_idx, e_idx in pump_info:
                f = flows[idx]
                hl = head[e_idx] - head[s_idx]
                e = 1000.0 * 9.81 * hl * f / eff * 3600.0
                energy_term = energy_term + e
                if t == 0 and return_energy:
                    energy_first = energy_first + e
        else:
            energy_term = torch.sum(ctrl ** 2)

        step_cost = (
            w_p * pressure_penalty
            + w_c * chlorine_penalty
            + w_e * energy_term
        )

        total_cost = total_cost + step_cost

        if t > 0:
            # small penalty on rapid pump switching to produce smoother
            # control sequences
            prev_ctrl = torch.clamp(u[t - 1], 0.0, 1.0)
            smoothness_penalty = smoothness_penalty + torch.sum((ctrl - prev_ctrl) ** 2)

        # update dictionaries for next step
        cur_p = pred_p
        cur_c = pred_c

    total_cost = total_cost + w_s * smoothness_penalty

    if return_energy:
        return total_cost, energy_first
    return total_cost, None


def run_mpc_step(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    feature_template: torch.Tensor,
    pressures: torch.Tensor,
    chlorine: torch.Tensor,
    horizon: int,
    iterations: int,
    device: torch.device,
    Pmin: float,
    Cmin: float,
    demands: Optional[torch.Tensor] = None,
    u_warm: Optional[torch.Tensor] = None,
    pump_info: Optional[list[tuple[int, int, int]]] = None,
    profile: bool = False,
) -> tuple[torch.Tensor, List[float], float]:
    """Optimize pump controls for one hour using gradient-based MPC.

    The optimization is performed in two phases: a short warm start with
    Adam followed by refinement using L-BFGS. ``iterations`` controls the total
    number of optimization steps, split approximately 20/80 between the two
    phases. ``u`` is clamped to ``[0, 1]`` after each update to enforce valid
    pump settings.

    Parameters
    ----------
    demands : torch.Tensor, optional
        Horizon x num_nodes demand values used when assembling node features.
    u_warm : torch.Tensor, optional
        Previous control sequence to warm start the optimization.
    """
    num_pumps = feature_template.size(1) - 4
    cost_history: List[float] = []
    start_time = time.time() if profile else None
    pressures = pressures.to(device)
    chlorine = chlorine.to(device)
    init_levels = None
    if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
        init_press = pressures[model.tank_indices].unsqueeze(0)
        init_levels = init_press * model.tank_areas
    if u_warm is not None and u_warm.shape[0] >= horizon:
        init = torch.cat([u_warm[1:horizon], u_warm[horizon - 1 : horizon]], dim=0)
    else:
        init = torch.ones(horizon, num_pumps, device=device)
    u = init.clone().detach().requires_grad_(True)

    # ``torch.nn.LSTM`` in evaluation mode does not support the backward
    # pass when using cuDNN.  Some surrogates (``RecurrentGNNSurrogate`` or
    # ``MultiTaskGNNSurrogate``) therefore need to be temporarily switched to
    # training mode during optimisation.  The models loaded by
    # ``load_surrogate_model`` disable dropout, so toggling the mode has no
    # effect on the forward computation.
    prev_training = model.training
    if hasattr(model, "rnn") and not model.training:
        model.train()

    # --- Phase 1: warm start with Adam -------------------------------------
    adam_steps = max(1, min(10, iterations // 5))
    adam_opt = torch.optim.Adam([u], lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_opt, "min", patience=3, factor=0.5)

    for _ in range(adam_steps):
        adam_opt.zero_grad()
        cost, _ = compute_mpc_cost(
            u,
            wn,
            model,
            edge_index,
            edge_attr,
            node_types,
            edge_types,
            feature_template,
            pressures,
            chlorine,
            horizon,
            device,
            Pmin,
            Cmin,
            demands,
            pump_info,
            False,
            init_levels,
        )
        cost.backward()
        adam_opt.step()
        with torch.no_grad():
            u.data.clamp_(0.0, 1.0)
        scheduler.step(cost.item())
        cost_history.append(float(cost.item()))

    # --- Phase 2: L-BFGS refinement --------------------------------------
    lbfgs_steps = max(iterations - adam_steps, 1)
    lbfgs_opt = torch.optim.LBFGS([u], max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

    def closure():
        lbfgs_opt.zero_grad()
        c, _ = compute_mpc_cost(
            u,
            wn,
            model,
            edge_index,
            edge_attr,
            node_types,
            edge_types,
            feature_template,
            pressures,
            chlorine,
            horizon,
            device,
            Pmin,
            Cmin,
            demands,
            pump_info,
            False,
            init_levels,
        )
        c.backward()
        return c

    final_cost = lbfgs_opt.step(closure)
    cost_history.append(float(final_cost))
    _, energy_first = compute_mpc_cost(
        u,
        wn,
        model,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
        pressures,
        chlorine,
        1,
        device,
        Pmin,
        Cmin,
        demands[:1] if demands is not None else None,
        pump_info,
        True,
        init_levels,
    )
    with torch.no_grad():
        u.data.clamp_(0.0, 1.0)

    if hasattr(model, "rnn") and not prev_training:
        model.eval()

    if profile and start_time is not None:
        end_time = time.time()
        print(f"[profile] run_mpc_step: {end_time - start_time:.4f}s")

    return u.detach(), cost_history, float(energy_first.item()) if energy_first is not None else 0.0


def propagate_with_surrogate(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    feature_template: torch.Tensor,
    pressures: Dict[str, float],
    chlorine: Dict[str, float],
    control_seq: torch.Tensor,
    device: torch.device,
    demands: Optional[torch.Tensor] = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Propagate the network state using the surrogate model.

    The current state ``pressures``/``chlorine`` is advanced through the
    sequence of pump controls ``control_seq`` without running EPANET.  The
    function returns dictionaries for the next pressures and chlorine levels
    after applying the entire sequence.  ``pressures`` and ``chlorine`` can be
    either dictionaries for a single scenario or lists of dictionaries for
    batched evaluation:

    ``pressures = [dict(...), dict(...)]``
    ``chlorine = [dict(...), dict(...)]``
    """

    single = isinstance(pressures, dict)
    if single:
        cur_p = torch.tensor([pressures[n] for n in wn.node_name_list], device=device)
        cur_c = torch.tensor([chlorine[n] for n in wn.node_name_list], device=device)
        b_edge_index = edge_index
        b_edge_attr = edge_attr
        b_node_type = node_types
        b_edge_type = edge_types
        batch_size = 1
    else:
        batch_size = len(pressures)
        cur_p = torch.stack([
            torch.tensor([p[n] for n in wn.node_name_list], device=device)
            for p in pressures
        ])
        cur_c = torch.stack([
            torch.tensor([c[n] for n in wn.node_name_list], device=device)
            for c in chlorine
        ])
        num_nodes = feature_template.size(0)
        E = edge_index.size(1)
        b_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        b_edge_attr = edge_attr.repeat(batch_size, 1) if edge_attr is not None else None
        b_node_type = node_types.repeat(batch_size) if node_types is not None else None
        b_edge_type = edge_types.repeat(batch_size) if edge_types is not None else None

    if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
        if batch_size == 1:
            init_press = cur_p[model.tank_indices].unsqueeze(0)
        else:
            init_press = cur_p[:, model.tank_indices]
        init_levels = init_press * model.tank_areas
        model.reset_tank_levels(init_levels)

    with torch.no_grad():
        for t, u in enumerate(control_seq):
            if u.dim() == 1:
                u_in = u.view(1, -1).expand(batch_size, -1)
            else:
                u_in = u
            d = demands[t] if demands is not None else None
            x = prepare_node_features(
                feature_template, cur_p, cur_c, u_in, model, d
            )
            if hasattr(model, "rnn"):
                seq_in = x.view(batch_size, 1, feature_template.size(0), x.size(-1))
                pred = model(seq_in, b_edge_index, b_edge_attr, b_node_type, b_edge_type)
                if isinstance(pred, dict):
                    pred = pred.get("node_outputs")[:, 0]
            else:
                pred = model(x, b_edge_index, b_edge_attr, b_node_type, b_edge_type)
                if isinstance(pred, dict):
                    pred = pred.get("node_outputs")
            if getattr(model, "y_mean", None) is not None:
                pred = pred * model.y_std + model.y_mean
            assert not torch.isnan(pred).any(), "NaN prediction"
            pred = pred.view(batch_size, feature_template.size(0), -1)
            cur_p = pred[:, :, 0]
            cur_c = pred[:, :, 1]

    if single:
        out_p = {n: float(cur_p[0, i]) for i, n in enumerate(wn.node_name_list)}
        out_c = {n: float(cur_c[0, i]) for i, n in enumerate(wn.node_name_list)}
        return out_p, out_c

    out_ps = []
    out_cs = []
    for b in range(batch_size):
        out_ps.append({n: float(cur_p[b, i]) for i, n in enumerate(wn.node_name_list)})
        out_cs.append({n: float(cur_c[b, i]) for i, n in enumerate(wn.node_name_list)})
    return out_ps, out_cs



def simulate_closed_loop(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    feature_template: torch.Tensor,
    node_types: torch.Tensor,
    edge_types: torch.Tensor,
    horizon: int,
    iterations: int,
    node_to_index: Dict[str, int],
    pump_names: List[str],
    device: torch.device,
    Pmin: float,
    Cmin: float,
    feedback_interval: int = 24,
    run_name: str = "",
    profile: bool = False,
) -> pd.DataFrame:
    """Run 24-hour closed-loop MPC using the surrogate for fast updates.

    EPANET is invoked only every ``feedback_interval`` hours (default once per
    day) to obtain ground-truth measurements.  All intermediate steps update the
    pressures and chlorine levels using the GNN surrogate which allows the loop
    to run nearly instantly.
    """
    expected_in_dim = 4 + len(pump_names)
    in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", expected_in_dim)
    if in_dim < expected_in_dim:
        raise ValueError(
            "Loaded model was trained without pump controls - rerun train_gnn.py"
        )

    if edge_attr is not None and hasattr(model, "edge_dim"):
        if edge_attr.size(1) != model.edge_dim:
            raise ValueError(
                f"Edge attribute dimension mismatch: model expects {model.edge_dim}, "
                f"but received {edge_attr.size(1)}."
            )

    log = []
    pressure_violations = 0
    chlorine_violations = 0
    total_energy = 0.0
    pump_info = []
    node_idx = node_to_index
    ei_np = edge_index.cpu().numpy()
    for name in pump_names:
        pump = wn.get_link(name)
        s = node_idx[pump.start_node.name]
        e = node_idx[pump.end_node.name]
        idx = int(np.where((ei_np[0] == s) & (ei_np[1] == e))[0][0])
        pump_info.append((idx, s, e))
    # obtain hydraulic state at time zero
    wn.options.time.duration = 0
    wn.options.time.report_timestep = 0
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(str(TEMP_DIR / "temp"))
    p_arr = results.node["pressure"].iloc[0].to_numpy(dtype=np.float32)
    c_arr = results.node["quality"].iloc[0].to_numpy(dtype=np.float32)
    pressures = dict(zip(wn.node_name_list, p_arr))
    chlorine = dict(zip(wn.node_name_list, c_arr))
    # Keep state tensors on GPU to avoid repeated host transfers
    cur_p = (
        torch.from_numpy(p_arr)
        .pin_memory() if torch.cuda.is_available() else torch.from_numpy(p_arr)
    )
    cur_p = cur_p.to(device, non_blocking=True)
    cur_c = (
        torch.from_numpy(c_arr)
        .pin_memory() if torch.cuda.is_available() else torch.from_numpy(c_arr)
    )
    cur_c = cur_c.to(device, non_blocking=True)

    if hasattr(model, "reset_tank_levels") and hasattr(model, "tank_indices"):
        init_press = cur_p[model.tank_indices].unsqueeze(0)
        init_levels = init_press * model.tank_areas
        model.reset_tank_levels(init_levels)

    base_demands = {
        j: wn.get_node(j).demand_timeseries_list[0].base_value
        for j in wn.junction_name_list
    }
    prev_u: Optional[torch.Tensor] = None
    all_costs: List[float] = []

    for hour in range(24):
        start = time.time()
        demands = compute_demand_vectors(
            wn, node_to_index, base_demands, hour, horizon
        ).to(device)
        u_opt, costs, energy_first = run_mpc_step(
            wn,
            model,
            edge_index,
            edge_attr,
            node_types,
            edge_types,
            feature_template,
            cur_p,
            cur_c,
            horizon,
            iterations,
            device,
            Pmin,
            Cmin,
            demands,
            prev_u,
            pump_info,
            profile,
            )
        all_costs.extend(costs)
        prev_u = u_opt.detach()

        # apply control to network object for consistency
        first_controls = u_opt[0]
        for i, pump in enumerate(pump_names):
            link = wn.get_link(pump)
            if first_controls[i].item() < 0.5:
                link.initial_status = wntr.network.base.LinkStatus.Closed
            else:
                link.initial_status = wntr.network.base.LinkStatus.Open
                link.base_speed = float(first_controls[i].item())

        # update demands based on the diurnal pattern
        t = hour * 3600
        for j in wn.junction_name_list:
            junc = wn.get_node(j)
            mult = 1.0
            pat_name = junc.demand_timeseries_list[0].pattern_name
            if pat_name:
                mult = wn.get_pattern(pat_name).at(t)
            junc.demand_timeseries_list[0].base_value = base_demands[j] * mult

        if feedback_interval > 0 and hour % feedback_interval == 0:
            # Periodic ground truth synchronization using EPANET
            wn.options.time.start_clocktime = t
            wn.options.time.duration = 3600
            wn.options.time.report_timestep = 3600
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim(str(TEMP_DIR / "temp"))
            p_arr = results.node["pressure"].iloc[-1].to_numpy(dtype=np.float32)
            c_arr = results.node["quality"].iloc[-1].to_numpy(dtype=np.float32)
            pressures = dict(zip(wn.node_name_list, p_arr))
            chlorine = dict(zip(wn.node_name_list, c_arr))
            # Using non_blocking transfer for EPANET output
            cur_p = (
                torch.from_numpy(p_arr)
                .pin_memory() if torch.cuda.is_available() else torch.from_numpy(p_arr)
            )
            cur_p = cur_p.to(device, non_blocking=True)
            cur_c = (
                torch.from_numpy(c_arr)
                .pin_memory() if torch.cuda.is_available() else torch.from_numpy(c_arr)
            )
            cur_c = cur_c.to(device, non_blocking=True)
            energy_df = pump_energy(
                results.link["flowrate"][pump_names], results.node["head"], wn
            )
            energy = energy_df[pump_names].iloc[-1].sum()
            assert not np.isnan(energy), "NaN energy calculation"
            end = time.time()
        else:
            # Fast surrogate-based propagation
            pressures, chlorine = propagate_with_surrogate(
                wn,
                model,
                edge_index,
                edge_attr,
                node_types,
                edge_types,
                feature_template,
                pressures,
                chlorine,
                u_opt,
                device,
                demands,
            )
            cur_p = torch.tensor([pressures[n] for n in wn.node_name_list], dtype=torch.float32, device=device)
            cur_c = torch.tensor([chlorine[n] for n in wn.node_name_list], dtype=torch.float32, device=device)
            end = time.time()
            energy = energy_first
        min_p = max(
            min(pressures[n] for n in wn.junction_name_list + wn.tank_name_list),
            0.0,
        )
        min_c = max(
            min(chlorine[n] for n in wn.junction_name_list + wn.tank_name_list),
            0.0,
        )
        if min_p < Pmin:
            pressure_violations += 1
        if min_c < Cmin:
            chlorine_violations += 1
        total_energy += energy
        log.append(
            {
                "time": hour,
                "min_pressure": min_p,
                "min_chlorine": min_c,
                "energy": energy,
                "runtime_sec": end - start,
                "controls": first_controls.cpu().numpy().tolist(),
            }
        )
        print(
            f"Hour {hour}: minP={min_p:.2f}, minC={min_c:.3f}, energy={energy:.2f}, runtime={end-start:.2f}s"
        )
    df = pd.DataFrame(log)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "mpc_history.csv"), index=False)
    summary = {
        "pressure_violations": pressure_violations,
        "chlorine_violations": chlorine_violations,
        "total_energy": float(total_energy),
        "hours": len(log),
    }
    print(
        f"[MPC Summary] Pressure violations: {pressure_violations}/{len(log)}h"
    )
    print(
        f"[MPC Summary] Chlorine violations: {chlorine_violations}/{len(log)}h"
    )
    print(f"[MPC Summary] Total pump energy used: {total_energy:.2f} J")
    os.makedirs(REPO_ROOT / "logs", exist_ok=True)
    with open(REPO_ROOT / "logs" / "mpc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if run_name:
        plot_mpc_time_series(df, Pmin, Cmin, run_name)
        plot_convergence_curve(all_costs, run_name)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6, help="MPC horizon length")
    parser.add_argument(
        "--iterations", type=int, default=50, help="Gradient descent iterations"
    )
    parser.add_argument("--Pmin", type=float, default=20.0, help="Pressure threshold")
    parser.add_argument("--Cmin", type=float, default=0.2, help="Chlorine threshold")
    parser.add_argument(
        "--feedback-interval",
        type=int,
        default=24,
        help="Hours between EPANET synchronizations",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable TorchScript compilation of the surrogate",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print runtime of each MPC optimisation step",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_path = os.path.join(REPO_ROOT, "CTown.inp")
    (
        wn,
        node_to_index,
        pump_names,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
        feature_template,
    ) = load_network(inp_path, return_edge_attr=True, return_features=True)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    node_types = torch.tensor(node_types, dtype=torch.long, device=device)
    edge_types = torch.tensor(edge_types, dtype=torch.long, device=device)
    try:
        model = load_surrogate_model(device, use_jit=not args.no_jit)
    except FileNotFoundError as e:
        print(e)
        return

    expected_in_dim = 4 + len(pump_names)
    in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", expected_in_dim)
    if in_dim != expected_in_dim:
        print(
            f"Loaded surrogate expects {in_dim} input features "
            f"but the network requires {expected_in_dim}."
        )
        print(
            "The provided model was likely trained without pump control inputs.\n"
            "Re-train the surrogate using data generated with pump features."
        )
        return

    simulate_closed_loop(
        wn,
        model,
        edge_index,
        edge_attr,
        feature_template.to(device),
        node_types.to(device),
        edge_types.to(device),
        args.horizon,
        args.iterations,
        node_to_index,
        pump_names,
        device,
        args.Pmin,
        args.Cmin,
        args.feedback_interval,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        args.profile,
    )


if __name__ == "__main__":
    main()
