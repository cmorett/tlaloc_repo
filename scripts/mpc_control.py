import argparse
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import inspect
from collections import deque
from torch_geometric.nn import GCNConv, GATConv
from sklearn.preprocessing import MinMaxScaler
try:
    from .reproducibility import configure_seeds, save_config
except ImportError:  # pragma: no cover
    from reproducibility import configure_seeds, save_config
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
    )  # type: ignore
    from .feature_utils import (
        build_edge_attr,
        build_node_type,
        build_static_node_features,
        prepare_node_features,
    )
except ImportError:  # pragma: no cover - executed when run as a script
    from train_gnn import (
        HydroConv,
        RecurrentGNNSurrogate,
        MultiTaskGNNSurrogate,
    )
    from feature_utils import (
        build_edge_attr,
        build_node_type,
        build_static_node_features,
        prepare_node_features,
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

EPS = 1e-8
MAX_PUMP_SPEED = 1.8


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


def plot_network_state_epyt(
    pressures: Dict[str, float],
    pump_controls: Dict[str, float],
    run_name: str,
    timestep: int,
    plots_dir: Optional[Path] = None,
    inp_file: Optional[Path] = None,
) -> Path:
    """Visualise network pressures and pump actions using EPyT.

    Parameters
    ----------
    pressures : Dict[str, float]
        Mapping from node name to pressure value.
    pump_controls : Dict[str, float]
        Mapping from pump link name to control input (speed).
    run_name : str
        Identifier for the MPC run used in the output filename.
    timestep : int
        Hour index used in the output filename.
    plots_dir : Optional[Path]
        Directory where the figure will be stored. Defaults to ``plots/``.
    inp_file : Optional[Path]
        Explicit path to the EPANET ``.inp`` file.  Defaults to ``CTown.inp``
        in the repository root.
    """
    from epyt import epanet as _epanet  # local import to avoid heavy dependency at module import

    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    if inp_file is None:
        inp_file = REPO_ROOT / "CTown.inp"

    net = _epanet(str(inp_file))
    coords = net.getNodeCoordinates()
    node_names = net.getNodeNameID()
    xs = [coords["x"][i + 1] for i in range(len(node_names))]
    ys = [coords["y"][i + 1] for i in range(len(node_names))]
    vals = [pressures.get(n, 0.0) for n in node_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=vals, cmap="coolwarm", s=25)
    plt.colorbar(sc, ax=ax, label="Pressure (m)")

    link_names = net.getLinkNameID()
    for lname in link_names:
        start_idx, end_idx = net.getLinkNodesIndex(lname)
        x1, y1 = coords["x"][start_idx], coords["y"][start_idx]
        x2, y2 = coords["x"][end_idx], coords["y"][end_idx]
        width = 1.0
        color = "lightgray"
        if lname in pump_controls:
            width = 0.5 + pump_controls[lname]
            color = "tab:green"
        ax.plot([x1, x2], [y1, y2], linewidth=width, color=color)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out_path = plots_dir / f"mpc_network_state_{run_name}_t{timestep}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    net.closeNetwork()
    return out_path


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
        if link_name in wn.pump_name_list:
            attrs.append([length, diam, rough, 1.0])
            attrs.append([length, diam, rough, 0.0])
            t = 1
        else:
            attrs.append([length, diam, rough, 1.0])
            attrs.append([length, diam, rough, 1.0])
            if link_name in wn.pipe_name_list:
                t = 0
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
        static_feats = build_static_node_features(wn, len(wn.pump_name_list))

    if return_edge_attr:
        edge_attr = build_edge_attr(wn, edge_index.numpy()).astype(np.float32)
        edge_attr[:, 2] = np.log1p(edge_attr[:, 2])
        edge_attr = MinMaxScaler().fit_transform(edge_attr).astype(np.float32)
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
    norm_stats_path: Optional[str] = None,
    cfg_meta: Optional[Dict[str, Any]] = None,
    force_arch_mismatch: bool = False,
) -> GNNSurrogate:
    """Load trained GNN surrogate weights.

    Parameters
    ----------
    device : torch.device
        Device to map the model to.
    path : Optional[str], optional
        Location of the saved state dict.  If ``None`` the newest ``.pth`` file
        in the ``models`` directory is loaded.
    norm_stats_path : Optional[str], optional
        Explicit path to an ``.npz`` file containing normalization statistics.
        Overrides the default ``*_norm.npz`` lookup based on ``path``.

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
    checkpoint = torch.load(str(full_path), map_location=device, weights_only=False)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    raw_state = dict(state)  # preserve original keys for loading
    ckpt_meta = checkpoint.get("model_meta") if isinstance(checkpoint, dict) else None
    model_class_meta = ckpt_meta.get("model_class") if ckpt_meta else None
    if ckpt_meta is not None:
        arch_cfg = {
            k: ckpt_meta.get(k)
            for k in [
                "hidden_dim",
                "num_layers",
                "use_attention",
                "gat_heads",
                "residual",
            ]
        }
    else:
        arch_cfg = {}
    cfg_meta = cfg_meta or {}
    mismatch = {
        k: (arch_cfg.get(k), cfg_meta.get(k))
        for k in cfg_meta
        if arch_cfg.get(k) is not None and arch_cfg.get(k) != cfg_meta.get(k)
    }
    if mismatch and not force_arch_mismatch:
        raise ValueError(f"Model hyperparameter mismatch: {mismatch}")
    if ckpt_meta is not None:
        print(
            "[ARCH] from_ckpt: hidden_dim={hidden_dim}, num_layers={num_layers}, use_attention={use_attention}, gat_heads={gat_heads}, residual={residual}".format(
                **arch_cfg
            )
        )
    if cfg_meta:
        print(
            "[ARCH] from_cfg : hidden_dim={hidden_dim}, num_layers={num_layers}, use_attention={use_attention}, gat_heads={gat_heads}, residual={residual}".format(
                hidden_dim=cfg_meta.get("hidden_dim"),
                num_layers=cfg_meta.get("num_layers"),
                use_attention=cfg_meta.get("use_attention"),
                gat_heads=cfg_meta.get("gat_heads"),
                residual=cfg_meta.get("residual"),
            )
        )
    source = "CHECKPOINT (cfg ignored)" if ckpt_meta is not None else (
        "CFG" if cfg_meta else "HEURISTIC"
    )
    print(f"[ARCH] source   : {source}")

    # Support different historical parameter naming schemes.  Regardless of
    # whether checkpoint metadata is present, normalise keys so downstream
    # logic can rely on the ``layers.X`` convention.
    if any(k.startswith("conv1") for k in state) and not any(
        k.startswith("layers.0") for k in state
    ):
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
        # optionally duplicate the first/last layer under ``conv1``/``conv2`` for
        # backwards compatibility when no metadata is available.
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
                if ckpt_meta is None:
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
            if k.startswith("convs"):
                parts = k.split(".")
                idx = int(parts[1])
                rest = ".".join(parts[2:])
                base_key = f"layers.{idx}.{rest}"
                renamed[base_key] = v
                if ckpt_meta is None:
                    if idx == 0:
                        renamed[f"conv1.{rest}"] = v
                    if idx == last_idx:
                        renamed[f"conv2.{rest}"] = v
            else:
                renamed[k] = v
        state = renamed

    layer_keys = [
        k
        for k in state
        if k.startswith("layers.")
        and (
            k.endswith("weight")
            or k.endswith("lin.weight")
            or k.endswith("lin_src.weight")
        )
    ]
    fc_layer = None
    if layer_keys:
        indices = sorted({int(k.split(".")[1]) for k in layer_keys})
        conv_layers = []
        for i in indices:
            base = f"layers.{i}"
            gat_att_key = f"{base}.att_src"
            gat_lin_key = (
                f"{base}.lin_src.weight"
                if f"{base}.lin_src.weight" in state
                else f"{base}.lin.weight"
            )

            if gat_att_key in state or f"{base}.lin_src.weight" in state:
                w = state[gat_lin_key]
                in_dim = w.shape[1]
                att = state.get(gat_att_key)
                if att is not None:
                    heads = att.shape[1]
                    out_per_head = att.shape[2]
                else:
                    heads = 1
                    out_per_head = w.shape[0]
                edge_dim_layer = None
                edge_w_key = f"{base}.lin_edge.weight"
                if edge_w_key in state:
                    edge_dim_layer = state[edge_w_key].shape[1]
                conv_layers.append(
                    GATConv(
                        in_dim,
                        out_per_head,
                        heads=heads,
                        edge_dim=edge_dim_layer,
                    )
                )
            else:
                w_key = (
                    f"{base}.weight"
                    if f"{base}.weight" in state
                    else f"{base}.lin.weight"
                )
                w = state[w_key]
                in_dim = w.shape[1]
                out_dim = w.shape[0]

                edge_key = None
                if f"{base}.edge_mlp.0.weight" in state:
                    edge_key = f"{base}.edge_mlp.0.weight"
                elif f"{base}.edge_mlps.0.0.weight" in state:
                    edge_key = f"{base}.edge_mlps.0.0.weight"

                if edge_key is not None:
                    e_dim = state[edge_key].shape[1]

                    node_keys = [
                        k
                        for k in state
                        if k.startswith(f"{base}.lin.") and k.endswith("weight")
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
                        if k.startswith(f"{base}.edge_mlps") and k.endswith("weight")
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
    elif isinstance(conv_layers[0], GATConv):
        edge_dim = conv_layers[0].edge_dim

    hidden_dim = conv_layers[0].out_channels
    if isinstance(conv_layers[0], GATConv) and getattr(conv_layers[0], "concat", True):
        hidden_dim = hidden_dim * conv_layers[0].heads

    if multitask:
        node_out_dim, rnn_hidden_dim = state["node_decoder.weight"].shape
        edge_out_dim = state["edge_decoder.weight"].shape[0]
        use_att = isinstance(conv_layers[0], GATConv)
        heads = conv_layers[0].heads if use_att else 1
        model = MultiTaskGNNSurrogate(
            in_channels=conv_layers[0].in_channels,
            hidden_channels=hidden_dim,
            edge_dim=edge_dim if edge_dim is not None else 0,
            node_output_dim=node_out_dim,
            edge_output_dim=edge_out_dim,
            num_layers=len(conv_layers),
            use_attention=use_att,
            gat_heads=heads,
            dropout=0.0,
            residual=False,
            rnn_hidden_dim=rnn_hidden_dim,
            num_node_types=getattr(conv_layers[0], "num_node_types", 1),
            num_edge_types=getattr(conv_layers[0], "num_edge_types", 1),
        ).to(device)
    elif has_rnn:
        out_dim, rnn_hidden_dim = state["decoder.weight"].shape
        use_att = isinstance(conv_layers[0], GATConv)
        heads = conv_layers[0].heads if use_att else 1
        model = RecurrentGNNSurrogate(
            in_channels=conv_layers[0].in_channels,
            hidden_channels=hidden_dim,
            edge_dim=edge_dim if edge_dim is not None else 0,
            output_dim=out_dim,
            num_layers=len(conv_layers),
            use_attention=use_att,
            gat_heads=heads,
            dropout=0.0,
            residual=False,
            rnn_hidden_dim=rnn_hidden_dim,
            num_node_types=getattr(conv_layers[0], "num_node_types", 1),
            num_edge_types=getattr(conv_layers[0], "num_edge_types", 1),
        ).to(device)
    else:
        model = GNNSurrogate(conv_layers, fc_out=fc_layer).to(device)

    tank_keys = ["tank_indices", "tank_areas", "tank_edges", "tank_signs"]
    present_tanks = [k for k in tank_keys if k in state]
    if present_tanks:
        missing_tanks = [k for k in tank_keys if k not in state]
        if missing_tanks:
            raise KeyError(f"Missing tank parameters in checkpoint: {missing_tanks}")
        for k in tank_keys:
            model.register_buffer(k, state[k])

    if model_class_meta in {"RecurrentGNNSurrogate", "MultiTaskGNNSurrogate"}:
        load_state = raw_state
    else:
        load_state = state

    load_res = model.load_state_dict(load_state, strict=False)
    if isinstance(load_res, tuple):
        missing_keys, unexpected_keys = load_res
    else:
        missing_keys = load_res.missing_keys
        unexpected_keys = load_res.unexpected_keys
    critical_missing = [
        k
        for k in missing_keys
        if k.startswith("decoder")
        or k.startswith("node_decoder")
        or k.startswith("edge_decoder")
        or k.startswith("tank_")
    ]
    if critical_missing:
        raise KeyError(f"Missing keys in state dict: {critical_missing}")

    strict_load = (
        ckpt_meta is not None
        and model_class_meta == model.__class__.__name__
        and not missing_keys
        and not unexpected_keys
    )
    if strict_load:
        model.load_state_dict(load_state, strict=True)

    # ensure LayerNorm modules expose ``normalized_shape`` for compatibility
    enc = getattr(model, "encoder", None)
    if enc is not None:
        for norm in getattr(enc, "norms", []):
            if not hasattr(norm, "normalized_shape") and hasattr(norm, "weight"):
                norm.normalized_shape = (norm.weight.numel(),)

    # store expected edge attribute dimension for input checks
    model.edge_dim = edge_dim if edge_dim is not None else 0

    if norm_stats_path is not None:
        norm_path = Path(norm_stats_path)
        if not norm_path.is_absolute():
            norm_path = REPO_ROOT / norm_path
    else:
        norm_path = Path(str(full_path.with_suffix("")) + "_norm.npz")

    norm_stats = (
        checkpoint.get("norm_stats")
        if isinstance(checkpoint, dict)
        else None
    )
    if norm_stats is not None:
        model.x_mean = torch.tensor(norm_stats["x_mean"], dtype=torch.float32, device=device)
        model.x_std = torch.tensor(norm_stats["x_std"], dtype=torch.float32, device=device)
        y_mean_np = norm_stats.get("y_mean")
        y_std_np = norm_stats.get("y_std")
        if isinstance(y_mean_np, dict):
            node_mean = torch.tensor(
                y_mean_np.get("node_outputs"), dtype=torch.float32, device=device
            )
            node_std = torch.tensor(
                y_std_np.get("node_outputs"), dtype=torch.float32, device=device
            )
            model.y_mean = {"node_outputs": node_mean}
            model.y_std = {"node_outputs": node_std}
            y_mean_edge_np = y_mean_np.get("edge_outputs")
            y_std_edge_np = y_std_np.get("edge_outputs")
            if y_mean_edge_np is not None:
                model.y_mean_edge = torch.tensor(
                    y_mean_edge_np, dtype=torch.float32, device=device
                )
                model.y_std_edge = torch.tensor(
                    y_std_edge_np, dtype=torch.float32, device=device
                )
            else:
                model.y_mean_edge = model.y_std_edge = None
        elif y_mean_np is not None:
            model.y_mean = torch.tensor(y_mean_np, dtype=torch.float32, device=device)
            model.y_std = torch.tensor(y_std_np, dtype=torch.float32, device=device)
            model.y_mean_edge = model.y_std_edge = None
        else:
            model.y_mean = model.y_std = None
            model.y_mean_edge = model.y_std_edge = None

        model.y_mean_energy = None
        model.y_std_energy = None
        edge_mean_np = norm_stats.get("edge_mean")
        edge_std_np = norm_stats.get("edge_std")
        if edge_mean_np is not None:
            model.edge_mean = torch.tensor(edge_mean_np, dtype=torch.float32, device=device)
            model.edge_std = torch.tensor(edge_std_np, dtype=torch.float32, device=device)
        else:
            model.edge_mean = model.edge_std = None
    elif norm_path.exists():
        arr = np.load(norm_path)
        # Moved normalization constants to GPU to avoid device transfer
        model.x_mean = torch.tensor(arr["x_mean"], dtype=torch.float32, device=device)
        model.x_std = torch.tensor(arr["x_std"], dtype=torch.float32, device=device)
        if "y_mean" in arr:
            model.y_mean = torch.tensor(arr["y_mean"], dtype=torch.float32, device=device)
            model.y_std = torch.tensor(arr["y_std"], dtype=torch.float32, device=device)
            model.y_mean_edge = model.y_std_edge = None
        elif "y_mean_node" in arr:
            node_mean = torch.tensor(arr["y_mean_node"], dtype=torch.float32, device=device)
            node_std = torch.tensor(arr["y_std_node"], dtype=torch.float32, device=device)
            model.y_mean = {"node_outputs": node_mean}
            model.y_std = {"node_outputs": node_std}
            if "y_mean_edge" in arr:
                model.y_mean_edge = torch.tensor(
                    arr["y_mean_edge"], dtype=torch.float32, device=device
                )
                model.y_std_edge = torch.tensor(
                    arr["y_std_edge"], dtype=torch.float32, device=device
                )
            else:
                model.y_mean_edge = model.y_std_edge = None
        else:
            model.y_mean = model.y_std = None
            model.y_mean_edge = model.y_std_edge = None

        model.y_mean_energy = None
        model.y_std_energy = None
        if "edge_mean" in arr:
            model.edge_mean = torch.tensor(arr["edge_mean"], dtype=torch.float32, device=device)
            model.edge_std = torch.tensor(arr["edge_std"], dtype=torch.float32, device=device)
        else:
            model.edge_mean = model.edge_std = None
    else:
        model.x_mean = model.x_std = model.y_mean = model.y_std = None
        model.y_mean_edge = model.y_std_edge = None
        model.edge_mean = model.edge_std = None

    # Log shapes and checksum of normalization statistics for reproducibility
    norm_hash: Optional[str] = None
    stats_tensors = []
    if getattr(model, "x_mean", None) is not None:
        stats_tensors.append(model.x_mean.detach().cpu())
    if getattr(model, "x_std", None) is not None:
        stats_tensors.append(model.x_std.detach().cpu())
    if getattr(model, "y_mean", None) is not None:
        if isinstance(model.y_mean, dict):
            stats_tensors.extend(
                [v.detach().cpu() for v in model.y_mean.values()]
            )
            stats_tensors.extend(
                [v.detach().cpu() for v in model.y_std.values()]
            )
        else:
            stats_tensors.append(model.y_mean.detach().cpu())
            stats_tensors.append(model.y_std.detach().cpu())
    if getattr(model, "y_mean_edge", None) is not None:
        stats_tensors.append(model.y_mean_edge.detach().cpu())
        stats_tensors.append(model.y_std_edge.detach().cpu())
    if getattr(model, "edge_mean", None) is not None:
        stats_tensors.append(model.edge_mean.detach().cpu())
        stats_tensors.append(model.edge_std.detach().cpu())
    if stats_tensors:
        import hashlib

        md5 = hashlib.md5()
        for t in stats_tensors:
            md5.update(t.to(torch.float32).numpy().tobytes())
        y_mean_attr = getattr(model, "y_mean", None)
        y_std_attr = getattr(model, "y_std", None)
        y_mean_node_shape = None
        y_std_node_shape = None
        if isinstance(y_mean_attr, dict):
            y_mean_node_shape = tuple(y_mean_attr["node_outputs"].shape)
            y_std_node_shape = (
                tuple(y_std_attr["node_outputs"].shape)
                if isinstance(y_std_attr, dict)
                else None
            )
        elif y_mean_attr is not None:
            y_mean_node_shape = tuple(y_mean_attr.shape)
            y_std_node_shape = (
                tuple(y_std_attr.shape) if y_std_attr is not None else None
            )
        shapes = {
            "x_mean": tuple(model.x_mean.shape) if getattr(model, "x_mean", None) is not None else None,
            "x_std": tuple(model.x_std.shape) if getattr(model, "x_std", None) is not None else None,
            "y_mean_node": y_mean_node_shape,
            "y_std_node": y_std_node_shape,
            "y_mean_edge": tuple(model.y_mean_edge.shape) if getattr(model, "y_mean_edge", None) is not None else None,
            "y_std_edge": tuple(model.y_std_edge.shape) if getattr(model, "y_std_edge", None) is not None else None,
            "edge_mean": tuple(model.edge_mean.shape) if getattr(model, "edge_mean", None) is not None else None,
            "edge_std": tuple(model.edge_std.shape) if getattr(model, "edge_std", None) is not None else None,
        }
        norm_hash = md5.hexdigest()
        if norm_stats is not None:
            stored_hash = norm_stats.get("hash")
            if stored_hash and stored_hash != norm_hash:
                raise ValueError("Stored normalization stats hash mismatch")
            if norm_path.exists():
                arr = np.load(norm_path)
                md5_npz = hashlib.md5()
                npz_arrays = [arr["x_mean"], arr["x_std"]]
                if "y_mean_node" in arr:
                    npz_arrays.extend([arr["y_mean_node"], arr["y_std_node"]])
                    if "y_mean_edge" in arr:
                        npz_arrays.extend([arr["y_mean_edge"], arr["y_std_edge"]])
                elif "y_mean" in arr:
                    npz_arrays.extend([arr["y_mean"], arr["y_std"]])
                if "edge_mean" in arr:
                    npz_arrays.extend([arr["edge_mean"], arr["edge_std"]])
                for a in npz_arrays:
                    md5_npz.update(a.tobytes())
                npz_hash = md5_npz.hexdigest()
                if stored_hash and npz_hash != stored_hash:
                    raise ValueError("_norm.npz hash mismatch with checkpoint")
            if stored_hash:
                norm_hash = stored_hash
        print(f"Loaded normalization stats shapes: {shapes}, md5: {norm_hash}")
        model.norm_hash = norm_hash

    model._ckpt_meta = ckpt_meta
    model._cfg_meta = cfg_meta
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

    if norm_hash is None:
        model.norm_hash = None
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




def compute_mpc_cost(
    pump_speeds: torch.Tensor,
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
    pump_info: Optional[List[Tuple[int, int, int]]] = None,
    return_energy: bool = False,
    init_tank_levels: Optional[torch.Tensor] = None,
    skip_normalization: bool = False,
    w_p: float = 100.0,
    w_c: float = 100.0,
    w_e: float = 1.0,
    energy_scale: float = 1e-9,
    barrier: str = "softplus",
    auto_energy_scale: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the MPC cost for a sequence of pump speeds.

    The cost combines pressure and chlorine constraint violations, pump
    energy use and a smoothness term on control differences. ``demands`` can be
    used to provide per-node demand values for each step which keeps the
    surrogate inputs consistent with training.  The energy term is scaled by
    ``energy_scale`` which defaults to converting Joules to megawatt-hours
    (``1e-9``).  ``w_p``, ``w_c`` and ``w_e`` weight the respective cost
    components. Pressure violations are always penalised with a squared hinge
    which softly enforces ``Pmin``. ``barrier`` selects how chlorine
    violations are penalised: ``"softplus"`` (default) applies a smooth
    softplus barrier, ``"exp"`` uses an exponential barrier and
    ``"cubic"`` falls back to the previous cubic hinge.  When
    ``auto_energy_scale`` is ``True`` the average hourly energy with all pumps
    operating at base speed is estimated and its reciprocal is used as the
    scaling factor so that energy penalties are on a comparable scale across
    networks.
    """
    if auto_energy_scale:
        with torch.no_grad():
            base_speeds = torch.ones_like(pump_speeds)
            base_cost, _ = compute_mpc_cost(
                base_speeds,
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
                init_tank_levels,
                skip_normalization,
                0.0,
                0.0,
                1.0,
                1.0,
                barrier,
                False,
            )
        avg_energy = base_cost.item() / horizon
        energy_scale = 1.0 / (avg_energy + EPS)

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

    if edge_attr is not None and getattr(model, "edge_mean", None) is not None and not skip_normalization:
        edge_attr_norm = (edge_attr - model.edge_mean) / (model.edge_std + EPS)
    else:
        edge_attr_norm = edge_attr

    for t in range(horizon):
        d = demands[t] if demands is not None else None
        raw_speed = pump_speeds[t]
        clamped = torch.clamp(raw_speed, 0.0, MAX_PUMP_SPEED)
        speed = raw_speed + (clamped - raw_speed).detach()
        x = prepare_node_features(
            feature_template, cur_p, cur_c, speed, model, d, skip_normalization
        )
        if hasattr(model, "rnn"):
            seq_in = x.unsqueeze(0).unsqueeze(0)
            out = model(seq_in, edge_index, edge_attr_norm, node_types, edge_types)
            if isinstance(out, dict):
                pred = out.get("node_outputs")[0, 0]
                flows = out.get("edge_outputs")[0, 0].squeeze(-1)
            else:
                pred = out[0, 0]
                flows = None
        else:
            out = model(x, edge_index, edge_attr_norm, node_types, edge_types)
            if isinstance(out, dict):
                pred = out.get("node_outputs")
                flows = out.get("edge_outputs").squeeze(-1)
            else:
                pred = out
                flows = None

        if getattr(model, "y_mean", None) is not None:
            y_mean_attr = model.y_mean
            y_std_attr = model.y_std
            if isinstance(y_mean_attr, dict):
                y_mean = y_mean_attr.get("node_outputs")
                y_std = (
                    y_std_attr.get("node_outputs")
                    if isinstance(y_std_attr, dict)
                    else None
                )
            else:
                y_mean = y_mean_attr
                y_std = y_std_attr

            if y_mean is not None and y_std is not None:
                # Handle per-node normalisation statistics where y_mean has
                # shape (num_nodes, num_outputs). In this case we can
                # denormalise directly via broadcasting.
                if y_mean.dim() == 2 and y_mean.shape == pred.shape:
                    pred = pred * (y_std + EPS) + y_mean
                else:
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(-1)
                    if (
                        pred.dim() == 2
                        and pred.shape[1] != y_mean.shape[0]
                        and pred.shape[0] == y_mean.shape[0]
                    ):
                        pred = pred.t()
                    target_dim = min(pred.shape[1], y_mean.shape[0])
                    pred = torch.cat(
                        [
                            pred[:, :target_dim]
                            * (y_std[:target_dim].view(1, -1) + EPS)
                            + y_mean[:target_dim].view(1, -1),
                            pred[:, target_dim:],
                        ],
                        dim=1,
                    )
        assert not torch.isnan(pred).any(), "NaN prediction"
        pred_p = pred[:, 0]
        if pred.shape[1] > 1:
            pred_c = torch.expm1(pred[:, 1]) * 1000.0
        else:
            pred_c = torch.zeros_like(pred_p)

        # ------------------------------------------------------------------
        # Cost terms
        # ------------------------------------------------------------------
        w_s = 0.01

        Cmin_safe = Cmin + 0.05

        psf = torch.relu(Pmin - pred_p)
        pressure_penalty = torch.sum(psf ** 2)

        csf = torch.clamp(Cmin_safe - pred_c, min=0.0)
        if barrier == "exp":
            chlorine_penalty = torch.sum(torch.expm1(csf))
        elif barrier == "cubic":
            chlorine_penalty = torch.sum(csf ** 3)
        else:
            chlorine_penalty = torch.sum(F.softplus(csf) ** 2)

        if flows is not None and pump_info is not None:
            head = pred_p + feature_template[:, 3]
            energy_term_j = torch.tensor(0.0, device=device)
            eff = wn.options.energy.global_efficiency / 100.0
            for idx, s_idx, e_idx in pump_info:
                raw_f = flows[idx]
                raw_hl = head[e_idx] - head[s_idx]
                if raw_f.item() < 0 or raw_hl.item() < 0:
                    warnings.warn(
                        "Negative headloss or flow predicted; clamping.",
                        RuntimeWarning,
                    )
                f = torch.abs(raw_f)
                hl = torch.clamp(raw_hl, min=0.0)
                e = 1000.0 * 9.81 * hl * f / eff * 3600.0
                energy_term_j = energy_term_j + e
                if t == 0 and return_energy:
                    energy_first = energy_first + e
        else:
            energy_term_j = torch.sum(speed ** 2)

        energy_term = energy_term_j * energy_scale

        step_cost = (
            w_p * pressure_penalty
            + w_c * chlorine_penalty
            + w_e * energy_term
        )

        total_cost = total_cost + step_cost

        if t > 0:
            # small penalty on rapid pump switching to produce smoother
            # control sequences
            prev_raw = pump_speeds[t - 1]
            prev_clamped = torch.clamp(prev_raw, 0.0, MAX_PUMP_SPEED)
            prev_speed = prev_raw + (prev_clamped - prev_raw).detach()
            smoothness_penalty = smoothness_penalty + torch.sum((speed - prev_speed) ** 2)

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
    pump_info: Optional[List[Tuple[int, int, int]]] = None,
    profile: bool = False,
    skip_normalization: bool = False,
    w_p: float = 100.0,
    w_c: float = 100.0,
    w_e: float = 1.0,
    energy_scale: float = 1e-9,
    barrier: str = "softplus",
    gmax: float = 1.0,
    auto_energy_scale: bool = False,
) -> Tuple[torch.Tensor, List[float], float]:
    """Optimize pump speeds for one hour using gradient-based MPC.

    The optimization is performed in two phases: a short warm start with
    Adam followed by refinement using L-BFGS. ``iterations`` controls the total
    number of optimization steps, split approximately 20/80 between the two
    phases. The speed variables are clamped to ``[0, MAX_PUMP_SPEED]`` after each update to
    enforce valid pump settings. Gradients on the control variables are clipped
    to ``[-gmax, gmax]`` to improve numerical stability.

    Parameters
    ----------
    demands : torch.Tensor, optional
        Horizon x num_nodes demand values used when assembling node features.
    u_warm : torch.Tensor, optional
        Previous speed sequence to warm start the optimization.
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
        init = torch.full((horizon, num_pumps), MAX_PUMP_SPEED, device=device)
    pump_speeds = init.clone().detach().requires_grad_(True)

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
    adam_opt = torch.optim.Adam([pump_speeds], lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_opt, "min", patience=3, factor=0.5)

    for _ in range(adam_steps):
        adam_opt.zero_grad()
        cost, _ = compute_mpc_cost(
            pump_speeds,
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
            skip_normalization,
            w_p,
            w_c,
            w_e,
            energy_scale,
            barrier,
            auto_energy_scale,
        )
        cost.backward()
        if gmax is not None:
            pump_speeds.grad.clamp_(-gmax, gmax)
        adam_opt.step()
        with torch.no_grad():
            pump_speeds.copy_(pump_speeds.clamp(0.0, MAX_PUMP_SPEED))
        scheduler.step(cost.item())
        cost_history.append(float(cost.item()))

    # --- Phase 2: L-BFGS refinement --------------------------------------
    lbfgs_steps = max(iterations - adam_steps, 1)
    lbfgs_opt = torch.optim.LBFGS([pump_speeds], max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

    def closure():
        lbfgs_opt.zero_grad()
        c, _ = compute_mpc_cost(
            pump_speeds,
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
            skip_normalization,
            w_p,
            w_c,
            w_e,
            energy_scale,
            barrier,
            auto_energy_scale,
        )
        c.backward()
        if gmax is not None:
            pump_speeds.grad.clamp_(-gmax, gmax)
        return c

    final_cost = lbfgs_opt.step(closure)
    cost_history.append(float(final_cost))
    _, energy_first = compute_mpc_cost(
        pump_speeds,
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
        skip_normalization,
        w_p,
        w_c,
        w_e,
        energy_scale,
        barrier,
        auto_energy_scale,
    )
    with torch.no_grad():
        pump_speeds.copy_(pump_speeds.clamp(0.0, MAX_PUMP_SPEED))

    if hasattr(model, "rnn") and not prev_training:
        model.eval()

    if profile and start_time is not None:
        end_time = time.time()
        print(f"[profile] run_mpc_step: {end_time - start_time:.4f}s")

    return pump_speeds.detach(), cost_history, float(energy_first.item()) if energy_first is not None else 0.0


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
    speed_seq: torch.Tensor,
    device: torch.device,
    demands: Optional[torch.Tensor] = None,
    skip_normalization: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Propagate the network state using the surrogate model.

    The current state ``pressures``/``chlorine`` is advanced through the
    sequence of pump speeds ``speed_seq`` without running EPANET.  The
    function returns dictionaries for the next pressures and chlorine levels
    after applying the entire sequence.  ``pressures`` and ``chlorine`` can be
    either dictionaries for a single scenario or lists of dictionaries for
    batched evaluation:

    ``pressures = [dict(...), dict(...)]``
    ``chlorine = [dict(...), dict(...)]``
    """

    if edge_attr is not None and getattr(model, "edge_mean", None) is not None and not skip_normalization:
        edge_attr = (edge_attr - model.edge_mean) / (model.edge_std + EPS)

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
        for t, speed in enumerate(speed_seq):
            if speed.dim() == 1:
                speed_in = speed.view(1, -1).expand(batch_size, -1)
            else:
                speed_in = speed
            d = demands[t] if demands is not None else None
            x = prepare_node_features(
                feature_template, cur_p, cur_c, speed_in, model, d, skip_normalization
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
            pred = pred.view(batch_size, feature_template.size(0), -1)
            if getattr(model, "y_mean", None) is not None:
                y_mean_attr = model.y_mean
                y_std_attr = model.y_std
                if isinstance(y_mean_attr, dict):
                    y_mean = y_mean_attr.get("node_outputs")
                    y_std = (
                        y_std_attr.get("node_outputs")
                        if isinstance(y_std_attr, dict)
                        else None
                    )
                else:
                    y_mean = y_mean_attr
                    y_std = y_std_attr
                if y_mean is not None and y_std is not None:
                    if y_mean.dim() == 2 and y_mean.shape == pred.shape[1:]:
                        pred = pred * (y_std.unsqueeze(0) + EPS) + y_mean.unsqueeze(0)
                    else:
                        pred = pred * (y_std + EPS) + y_mean
            assert not torch.isnan(pred).any(), "NaN prediction"
            cur_p = pred[:, :, 0]
            cur_c = torch.expm1(pred[:, :, 1]) * 1000.0

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


class PressureBiasCorrector:
    """Maintain a rolling mean of pressure residuals and apply corrections."""

    def __init__(self, num_nodes: int, window: int):
        self.window = max(1, window)
        self.buffers = [deque(maxlen=self.window) for _ in range(num_nodes)]
        self.bias = np.zeros(num_nodes, dtype=np.float32)

    def apply(self, pressures: Dict[str, float], node_to_index: Dict[str, int]) -> Dict[str, float]:
        for name, idx in node_to_index.items():
            pressures[name] -= float(self.bias[idx])
        return pressures

    def update(self, residual: np.ndarray) -> None:
        for i, r in enumerate(residual):
            self.buffers[i].append(float(r))
            self.bias[i] = float(np.mean(self.buffers[i]))

    def reset(self) -> None:
        for buf in self.buffers:
            buf.clear()
        self.bias.fill(0.0)



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
    feedback_interval: int = 1,
    run_name: str = "",
    profile: bool = False,
    skip_normalization: bool = False,
    w_p: float = 100.0,
    w_c: float = 100.0,
    w_e: float = 1.0,
    energy_scale: float = 1e-9,
    barrier: str = "softplus",
    gmax: float = 1.0,
    bias_correction: bool = False,
    bias_window: int = 1,
    auto_energy_scale: bool = False,
) -> pd.DataFrame:
    """Run 24-hour closed-loop MPC using the surrogate for fast updates.

    EPANET is invoked only every ``feedback_interval`` hours (default once per
    hour) to obtain ground-truth measurements.  All intermediate steps update
    the pressures and chlorine levels using the GNN surrogate which allows the
    loop to run nearly instantly.
    """
    if feedback_interval > 1:
        print(
            f"WARNING: feedback_interval is {feedback_interval}; "
            "surrogate predictions may drift without hourly EPANET feedback."
        )

    expected_in_dim = 4 + len(pump_names)
    in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", expected_in_dim)
    if in_dim < expected_in_dim:
        raise ValueError(
            "Loaded model was trained without pump speed inputs - rerun train_gnn.py"
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

    bias_corrector = None
    if bias_correction:
        bias_corrector = PressureBiasCorrector(len(node_to_index), bias_window)
    # obtain hydraulic state at time zero
    wn.options.time.duration = 0
    wn.options.time.report_timestep = 0
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(str(TEMP_DIR / "temp"))
    p_arr = results.node["pressure"].iloc[0].to_numpy(dtype=np.float32)
    c_arr = results.node["quality"].iloc[0].to_numpy(dtype=np.float32)
    pressures = dict(zip(wn.node_name_list, p_arr))
    chlorine = dict(zip(wn.node_name_list, c_arr))
    for res_name in wn.reservoir_name_list:
        idx = node_idx[res_name]
        head = wn.get_node(res_name).base_head
        p_arr[idx] = head
        pressures[res_name] = head
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
    prev_speed: Optional[torch.Tensor] = None
    all_costs: List[float] = []

    for hour in range(24):
        start = time.time()
        demands = compute_demand_vectors(
            wn, node_to_index, base_demands, hour, horizon
        ).to(device)
        speed_opt, costs, energy_first = run_mpc_step(
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
            prev_speed,
            pump_info,
            profile,
            skip_normalization,
            w_p,
            w_c,
            w_e,
            energy_scale,
            barrier,
            gmax,
            auto_energy_scale,
        )
        all_costs.extend(costs)
        prev_speed = speed_opt.detach()

        # apply control to network object for consistency
        first_speeds = speed_opt[0]
        for i, pump in enumerate(pump_names):
            link = wn.get_link(pump)
            spd = float(max(0.0, min(MAX_PUMP_SPEED, first_speeds[i].item())))
            link.base_speed = spd
            link.initial_status = (
                wntr.network.base.LinkStatus.Closed
                if spd == 0.0
                else wntr.network.base.LinkStatus.Open
            )

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
            pred_pressures = None
            if bias_corrector is not None:
                pred_pressures, _ = propagate_with_surrogate(
                    wn,
                    model,
                    edge_index,
                    edge_attr,
                    node_types,
                    edge_types,
                    feature_template,
                    pressures,
                    chlorine,
                    speed_opt,
                    device,
                    demands,
                    skip_normalization,
                )
                pred_pressures = bias_corrector.apply(pred_pressures, node_idx)
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
            for res_name in wn.reservoir_name_list:
                idx = node_idx[res_name]
                head = wn.get_node(res_name).base_head
                p_arr[idx] = head
                pressures[res_name] = head
            if bias_corrector is not None and pred_pressures is not None:
                residual = p_arr - np.array([pred_pressures[n] for n in wn.node_name_list])
                bias_corrector.reset()
                bias_corrector.update(residual)
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
                speed_opt,
                device,
                demands,
                skip_normalization,
            )
            if bias_corrector is not None:
                pressures = bias_corrector.apply(pressures, node_idx)
            cur_p = torch.tensor([pressures[n] for n in wn.node_name_list], dtype=torch.float32, device=device)
            cur_c = torch.tensor([chlorine[n] for n in wn.node_name_list], dtype=torch.float32, device=device)
            end = time.time()
            energy = energy_first
        bias_min = bias_max = 0.0
        if bias_corrector is not None:
            b_abs = np.abs(bias_corrector.bias)
            bias_min = float(b_abs.min())
            bias_max = float(b_abs.max())
        min_p = max(min(pressures[n] for n in wn.junction_name_list), 0.0)
        min_c = max(min(chlorine[n] for n in wn.junction_name_list), 0.0)
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
                "controls": first_speeds.cpu().numpy().tolist(),
                "bias_min": bias_min,
                "bias_max": bias_max,
                # Store full network state for optional animations
                "pressures": dict(pressures),
                "chlorine": dict(chlorine),
            }
        )
        if run_name:
            pump_dict = {
                pump_names[i]: float(first_speeds[i])
                for i in range(len(pump_names))
            }
            try:
                plot_network_state_epyt(pressures, pump_dict, run_name, hour)
            except Exception as exc:
                warnings.warn(f"plot_network_state_epyt failed: {exc}")
        print(
            f"Hour {hour}: minP={min_p:.2f}, minC={min_c:.3f}, energy={energy:.2f}, runtime={end-start:.2f}s, bias=[{bias_min:.3f},{bias_max:.3f}]"
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
        default=1,
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
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Disable feature normalization for ablation",
    )
    parser.add_argument(
        "--energy-scale",
        type=float,
        default=1e-9,
        help="Scale factor applied to pump energy (e.g., 1e-9 converts J to MWh)",
    )
    parser.add_argument(
        "--auto-energy-scale",
        action="store_true",
        help="Estimate typical hourly energy at base speed to normalise energy cost",
    )
    parser.add_argument("--w_p", type=float, default=100.0, help="Weight on pressure violations")
    parser.add_argument("--w_c", type=float, default=100.0, help="Weight on chlorine violations")
    parser.add_argument("--w_e", type=float, default=1.0, help="Weight on energy usage")
    parser.add_argument(
        "--barrier",
        choices=["softplus", "exp", "cubic"],
        default="softplus",
        help="Penalty type for chlorine violations (pressure uses squared hinge)",
    )
    parser.add_argument(
        "--gmax",
        type=float,
        default=1.0,
        help="Infinity-norm gradient clipping threshold for controls",
    )
    parser.add_argument(
        "--bias-correction",
        action="store_true",
        help="Enable pressure bias correction using recent EPANET residuals",
    )
    parser.add_argument(
        "--bias-window",
        type=int,
        default=1,
        help="Number of residuals to average for bias correction",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch ops",
    )
    args = parser.parse_args()
    configure_seeds(args.seed, args.deterministic)
    if args.feedback_interval > 1:
        print(
            f"WARNING: --feedback-interval set to {args.feedback_interval}; "
            "surrogate predictions may drift without hourly EPANET feedback."
        )

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
    norm_md5 = getattr(model, "norm_hash", None)
    model_layers = len(getattr(model, "layers", []))
    model_hidden = getattr(getattr(model, "layers", [None])[0], "out_channels", None)

    expected_in_dim = 4 + len(pump_names)
    in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", expected_in_dim)
    if in_dim != expected_in_dim:
        print(
            f"Loaded surrogate expects {in_dim} input features "
            f"but the network requires {expected_in_dim}."
        )
        print(
            "The provided model was likely trained without pump speed inputs.\n"
            "Re-train the surrogate using data generated with pump features."
        )
        return

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        run_name,
        args.profile,
        args.skip_normalization,
        args.w_p,
        args.w_c,
        args.w_e,
        args.energy_scale,
        args.barrier,
        args.gmax,
        args.bias_correction,
        args.bias_window,
        args.auto_energy_scale,
    )

    cfg_extra = {
        "norm_stats_md5": norm_md5,
        "model_layers": model_layers,
        "model_hidden_dim": model_hidden,
    }
    save_config(
        REPO_ROOT / "logs" / f"config_mpc_{run_name}.yaml",
        vars(args),
        cfg_extra,
    )


if __name__ == "__main__":
    main()
