import argparse
import time
from typing import Dict, List, Optional
import os
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
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
    )  # type: ignore
except ImportError:  # pragma: no cover - executed when run as a script
    from train_gnn import HydroConv, RecurrentGNNSurrogate, MultiTaskGNNSurrogate
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


class GNNSurrogate(torch.nn.Module):
    """Flexible GCN used by the MPC controller."""

    def __init__(self, conv_layers: List[nn.Module], fc_out: Optional[nn.Linear] = None):
        super().__init__()
        self.layers = nn.ModuleList(conv_layers)
        self.conv1 = self.layers[0]
        self.conv2 = self.layers[-1]
        self.fc_out = fc_out

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, conv in enumerate(self.layers):
            if isinstance(conv, HydroConv):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        if self.fc_out is not None:
            x = self.fc_out(x)
        return x


def load_network(inp_file: str, return_edge_attr: bool = False):
    """Load EPANET network and build edge index for PyG.

    Parameters
    ----------
    return_edge_attr : bool, optional
        If ``True`` also return the normalized edge attributes.
    """
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.quality.parameter = "CHEMICAL"
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600

    node_to_index = {n: i for i, n in enumerate(wn.node_name_list)}
    edges = []
    attrs = []
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
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if return_edge_attr:
        edge_attr = np.array(attrs, dtype=np.float32)
        edge_attr[:, 2] = np.log1p(edge_attr[:, 2])
        scaler = MinMaxScaler()
        edge_attr = scaler.fit_transform(edge_attr)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        return wn, node_to_index, wn.pump_name_list, edge_index, edge_attr
    return wn, node_to_index, wn.pump_name_list, edge_index



def load_surrogate_model(device: torch.device, path: Optional[str] = None) -> GNNSurrogate:
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
        renamed = dict(state)
        mapping = {
            "conv1.bias": "layers.0.bias",
            "conv1.weight": "layers.0.weight",
            "conv1.lin.weight": "layers.0.lin.weight",
            "conv1.lin.bias": "layers.0.lin.bias",
            "conv1.edge_mlp.0.weight": "layers.0.edge_mlp.0.weight",
            "conv1.edge_mlp.0.bias": "layers.0.edge_mlp.0.bias",
            "conv2.bias": "layers.1.bias",
            "conv2.weight": "layers.1.weight",
            "conv2.lin.weight": "layers.1.lin.weight",
            "conv2.lin.bias": "layers.1.lin.bias",
            "conv2.edge_mlp.0.weight": "layers.1.edge_mlp.0.weight",
            "conv2.edge_mlp.0.bias": "layers.1.edge_mlp.0.bias",
        }
        for old, new in mapping.items():
            if old in state:
                renamed[new] = state[old]
                del renamed[old]
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
            if f"layers.{i}.edge_mlp.0.weight" in state:
                e_dim = state[f"layers.{i}.edge_mlp.0.weight"].shape[1]
                conv_layers.append(HydroConv(in_dim, out_dim, e_dim))
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
        edge_dim = conv_layers[0].edge_mlp[0].in_features

    if multitask:
        node_out_dim, rnn_hidden_dim = state["node_decoder.weight"].shape
        edge_out_dim = state["edge_decoder.weight"].shape[0]
        energy_out_dim = state["energy_decoder.weight"].shape[0]
        model = MultiTaskGNNSurrogate(
            in_channels=conv_layers[0].in_channels,
            hidden_channels=conv_layers[0].out_channels,
            edge_dim=edge_dim if edge_dim is not None else 0,
            node_output_dim=node_out_dim,
            edge_output_dim=edge_out_dim,
            energy_output_dim=energy_out_dim,
            num_layers=len(conv_layers),
            use_attention=False,
            gat_heads=1,
            dropout=0.0,
            residual=False,
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(device)
        model.conv1 = model.encoder.convs[0]
        model.conv2 = model.encoder.convs[-1]
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
        ).to(device)
        model.conv1 = model.encoder.convs[0]
        model.conv2 = model.encoder.convs[-1]
    else:
        model = GNNSurrogate(conv_layers, fc_out=fc_layer).to(device)

    model.load_state_dict(state, strict=False)

    norm_path = str(full_path.with_suffix("")) + "_norm.npz"
    if os.path.exists(norm_path):
        arr = np.load(norm_path)
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
    else:
        model.x_mean = model.x_std = model.y_mean = model.y_std = None

    model.eval()
    return model


def prepare_node_features(
    wn: wntr.network.WaterNetworkModel,
    pressures: Dict[str, float],
    chlorine: Dict[str, float],
    pump_controls: torch.Tensor,
    node_to_index: Dict[str, int],
    pump_names: List[str],
    device: torch.device,
    model: GNNSurrogate,
) -> torch.Tensor:
    """Build node features for the surrogate model.

    The function always constructs the full set of features, including pump
    control inputs.  However, the loaded surrogate model might have been trained
    without pump controls.  To accomodate both cases, the returned tensor is
    truncated to ``in_dim`` columns.
    """

    num_nodes = len(wn.node_name_list)
    num_pumps = len(pump_names)

    # ``pump_controls`` is a tensor that will be optimised by the MPC loop.
    # Building the feature matrix directly with PyTorch tensors preserves the
    # gradient through to the cost function.
    pump_controls = pump_controls.to(dtype=torch.float32, device=device)
    feats = torch.zeros((num_nodes, 4 + num_pumps), dtype=torch.float32, device=device)

    for name, idx in node_to_index.items():
        node = wn.get_node(name)
        if name in wn.junction_name_list:
            demand = node.demand_timeseries_list[0].base_value
        else:
            demand = 0.0
        if name in wn.junction_name_list or name in wn.tank_name_list:
            elev = node.elevation
        elif name in wn.reservoir_name_list:
            # ``Reservoir`` objects expose ``head`` as ``None`` and store the
            # hydraulic head in ``base_head``. Using ``head`` directly would
            # produce ``NaN`` feature values which later lead to ``NaN``
            # predictions during MPC optimisation.
            elev = node.base_head
        else:
            elev = node.head
        if elev is None:
            elev = 0.0

        feats[idx, 0] = float(demand)
        feats[idx, 1] = float(pressures.get(name, 0.0))
        feats[idx, 2] = float(chlorine.get(name, 0.0))
        feats[idx, 3] = float(elev)

    # broadcast the pump control vector to all nodes
    feats[:, 4 : 4 + num_pumps] = pump_controls.view(1, -1).expand(num_nodes, num_pumps)

    feats = feats[:, : model.conv1.in_channels]

    if getattr(model, "x_mean", None) is not None:
        feats = (feats - model.x_mean) / model.x_std
    return feats


def compute_mpc_cost(
    u: torch.Tensor,
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pressures: Dict[str, float],
    chlorine: Dict[str, float],
    horizon: int,
    node_to_index: Dict[str, int],
    pump_names: List[str],
    device: torch.device,
    Pmin: float,
    Cmin: float,
) -> torch.Tensor:
    """Return the MPC cost for a sequence of pump controls.

    The cost combines pressure and chlorine constraint violations, pump
    energy use and a smoothness term on control differences. Violations are
    penalized cubically to strongly discourage operating below the specified
    minimum thresholds.
    """
    cur_p = dict(pressures)
    cur_c = dict(chlorine)
    total_cost = torch.tensor(0.0, device=device)
    smoothness_penalty = torch.tensor(0.0, device=device)

    for t in range(horizon):
        x = prepare_node_features(
            wn,
            cur_p,
            cur_c,
            u[t],
            node_to_index,
            pump_names,
            device,
            model,
        )
        if hasattr(model, "rnn"):
            seq_in = x.unsqueeze(0).unsqueeze(0)
            pred = model(seq_in, edge_index, edge_attr)
            if isinstance(pred, dict):
                pred = pred.get("node_outputs")[0, 0]
        else:
            if hasattr(model, "rnn"):
                seq_in = x.unsqueeze(0).unsqueeze(0)
                pred = model(seq_in, edge_index, edge_attr)
                if isinstance(pred, dict):
                    pred = pred.get("node_outputs")[0, 0]
            else:
                pred = model(x, edge_index, edge_attr)
                if isinstance(pred, dict):
                    pred = pred.get("node_outputs")
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

        psf = torch.clamp(Pmin - pred_p, min=0.0)
        csf = torch.clamp(Cmin - pred_c, min=0.0)
        pressure_penalty = torch.sum(psf ** 3)
        chlorine_penalty = torch.sum(csf ** 3)

        energy_term = torch.sum(u[t] ** 2)

        step_cost = (
            w_p * pressure_penalty
            + w_c * chlorine_penalty
            + w_e * energy_term
        )

        total_cost = total_cost + step_cost

        if t > 0:
            # small penalty on rapid pump switching to produce smoother
            # control sequences
            smoothness_penalty = smoothness_penalty + torch.sum((u[t] - u[t - 1]) ** 2)

        # update dictionaries for next step
        for idx, name in enumerate(wn.node_name_list):
            cur_p[name] = pred_p[idx].item()
            cur_c[name] = pred_c[idx].item()

    total_cost = total_cost + w_s * smoothness_penalty

    return total_cost


def run_mpc_step(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pressures: Dict[str, float],
    chlorine: Dict[str, float],
    horizon: int,
    iterations: int,
    node_to_index: Dict[str, int],
    pump_names: List[str],
    device: torch.device,
    Pmin: float,
    Cmin: float,
    u_warm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Optimize pump controls for one hour using gradient-based MPC.

    The optimization is performed in two phases: a short warm start with
    Adam followed by refinement using L-BFGS. ``iterations`` controls the total
    number of optimization steps, split approximately 20/80 between the two
    phases. ``u`` is clamped to ``[0, 1]`` after each update to enforce valid
    pump settings.

    Parameters
    ----------
    u_warm : torch.Tensor, optional
        Previous control sequence to warm start the optimization.
    """
    num_pumps = len(pump_names)
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
        cost = compute_mpc_cost(
            u,
            wn,
            model,
            edge_index,
            edge_attr,
            pressures,
            chlorine,
            horizon,
            node_to_index,
            pump_names,
            device,
            Pmin,
            Cmin,
        )
        cost.backward()
        adam_opt.step()
        with torch.no_grad():
            u.data.clamp_(0.0, 1.0)
        scheduler.step(cost.item())

    # --- Phase 2: L-BFGS refinement --------------------------------------
    lbfgs_steps = max(iterations - adam_steps, 1)
    lbfgs_opt = torch.optim.LBFGS([u], max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

    def closure():
        lbfgs_opt.zero_grad()
        c = compute_mpc_cost(
            u,
            wn,
            model,
            edge_index,
            edge_attr,
            pressures,
            chlorine,
            horizon,
            node_to_index,
            pump_names,
            device,
            Pmin,
            Cmin,
        )
        c.backward()
        return c

    lbfgs_opt.step(closure)
    with torch.no_grad():
        u.data.clamp_(0.0, 1.0)

    if hasattr(model, "rnn") and not prev_training:
        model.eval()

    return u.detach()


def propagate_with_surrogate(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pressures: Dict[str, float],
    chlorine: Dict[str, float],
    control_seq: torch.Tensor,
    node_to_index: Dict[str, int],
    pump_names: List[str],
    device: torch.device,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Propagate the network state using the surrogate model.

    The current state ``pressures``/``chlorine`` is advanced through the
    sequence of pump controls ``control_seq`` without running EPANET.  The
    function returns dictionaries for the next pressures and chlorine levels
    after applying the entire sequence.
    """

    cur_p = dict(pressures)
    cur_c = dict(chlorine)
    with torch.no_grad():
        for u in control_seq:
            x = prepare_node_features(
                wn,
                cur_p,
                cur_c,
                u,
                node_to_index,
                pump_names,
                device,
                model,
            )
            if hasattr(model, "rnn"):
                seq_in = x.unsqueeze(0).unsqueeze(0)
                pred = model(seq_in, edge_index, edge_attr)
                if isinstance(pred, dict):
                    pred = pred.get("node_outputs")[0, 0]
            else:
                pred = model(x, edge_index, edge_attr)
                if isinstance(pred, dict):
                    pred = pred.get("node_outputs")
            if getattr(model, "y_mean", None) is not None:
                pred = pred * model.y_std + model.y_mean
            assert not torch.isnan(pred).any(), "NaN prediction"
            pred_p = pred[:, 0].tolist()
            pred_c = pred[:, 1].tolist()
            cur_p = {n: float(pred_p[i]) for i, n in enumerate(wn.node_name_list)}
            cur_c = {n: float(pred_c[i]) for i, n in enumerate(wn.node_name_list)}
    return cur_p, cur_c


def plot_single_run(df: pd.DataFrame, run_name: str) -> None:
    """Generate basic time series plots for a single MPC run."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.plot(df["time"], df["min_pressure"], label="min pressure")
    plt.xlabel("Hour")
    plt.ylabel("Pressure [m]")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_min_pressure_{run_name}.png"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(df["time"], df["min_chlorine"], label="min chlorine", color="tab:orange")
    plt.xlabel("Hour")
    plt.ylabel("Chlorine [mg/L]")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_chlorine_{run_name}.png"))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(df["time"], df["energy"], label="energy", color="tab:green")
    plt.xlabel("Hour")
    plt.ylabel("Energy [kWh]")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_energy_{run_name}.png"))
    plt.close()

    plt.figure(figsize=(8, 3))
    ctrl = np.stack(df["controls"].to_list())
    avg = ctrl.mean(axis=1)
    plt.plot(df["time"], avg, label="avg speed")
    plt.xlabel("Hour")
    plt.ylabel("Average Pump Speed")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mpc_controls_{run_name}.png"))
    plt.close()


def simulate_closed_loop(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    horizon: int,
    iterations: int,
    node_to_index: Dict[str, int],
    pump_names: List[str],
    device: torch.device,
    Pmin: float,
    Cmin: float,
    feedback_interval: int = 24,
    run_name: str = "",
) -> pd.DataFrame:
    """Run 24-hour closed-loop MPC using the surrogate for fast updates.

    EPANET is invoked only every ``feedback_interval`` hours (default once per
    day) to obtain ground-truth measurements.  All intermediate steps update the
    pressures and chlorine levels using the GNN surrogate which allows the loop
    to run nearly instantly.
    """
    expected_in_dim = 4 + len(pump_names)
    in_dim = getattr(getattr(model, "conv1", None), "in_channels", None)
    if in_dim is None or in_dim < expected_in_dim:
        raise ValueError(
            "Loaded model was trained without pump controls - rerun train_gnn.py"
        )

    log = []
    pressure_violations = 0
    chlorine_violations = 0
    total_energy = 0.0
    # obtain hydraulic state at time zero
    wn.options.time.duration = 0
    wn.options.time.report_timestep = 0
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(str(TEMP_DIR / "temp"))
    pressures = results.node["pressure"].iloc[0].to_dict()
    chlorine = results.node["quality"].iloc[0].to_dict()

    base_demands = {
        j: wn.get_node(j).demand_timeseries_list[0].base_value
        for j in wn.junction_name_list
    }
    prev_u: Optional[torch.Tensor] = None

    for hour in range(24):
        start = time.time()
        u_opt = run_mpc_step(
            wn,
            model,
            edge_index,
            edge_attr,
            pressures,
            chlorine,
            horizon,
            iterations,
            node_to_index,
            pump_names,
            device,
            Pmin,
            Cmin,
            prev_u,
        )
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

        # update demands based on patterns and random variation
        t = hour * 3600
        for j in wn.junction_name_list:
            junc = wn.get_node(j)
            mult = 1.0
            pat_name = junc.demand_timeseries_list[0].pattern_name
            if pat_name:
                mult = wn.get_pattern(pat_name).at(t)
            noise = np.random.normal(1.0, 0.05)
            junc.demand_timeseries_list[0].base_value = base_demands[j] * mult * noise

        if feedback_interval > 0 and hour % feedback_interval == 0:
            # Periodic ground truth synchronization using EPANET
            wn.options.time.start_clocktime = t
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
            assert not np.isnan(energy), "NaN energy calculation"
            end = time.time()
        else:
            # Fast surrogate-based propagation
            pressures, chlorine = propagate_with_surrogate(
                wn,
                model,
                edge_index,
                edge_attr,
                pressures,
                chlorine,
                u_opt,
                node_to_index,
                pump_names,
                device,
            )
            end = time.time()
            energy = 0.0
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
    print(f"[MPC Summary] Total pump energy used: {total_energy:.2f} kWh")
    os.makedirs(REPO_ROOT / "logs", exist_ok=True)
    with open(REPO_ROOT / "logs" / "mpc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if run_name:
        plot_single_run(df, run_name)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_path = os.path.join(REPO_ROOT, "CTown.inp")
    wn, node_to_index, pump_names, edge_index, edge_attr = load_network(
        inp_path, return_edge_attr=True
    )
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    try:
        model = load_surrogate_model(device)
    except FileNotFoundError as e:
        print(e)
        return

    expected_in_dim = 4 + len(pump_names)
    if model.conv1.in_channels != expected_in_dim:
        print(
            f"Loaded surrogate expects {model.conv1.in_channels} input features "
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
        args.horizon,
        args.iterations,
        node_to_index,
        pump_names,
        device,
        args.Pmin,
        args.Cmin,
        args.feedback_interval,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


if __name__ == "__main__":
    main()
