import argparse
import time
from typing import Dict, List, Optional
import os
from pathlib import Path


import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GCNConv
import wntr


# Resolve the repository root so files are written relative to the project
# instead of the current working directory.  This avoids permission errors
# when the script is executed from another location.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


class GNNSurrogate(torch.nn.Module):
    """Two-layer GCN surrogate used for one-hour pressure/quality prediction."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def load_network(inp_file: str):
    """Load EPANET network and build edge index for PyG."""
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.quality.parameter = "CHEMICAL"
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600

    node_to_index = {n: i for i, n in enumerate(wn.node_name_list)}
    edges = []
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_to_index[link.start_node.name]
        j = node_to_index[link.end_node.name]
        edges.append([i, j])
        edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return wn, node_to_index, wn.pump_name_list, edge_index



def load_surrogate_model(device: torch.device, path: str = "models/gnn_surrogate.pth") -> GNNSurrogate:
    """Load trained GNN surrogate weights.

    Parameters
    ----------
    device : torch.device
        Device to map the model to.
    path : str, optional
        Location of the saved state dict, by default ``models/gnn_surrogate.pth``.

    Returns
    -------
    GNNSurrogate
        Loaded surrogate model set to eval mode.
    """
    # Resolve the path relative to the repository root so the script can be
    # executed from any working directory.
    full_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"{full_path} not found. Run train_gnn.py to generate the surrogate weights."
        )
    state = torch.load(full_path, map_location=device)
    weight_key = "conv1.weight" if "conv1.weight" in state else "conv1.lin.weight"
    in_dim = state[weight_key].shape[1]
    model = GNNSurrogate(in_dim=in_dim, hidden_dim=64, out_dim=2).to(device)
    model.load_state_dict(state)
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
    in_dim: int,
) -> torch.Tensor:
    """Build node features for the surrogate model.

    The function always constructs the full set of features, including pump
    control inputs.  However, the loaded surrogate model might have been trained
    without pump controls.  To accomodate both cases, the returned tensor is
    truncated to ``in_dim`` columns.
    """

    num_nodes = len(wn.node_name_list)
    num_pumps = len(pump_names)
    feats = np.zeros((num_nodes, 4 + num_pumps), dtype=np.float32)
    for name, idx in node_to_index.items():
        node = wn.get_node(name)
        if name in wn.junction_name_list:
            demand = node.demand_timeseries_list[0].base_value
        else:
            demand = 0.0
        if name in wn.junction_name_list or name in wn.tank_name_list:
            elev = node.elevation
        else:
            elev = node.head
        base = [demand, pressures.get(name, 0.0), chlorine.get(name, 0.0), elev]
        base.extend(pump_controls.tolist())
        feats[idx] = np.array(base, dtype=np.float32)

    return torch.tensor(feats[:, :in_dim], dtype=torch.float32, device=device)


def run_mpc_step(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
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
    u = init.clone().to(device).requires_grad_()
    optimizer = torch.optim.Adam([u], lr=0.05)

    for _ in range(iterations):
        optimizer.zero_grad()
        cur_p = dict(pressures)
        cur_c = dict(chlorine)
        total_cost = torch.tensor(0.0, device=device)
        for t in range(horizon):
            x = prepare_node_features(
                wn,
                cur_p,
                cur_c,
                u[t],
                node_to_index,
                pump_names,
                device,
                model.conv1.in_channels,
            )
            pred = model(x, edge_index)
            pred_p = pred[:, 0]
            pred_c = pred[:, 1]
            w_p, w_c, w_e = 1.0, 1.0, 0.1
            psf = torch.clamp(Pmin - pred_p, min=0.0)
            csf = torch.clamp(Cmin - pred_c, min=0.0)
            cost = w_p * torch.sum(psf ** 2) + w_c * torch.sum(csf ** 2)
            cost = cost + w_e * torch.sum(u[t] ** 2)
            total_cost = total_cost + cost
            # update state dictionaries
            for idx, name in enumerate(wn.node_name_list):
                cur_p[name] = pred_p[idx].item()
                cur_c[name] = pred_c[idx].item()
        total_cost.backward()
        optimizer.step()
        with torch.no_grad():
            u.data.clamp_(0.0, 1.0)
    return u.detach()


def simulate_closed_loop(
    wn: wntr.network.WaterNetworkModel,
    model: GNNSurrogate,
    edge_index: torch.Tensor,
    horizon: int,
    iterations: int,
    node_to_index: Dict[str, int],
    pump_names: List[str],
    device: torch.device,
    Pmin: float,
    Cmin: float,
) -> pd.DataFrame:
    """Run 24-hour closed-loop simulation applying MPC controls each hour."""
    log = []
    # obtain hydraulic state at time zero
    wn.options.time.duration = 0
    wn.options.time.report_timestep = 0
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
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
        controls = u_opt[0]
        for i, pump in enumerate(pump_names):
            link = wn.get_link(pump)
            if controls[i].item() < 0.5:
                link.initial_status = wntr.network.base.LinkStatus.Closed
            else:
                link.initial_status = wntr.network.base.LinkStatus.Open
                link.base_speed = float(controls[i].item())

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

        wn.options.time.start_clocktime = t
        wn.options.time.duration = 3600
        wn.options.time.report_timestep = 3600
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        end = time.time()
        pressures = results.node["pressure"].iloc[-1].to_dict()
        chlorine = results.node["quality"].iloc[-1].to_dict()
        heads = results.node["head"].iloc[-1].to_dict()

        # update network state for next iteration
        for name in wn.tank_name_list:
            tank = wn.get_node(name)
            tank.init_level = heads[name] - tank.elevation
            tank.initial_quality = chlorine[name]
        for name in wn.reservoir_name_list:
            res = wn.get_node(name)
            res.base_head = heads[name]
            res.initial_quality = chlorine[name]
        for name in wn.junction_name_list:
            wn.get_node(name).initial_quality = chlorine[name]

        flows = results.link["flowrate"].iloc[-1]
        headloss = results.link["headloss"].iloc[-1]
        energy = float(sum(9.81 * abs(headloss[p]) * flows[p] for p in pump_names))
        min_p = min(pressures.values())
        min_c = min(chlorine.values())
        log.append(
            {
                "time": hour,
                "min_pressure": min_p,
                "min_chlorine": min_c,
                "energy": energy,
                "runtime_sec": end - start,
            }
        )
        print(
            f"Hour {hour}: minP={min_p:.2f}, minC={min_c:.3f}, energy={energy:.2f}, runtime={end-start:.2f}s"
        )
    df = pd.DataFrame(log)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "mpc_history.csv"), index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6, help="MPC horizon length")
    parser.add_argument(
        "--iterations", type=int, default=50, help="Gradient descent iterations"
    )
    parser.add_argument("--Pmin", type=float, default=20.0, help="Pressure threshold")
    parser.add_argument("--Cmin", type=float, default=0.2, help="Chlorine threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_path = os.path.join(REPO_ROOT, "CTown.inp")
    wn, node_to_index, pump_names, edge_index = load_network(inp_path)
    edge_index = edge_index.to(device)
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
        args.horizon,
        args.iterations,
        node_to_index,
        pump_names,
        device,
        args.Pmin,
        args.Cmin,
    )


if __name__ == "__main__":
    main()
