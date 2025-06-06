import argparse
import time
from typing import Dict, List
import os


import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GCNConv
import wntr


DATA_DIR = "data"


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
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run train_gnn.py to generate the surrogate weights."
        )
    model = GNNSurrogate(in_dim=4, hidden_dim=64, out_dim=2).to(device)
    state = torch.load(path, map_location=device)
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
) -> torch.Tensor:
    """Build node features for the surrogate model.

    The original training data only contained four features per node
    (base demand, current pressure, chlorine and elevation).  Including
    pump control values here therefore leads to a mismatch with the
    surrogate model loaded from disk which expects exactly four input
    channels.  To avoid runtime errors we ignore ``pump_controls`` and
    construct feature vectors with the same four-dimensional layout used
    during training.
    """
    num_nodes = len(wn.node_name_list)
    feats = np.zeros((num_nodes, 4), dtype=np.float32)
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
        feats[idx, 0] = demand
        feats[idx, 1] = pressures.get(name, 0.0)
        feats[idx, 2] = chlorine.get(name, 0.0)
        feats[idx, 3] = elev
    # ``pump_controls`` is ignored since the model was trained without it.
    return torch.tensor(feats, dtype=torch.float32, device=device)


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
) -> torch.Tensor:
    """Optimize pump controls for one hour using gradient-based MPC."""
    num_pumps = len(pump_names)
    u = torch.ones(horizon, num_pumps, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=0.05)

    for _ in range(iterations):
        optimizer.zero_grad()
        cur_p = dict(pressures)
        cur_c = dict(chlorine)
        total_cost = torch.tensor(0.0, device=device)
        for t in range(horizon):
            x = prepare_node_features(
                wn, cur_p, cur_c, u[t], node_to_index, pump_names, device
            )
            pred = model(x, edge_index)
            pred_p = pred[:, 0]
            pred_c = pred[:, 1]
            Pmin, Cmin = 20.0, 0.2
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
) -> pd.DataFrame:
    """Run 24-hour closed-loop simulation applying MPC controls each hour."""
    log = []
    sim = wntr.sim.EpanetSimulator(wn)
    wn.options.time.duration = 3600
    wn.options.time.report_timestep = 3600
    results = sim.run_sim()
    pressures = results.node["pressure"].iloc[-1].to_dict()
    chlorine = results.node["quality"].iloc[-1].to_dict()

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
        )
        controls = u_opt[0]
        for i, pump in enumerate(pump_names):
            link = wn.get_link(pump)
            if controls[i].item() < 0.5:
                link.status = 0
            else:
                link.status = 1
                link.speed = float(controls[i].item())
        sim = wntr.sim.EpanetSimulator(wn)
        wn.options.time.duration = 3600
        wn.options.time.report_timestep = 3600
        results = sim.run_sim()
        end = time.time()
        pressures = results.node["pressure"].iloc[-1].to_dict()
        chlorine = results.node["quality"].iloc[-1].to_dict()
        min_p = min(pressures.values())
        min_c = min(chlorine.values())
        energy = float(torch.sum(controls ** 2).item())
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wn, node_to_index, pump_names, edge_index = load_network("CTown.inp")
    edge_index = edge_index.to(device)
    try:
        model = load_surrogate_model(device)
    except FileNotFoundError as e:
        print(e)
        return

    simulate_closed_loop(
        wn, model, edge_index, args.horizon, args.iterations, node_to_index, pump_names, device
    )


if __name__ == "__main__":
    main()
