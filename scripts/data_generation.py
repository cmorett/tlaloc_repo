import random
import pickle
import os
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# Resolve repository paths so all files are created inside the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

import numpy as np
import wntr
from wntr.network.base import LinkStatus
from wntr.metrics.economic import pump_energy


def run_scenarios(
    inp_file: str, num_scenarios: int, seed: Optional[int] = None
) -> List[Tuple[wntr.sim.results.SimulationResults, Dict[str, float]]]:
    """Run a collection of randomized scenarios and return each result along
    with the demand scaling applied to every junction."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results: List[Tuple[wntr.sim.results.SimulationResults, Dict[str, float]]] = []

    for _ in range(num_scenarios):
        # Create a fresh copy of the water network for each scenario
        wn = wntr.network.WaterNetworkModel(inp_file)

        # Configure simulation options
        wn.options.time.duration = 24 * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.quality_timestep = 3600
        wn.options.time.report_timestep = 3600
        wn.options.quality.parameter = "CHEMICAL"

        # Randomly scale base demand at each junction
        scale_dict: Dict[str, float] = {}
        for jname in wn.junction_name_list:
            base = wn.get_node(jname).demand_timeseries_list[0].base_value
            scale = random.uniform(0.5, 1.5)
            wn.get_node(jname).demand_timeseries_list[0].base_value = base * scale
            scale_dict[jname] = scale

        # Open all pumps and then randomly close ~30%
        for pn in wn.pump_name_list:
            link = wn.get_link(pn)
            link.initial_status = LinkStatus.Open
            # Sample a random pump speed in [0, 1] so the surrogate is trained
            # on a wide range of control values rather than only binary
            # open/closed states.
            link.base_speed = random.uniform(0.0, 1.0)
        pumps_to_off = random.sample(
            wn.pump_name_list, max(1, int(0.3 * len(wn.pump_name_list)))
        )
        for pn in pumps_to_off:
            link = wn.get_link(pn)
            link.initial_status = LinkStatus.Closed
            link.base_speed = 0.0

        sim = wntr.sim.EpanetSimulator(wn)
        sim_result = sim.run_sim(str(TEMP_DIR / "temp"))
        # Verify energy calculation does not produce NaNs
        energy = pump_energy(
            sim_result.link["flowrate"][wn.pump_name_list],
            sim_result.node["head"],
            wn,
        )
        if energy[wn.pump_name_list].isna().any().any():
            raise ValueError("pump energy contains NaN")
        results.append((sim_result, scale_dict))

        # Ensure pumps are reopened (requirement 2)
        for pn in wn.pump_name_list:
            wn.get_link(pn).initial_status = LinkStatus.Open

    return results


def split_results(
    results: List[Tuple[wntr.sim.results.SimulationResults, Dict[str, float]]]
) -> Tuple[List, List, List]:
    num_total = len(results)
    indices = np.random.permutation(num_total)
    n_train = int(0.7 * num_total)
    n_val = int(0.15 * num_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_results = [results[i] for i in train_idx]
    val_results = [results[i] for i in val_idx]
    test_results = [results[i] for i in test_idx]
    return train_results, val_results, test_results


def build_dataset(
    results: Iterable[Tuple[wntr.sim.results.SimulationResults, Dict[str, float]]],
    wn_template: wntr.network.WaterNetworkModel,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for sim_results, demand_scale in results:
        pressures = sim_results.node["pressure"]
        quality = sim_results.node["quality"]
        pressure_array = pressures.values
        quality_array = quality.values
        times = pressures.index

        pump_status = sim_results.link["setting"][wn_template.pump_name_list].values

        for i in range(len(times) - 1):
            feat_nodes = []
            controls = pump_status[i]
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                p_t = pressure_array[i, idx]
                c_t = quality_array[i, idx]
                # clip negative values which arise from unrealistic scenarios
                # (e.g., pumps being disabled). Negative pressures are not
                # physically meaningful and would cause training to fail.
                if p_t < 0:
                    p_t = 0.0
                if c_t < 0:
                    c_t = 0.0

                if node in wn_template.junction_name_list:
                    base_base = wn_template.get_node(node).demand_timeseries_list[0].base_value
                    scale = demand_scale.get(node, 1.0)
                    base_d = base_base * scale
                else:
                    base_d = 0.0

                if node in wn_template.junction_name_list or node in wn_template.tank_name_list:
                    elev = wn_template.get_node(node).elevation
                elif node in wn_template.reservoir_name_list:
                    # ``Reservoir`` objects expose ``head`` as ``None`` and store
                    # the hydraulic head in ``base_head``. Using ``head`` directly
                    # produced ``NaN`` values in the dataset which later led to
                    # ``NaN`` training loss.
                    elev = wn_template.get_node(node).base_head
                else:
                    elev = wn_template.get_node(node).head

                if elev is None:
                    elev = 0.0

                feat = [base_d, p_t, c_t, elev]
                feat.extend(controls.tolist())
                feat_nodes.append(feat)
            X_sample = np.array(feat_nodes, dtype=np.float64)
            X_list.append(X_sample)

            out_nodes = []
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                p_next = pressure_array[i + 1, idx]
                c_next = quality_array[i + 1, idx]
                # apply the same clipping to the labels
                if p_next < 0:
                    p_next = 0.0
                if c_next < 0:
                    c_next = 0.0
                out_nodes.append([p_next, c_next])
            Y_sample = np.array(out_nodes, dtype=np.float64)
            Y_list.append(Y_sample)

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    return X, Y


def build_edge_index(wn: wntr.network.WaterNetworkModel) -> np.ndarray:
    node_index_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    edge_index = []
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i1 = node_index_map[link.start_node.name]
        i2 = node_index_map[link.end_node.name]
        edge_index.append([i1, i2])
        edge_index.append([i2, i1])
    edge_index = np.array(edge_index, dtype=np.int64).T
    assert edge_index.shape[0] == 2
    return edge_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=10,
        help="Number of random scenarios to simulate",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        help="Directory to store generated datasets",
    )
    args = parser.parse_args()

    inp_file = "CTown.inp"
    N = args.num_scenarios

    results = run_scenarios(inp_file, N, seed=args.seed)
    train_res, val_res, test_res = split_results(results)

    wn_template = wntr.network.WaterNetworkModel(inp_file)
    X_train, Y_train = build_dataset(train_res, wn_template)
    X_val, Y_val = build_dataset(val_res, wn_template)
    X_test, Y_test = build_dataset(test_res, wn_template)

    edge_index = build_edge_index(wn_template)

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "Y_val.npy"), Y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "Y_test.npy"), Y_test)
    np.save(os.path.join(out_dir, "edge_index.npy"), edge_index)

    with open(os.path.join(out_dir, "train_results_list.pkl"), "wb") as f:
        pickle.dump(train_res, f)
    with open(os.path.join(out_dir, "val_results_list.pkl"), "wb") as f:
        pickle.dump(val_res, f)
    with open(os.path.join(out_dir, "test_results_list.pkl"), "wb") as f:
        pickle.dump(test_res, f)


if __name__ == "__main__":
    main()
