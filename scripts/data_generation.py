import random
import pickle
import os
import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import warnings

# Resolve repository paths so all files are created inside the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

import numpy as np
import wntr
from wntr.network.base import LinkStatus
from wntr.metrics.economic import pump_energy



def _build_randomized_network(inp_file: str, idx: int) -> Tuple[wntr.network.WaterNetworkModel, Dict[str, np.ndarray], Dict[str, List[float]]]:
    """Create a network with randomized demand patterns and pump controls."""

    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.quality.parameter = "CHEMICAL"

    hours = int(wn.options.time.duration // wn.options.time.hydraulic_timestep)

    scale_dict: Dict[str, np.ndarray] = {}
    for jname in wn.junction_name_list:
        node = wn.get_node(jname)
        ts = node.demand_timeseries_list[0]
        if ts.pattern is None:
            base_mult = np.ones(hours, dtype=float)
        else:
            base_mult = np.array(ts.pattern.multipliers, dtype=float)
        multipliers = base_mult.copy()
        noise = 1.0 + np.random.normal(0.0, 0.05, size=len(multipliers))
        multipliers = multipliers * noise
        multipliers = np.clip(multipliers, a_min=0.0, a_max=None)
        pat_name = f"{jname}_pat_{idx}"
        wn.add_pattern(pat_name, wntr.network.elements.Pattern(pat_name, multipliers))
        ts.pattern_name = pat_name
        scale_dict[jname] = multipliers

    pump_controls: Dict[str, List[float]] = {pn: [] for pn in wn.pump_name_list}

    for _h in range(hours):
        for pn in wn.pump_name_list:
            spd = random.uniform(0.0, 1.0)
            if random.random() < 0.3:
                spd = 0.0
            pump_controls[pn].append(spd)

        if all(pump_controls[pn][-1] == 0.0 for pn in wn.pump_name_list):
            keep_on = random.choice(wn.pump_name_list)
            pump_controls[keep_on][-1] = random.uniform(0.5, 1.0)

    for pn in wn.pump_name_list:
        link = wn.get_link(pn)
        link.initial_status = LinkStatus.Open
        pat_name = f"{pn}_pat_{idx}"
        wn.add_pattern(pat_name, wntr.network.elements.Pattern(pat_name, pump_controls[pn]))
        link.base_speed = 1.0
        link.speed_pattern_name = pat_name

    return wn, scale_dict, pump_controls


def _run_single_scenario(args) -> Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]:
    """Run a single randomized scenario.

    EPANET occasionally fails to write results when the hydraulics become
    infeasible. To make the data generation robust we retry a few times,
    rebuilding the randomized scenario each time.
    """

    idx, inp_file, seed = args

    for attempt in range(3):
        if seed is not None:
            random.seed(seed + idx + attempt)
            np.random.seed(seed + idx + attempt)

        wn, scale_dict, pump_controls = _build_randomized_network(inp_file, idx)

        prefix = TEMP_DIR / f"temp_{os.getpid()}_{idx}_{attempt}"
        try:
            sim = wntr.sim.EpanetSimulator(wn)
            sim_results = sim.run_sim(file_prefix=str(prefix))
            break
        except wntr.epanet.exceptions.EpanetException:
            # Remove possible leftover files before retrying with a new
            # randomized scenario.  If we ran out of attempts, re-raise.
            for ext in [
                ".inp",
                ".rpt",
                ".bin",
                ".hyd",
                ".msx",
                ".msx-rpt",
                ".msx-bin",
                ".check.msx",
            ]:
                try:
                    os.remove(f"{prefix}{ext}")
                except FileNotFoundError:
                    pass
                except PermissionError:
                    warnings.warn(f"Could not remove file {prefix}{ext}")
            if attempt == 2:
                raise
            else:
                continue

    # Clean up temp files from the successful attempt
    for ext in [
        ".inp",
        ".rpt",
        ".bin",
        ".hyd",
        ".msx",
        ".msx-rpt",
        ".msx-bin",
        ".check.msx",
    ]:
        f = f"{prefix}{ext}"
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
        except PermissionError:
            warnings.warn(f"Could not remove file {f}")

    for ext in [".inp", ".rpt", ".bin", ".hyd", ".msx", ".msx-rpt", ".msx-bin", ".check.msx"]:
        f = f"{prefix}{ext}"
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
        except PermissionError:
            warnings.warn(f"Could not remove file {f}")

    flows = sim_results.link["flowrate"]
    heads = sim_results.node["head"]
    energy = pump_energy(flows, heads, wn)
    if energy[wn.pump_name_list].isna().any().any():
        raise ValueError("pump energy contains NaN")

    for df in [flows, heads, sim_results.node["pressure"], sim_results.node["quality"]]:
        if df.isna().any().any() or np.isinf(df.values).any():
            raise ValueError("invalid values detected in simulation results")

    return sim_results, scale_dict, pump_controls


def run_scenarios(
    inp_file: str, num_scenarios: int, seed: Optional[int] = None
) -> List[Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]]:
    """Run multiple randomized scenarios and return their results."""

    args_list = [(i, inp_file, seed) for i in range(num_scenarios)]
    results = [_run_single_scenario(a) for a in args_list]

    return results


def split_results(
    results: List[
        Tuple[
            wntr.sim.results.SimulationResults,
            Dict[str, np.ndarray],
            Dict[str, List[float]],
        ]
    ]
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
    results: Iterable[
        Tuple[
            wntr.sim.results.SimulationResults,
            Dict[str, np.ndarray],
            Dict[str, List[float]],
        ]
    ],
    wn_template: wntr.network.WaterNetworkModel,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    pumps = wn_template.pump_name_list

    for sim_results, _scale_dict, pump_ctrl in results:
        pressures = sim_results.node["pressure"]
        quality = sim_results.node["quality"]
        demands = sim_results.node.get("demand")
        times = pressures.index

        pressures = pressures.clip(lower=0.0)
        quality = quality.clip(lower=0.0)
        if demands is not None:
            demands = demands.clip(lower=0.0)

        for i in range(len(times) - 1):
            pump_vector = np.array([pump_ctrl[p][i] for p in pumps], dtype=np.float64)

            feat_nodes = []
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                p_t = max(pressures.iat[i, idx], 0.0)
                c_t = max(quality.iat[i, idx], 0.0)
                if demands is not None and node in wn_template.junction_name_list:
                    d_t = demands.iat[i, idx]
                else:
                    d_t = 0.0

                if node in wn_template.junction_name_list or node in wn_template.tank_name_list:
                    elev = wn_template.get_node(node).elevation
                elif node in wn_template.reservoir_name_list:
                    elev = wn_template.get_node(node).base_head
                else:
                    n = wn_template.get_node(node)
                    elev = getattr(n, "elevation", None) or getattr(n, "base_head", 0.0)
                if elev is None:
                    elev = 0.0

                feat = [d_t, p_t, c_t, elev]
                feat.extend(pump_vector.tolist())
                feat_nodes.append(feat)
            X_list.append(np.array(feat_nodes, dtype=np.float64))

            out_nodes = []
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                p_next = max(pressures.iat[i + 1, idx], 0.0)
                c_next = max(quality.iat[i + 1, idx], 0.0)
                out_nodes.append([p_next, c_next])
            Y_list.append(np.array(out_nodes, dtype=np.float64))

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    Y = np.clip(Y, a_min=0.0, a_max=None)
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

    inp_file = REPO_ROOT / "CTown.inp"
    N = args.num_scenarios

    results = run_scenarios(str(inp_file), N, seed=args.seed)
    train_res, val_res, test_res = split_results(results)

    wn_template = wntr.network.WaterNetworkModel(str(inp_file))
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
