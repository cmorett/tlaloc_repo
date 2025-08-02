import random
import pickle
import os
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# Minimum allowed pressure [m].  Values below this threshold are clipped
# in both data generation and validation to keep preprocessing consistent.
MIN_PRESSURE = 5.0

# Resolve repository paths so all files are created inside the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
PLOTS_DIR = REPO_ROOT / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

import numpy as np
import wntr
from wntr.network.base import LinkStatus
from wntr.metrics.economic import pump_energy


def simulate_extreme_event(
    wn: wntr.network.WaterNetworkModel,
    pump_controls: Dict[str, List[float]],
    idx: int,
    event_type: str,
) -> None:
    """Apply an extreme event to the water network model in-place."""

    if event_type == "fire_flow":
        node_id = random.choice(list(wn.junction_name_list))
        pattern_name = f"{node_id}_pat_{idx}"
        pattern = wn.get_pattern(pattern_name)
        pattern.multipliers = [m * 10.0 for m in pattern.multipliers]
    elif event_type == "pump_failure":
        pump_id = random.choice(list(wn.pump_name_list))
        for h in range(len(pump_controls[pump_id])):
            pump_controls[pump_id][h] = 0.0
        link = wn.get_link(pump_id)
        link.initial_status = LinkStatus.Closed
    elif event_type == "quality_variation":
        for source in wn.source_name_list:
            # Some INP files might register a "source" that isn't actually
            # present in the node registry (WNTR returns the file name with an
            # added index like ``INP1``).  Only modify nodes that truly exist.
            if source in wn.node_name_list:
                n = wn.get_node(source)
                if hasattr(n, "initial_quality"):
                    factor = random.uniform(0.5, 1.5)
                    n.initial_quality *= factor
                    src = wn.get_source(source, None)
                    if src is not None:
                        src.strength *= factor



def _build_randomized_network(
    inp_file: str,
    idx: int,
    event_type: Optional[str] = None,
) -> Tuple[wntr.network.WaterNetworkModel, Dict[str, np.ndarray], Dict[str, List[float]]]:
    """Create a network with randomized demand patterns and pump controls.

    Pipe roughness values are kept at their defaults. If ``event_type`` is not
    ``None`` an extreme scenario such as ``fire_flow`` or ``pump_failure`` is
    injected after the randomized controls are created.
    """

    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.quality.parameter = "CHEMICAL"

    # Pipe roughness coefficients (Hazen--Williams C) remain fixed so the
    # surrogate only sees variations from demand patterns and pump controls.

    # Randomize initial tank levels while keeping values within the
    # feasible range.  Sample from a Gaussian centred on the original
    # value so typical operating conditions remain likely.
    for tname in wn.tank_name_list:
        tank = wn.get_node(tname)
        span = max(tank.max_level - tank.min_level, 1e-6)
        std = 0.1 * span
        level = random.gauss(tank.init_level, std)
        level = min(max(level, tank.min_level), tank.max_level)
        tank.init_level = level

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
        # Sample demand multipliers uniformly in [0.8, 1.2] for broader variation
        scale_factors = np.random.uniform(0.8, 1.2, size=len(multipliers))
        multipliers = multipliers * scale_factors
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
        wn.add_pattern(
            pat_name, wntr.network.elements.Pattern(pat_name, pump_controls[pn])
        )
        link.base_speed = 1.0
        link.speed_pattern_name = pat_name

    if event_type is not None:
        simulate_extreme_event(wn, pump_controls, idx, event_type)

    return wn, scale_dict, pump_controls


def _run_single_scenario(
    args,
    extreme_event_prob: float = 0.0,
) -> Optional[Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]]:
    """Run a single randomized scenario.

    EPANET occasionally fails to write results when the hydraulics become
    infeasible. To make the data generation robust we retry a few times,
    rebuilding the randomized scenario each time.  If all attempts fail
    ``None`` is returned so the caller can skip this scenario.
    """

    idx, inp_file, seed = args

    scenario_label = "normal"

    for attempt in range(3):
        if seed is not None:
            random.seed(seed + idx + attempt)
            np.random.seed(seed + idx + attempt)

        if random.random() < extreme_event_prob:
            scenario_label = random.choice(
                ["fire_flow", "pump_failure", "quality_variation"]
            )
            wn, scale_dict, pump_controls = _build_randomized_network(
                inp_file, idx, scenario_label
            )
        else:
            wn, scale_dict, pump_controls = _build_randomized_network(inp_file, idx)

        prefix = TEMP_DIR / f"temp_{os.getpid()}_{idx}_{attempt}"
        try:
            sim = wntr.sim.EpanetSimulator(wn)
            sim_results = sim.run_sim(file_prefix=str(prefix))
            sim_results.scenario_type = scenario_label
            break
        except wntr.epanet.exceptions.EpanetException:
            # Remove possible leftover files before retrying with a new
            # randomized scenario.  If we ran out of attempts, return None so
            # the caller can skip this scenario instead of failing the entire
            # generation run.
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
                return None
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


    flows = sim_results.link["flowrate"]
    heads = sim_results.node["head"]
    energy = pump_energy(flows, heads, wn)
    if energy[wn.pump_name_list].isna().any().any():
        raise ValueError("pump energy contains NaN")

    for df in [flows, heads, sim_results.node["pressure"], sim_results.node["quality"]]:
        if df.isna().any().any() or np.isinf(df.values).any():
            raise ValueError("invalid values detected in simulation results")

    return sim_results, scale_dict, pump_controls


def extract_additional_targets(sim_results: wntr.sim.results.SimulationResults, wn: wntr.network.WaterNetworkModel) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of pipe flow rates and pump energy consumption."""

    flows_df = sim_results.link["flowrate"]
    heads_df = sim_results.node["head"]
    num_links = len(wn.link_name_list)
    flows_dir = np.zeros((len(flows_df), num_links * 2), dtype=np.float64)
    for idx, link_name in enumerate(wn.link_name_list):
        col = flows_df[link_name].values.astype(np.float64)
        flows_dir[:, idx * 2] = col
        flows_dir[:, idx * 2 + 1] = -col
    flow_rates = flows_dir
    energy_df = pump_energy(flows_df[wn.pump_name_list], heads_df, wn)
    pump_energy_arr = energy_df[wn.pump_name_list].values.astype(np.float64)
    return flow_rates, pump_energy_arr


def run_scenarios(
    inp_file: str,
    num_scenarios: int,
    seed: Optional[int] = None,
    extreme_event_prob: float = 0.0,
    num_workers: Optional[int] = None,
) -> List[
    Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]
]:
    """Run multiple randomized scenarios in parallel and return their results.

    Scenarios that remain infeasible after several retries are skipped, so the
    returned list may contain fewer elements than ``num_scenarios``.
    """

    args_list = [(i, inp_file, seed) for i in range(num_scenarios)]
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    with Pool(processes=num_workers) as pool:
        func = partial(_run_single_scenario, extreme_event_prob=extreme_event_prob)
        raw_results = pool.map(func, args_list)

    # Filter out failed scenarios returned as ``None``
    results = [res for res in raw_results if res is not None]

    return results


def plot_dataset_distributions(
    demand_mults: Iterable[float],
    pump_speeds: Iterable[float],
    out_prefix: str,
    plots_dir: Path = PLOTS_DIR,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot distributions of demand multipliers and pump speeds."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    d_arr = np.asarray(list(demand_mults), dtype=float)
    p_arr = np.asarray(list(pump_speeds), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(d_arr, bins=30, color="tab:blue", alpha=0.7)
    axes[0].set_xlabel("Demand Multiplier")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Demand Distribution")

    axes[1].hist(p_arr, bins=30, color="tab:orange", alpha=0.7)
    axes[1].set_xlabel("Pump Speed")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Pump Speed Distribution")

    fig.tight_layout()
    fig.savefig(plots_dir / f"dataset_distributions_{out_prefix}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


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


def build_sequence_dataset(
    results: Iterable[
        Tuple[
            wntr.sim.results.SimulationResults,
            Dict[str, np.ndarray],
            Dict[str, List[float]],
        ]
    ],
    wn_template: wntr.network.WaterNetworkModel,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a dataset of sequences from simulation results with multi-task targets."""

    X_list: List[np.ndarray] = []
    Y_list: List[dict] = []
    scenario_types: List[str] = []

    pumps = wn_template.pump_name_list

    for sim_results, _scale_dict, pump_ctrl in results:
        scenario_types.append(getattr(sim_results, "scenario_type", "normal"))
        # Clamp to avoid unrealistically low pressures while keeping the full
        # range otherwise.  Enforce a 5 m lower bound to match downstream
        # validation logic.
        pressures = sim_results.node["pressure"].clip(lower=MIN_PRESSURE)
        quality_df = sim_results.node["quality"].clip(lower=0.0, upper=4.0)
        param = str(wn_template.options.quality.parameter).upper()
        if "CHEMICAL" in param or "CHLORINE" in param:
            # convert mg/L to g/L used by CHEMICAL or CHLORINE models before
            # taking the logarithm so the surrogate sees reasonable scales
            quality_df = quality_df / 1000.0
        quality = np.log1p(quality_df)
        demands = sim_results.node.get("demand")
        if demands is not None:
            max_d = float(demands.max().max())
            demands = demands.clip(lower=0.0, upper=max_d * 1.5)
        times = pressures.index
        flows_arr, energy_arr = extract_additional_targets(sim_results, wn_template)

        if len(times) <= seq_len:
            warnings.warn(
                "Skipping scenario with only "
                f"{len(times)} timesteps (need {seq_len + 1})"
            )
            continue

        X_seq: List[np.ndarray] = []
        node_out_seq: List[np.ndarray] = []
        edge_out_seq: List[np.ndarray] = []
        energy_seq: List[np.ndarray] = []
        for t in range(seq_len):
            pump_vector = np.array([pump_ctrl[p][t] for p in pumps], dtype=np.float64)
            feat_nodes = []
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                if node in wn_template.reservoir_name_list:
                    # Reservoir nodes report ~0 pressure from EPANET. Use their
                    # fixed hydraulic head instead so the surrogate is aware of
                    # the supply level.
                    p_t = float(wn_template.get_node(node).base_head)
                else:
                    p_t = float(pressures.iat[t, idx])
                c_t = float(quality.iat[t, idx])
                if demands is not None and node in wn_template.junction_name_list:
                    d_t = float(demands.iat[t, idx])
                else:
                    d_t = 0.0
                if node in wn_template.junction_name_list or node in wn_template.tank_name_list:
                    elev = wn_template.get_node(node).elevation
                elif node in wn_template.reservoir_name_list:
                    elev = wn_template.get_node(node).base_head
                else:
                    n = wn_template.get_node(node)
                    if getattr(n, "elevation", None) is not None:
                        elev = n.elevation
                    else:
                        elev = getattr(n, "base_head", 0.0)
                if elev is None:
                    elev = 0.0
                feat = [d_t, p_t, c_t, elev]
                feat.extend(pump_vector.tolist())
                feat_nodes.append(feat)
            X_seq.append(np.array(feat_nodes, dtype=np.float64))

            out_nodes = []
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                p_next = float(pressures.iat[t + 1, idx])
                c_next = float(quality.iat[t + 1, idx])
                out_nodes.append([max(p_next, MIN_PRESSURE), max(c_next, 0.0)])
            node_out_seq.append(np.array(out_nodes, dtype=np.float64))

            edge_out_seq.append(flows_arr[t + 1].astype(np.float64))
            energy_seq.append(energy_arr[t + 1].astype(np.float64))

        X_list.append(np.stack(X_seq))
        Y_list.append({
            "node_outputs": np.stack(node_out_seq).astype(np.float32),
            "edge_outputs": np.stack(edge_out_seq).astype(np.float32),
            "pump_energy": np.stack(energy_seq).astype(np.float32),
        })

    if not X_list:
        raise ValueError(
            f"No scenarios contained at least {seq_len + 1} timesteps"
        )

    X = np.stack(X_list).astype(np.float32)
    Y = np.array(Y_list, dtype=object)
    return X, Y, np.array(scenario_types)

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
    Y_list: List[dict] = []

    pumps = wn_template.pump_name_list

    for sim_results, _scale_dict, pump_ctrl in results:
        # Enforce a 5 m lower bound while keeping the upper range unrestricted
        # to capture extreme pressure values.
        pressures = sim_results.node["pressure"].clip(lower=MIN_PRESSURE)
        quality_df = sim_results.node["quality"].clip(lower=0.0, upper=4.0)
        param = str(wn_template.options.quality.parameter).upper()
        if "CHEMICAL" in param or "CHLORINE" in param:
            # CHEMICAL or CHLORINE quality models return mg/L, scale to g/L
            # before applying the log transform
            quality_df = quality_df / 1000.0
        quality = np.log1p(quality_df)
        demands = sim_results.node.get("demand")
        times = pressures.index
        flows_arr, energy_arr = extract_additional_targets(sim_results, wn_template)


        if demands is not None:
            max_d = float(demands.max().max())
            demands = demands.clip(lower=0.0, upper=max_d * 1.5)

        for i in range(len(times) - 1):
            pump_vector = np.array([pump_ctrl[p][i] for p in pumps], dtype=np.float64)

            feat_nodes = []
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                if node in wn_template.reservoir_name_list:
                    # Use the reservoir's constant head as the pressure input
                    p_t = float(wn_template.get_node(node).base_head)
                else:
                    p_t = max(pressures.iat[i, idx], MIN_PRESSURE)
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
                    if getattr(n, "elevation", None) is not None:
                        elev = n.elevation
                    else:
                        elev = getattr(n, "base_head", 0.0)
                if elev is None:
                    elev = 0.0

                feat = [d_t, p_t, c_t, elev]
                feat.extend(pump_vector.tolist())
                feat_nodes.append(feat)
            X_list.append(np.array(feat_nodes, dtype=np.float64))

            out_nodes = []
            for node in wn_template.node_name_list:
                idx = pressures.columns.get_loc(node)
                p_next = max(pressures.iat[i + 1, idx], MIN_PRESSURE)
                c_next = max(quality.iat[i + 1, idx], 0.0)
                out_nodes.append([p_next, c_next])
            Y_list.append({
                "node_outputs": np.array(out_nodes, dtype=np.float32),
                "edge_outputs": flows_arr[i + 1].astype(np.float32),
                "pump_energy": energy_arr[i + 1].astype(np.float32),
            })

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    X = np.stack(X_list).astype(np.float32)
    Y = np.array(Y_list, dtype=object)
    return X, Y


def build_edge_index(
    wn: wntr.network.WaterNetworkModel,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed edge index, edge attributes and edge types."""

    node_index_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    edges: List[List[int]] = []
    attrs: List[List[float]] = []
    types: List[int] = []
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i1 = node_index_map[link.start_node.name]
        i2 = node_index_map[link.end_node.name]
        edges.append([i1, i2])
        edges.append([i2, i1])
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
        types.extend([t, t])

    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.array(attrs, dtype=np.float32)
    # log-normalize roughness then scale all features to [0,1]
    edge_attr[:, 2] = np.log1p(edge_attr[:, 2])
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)

    assert edge_index.shape[0] == 2
    edge_type = np.array(types, dtype=np.int64)
    return edge_index, edge_attr, edge_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=2000,
        help="Number of random scenarios to simulate",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--extreme-event-prob",
        type=float,
        default=0.2,
        help="Probability of injecting an extreme scenario in each run",
    )
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        help="Directory to store generated datasets",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers for scenario generation (defaults to CPU count)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1,
        help="Length of sequences to store in the dataset (1 for single-step)",
    )
    args = parser.parse_args()

    inp_file = REPO_ROOT / "CTown.inp"
    N = args.num_scenarios

    results = run_scenarios(
        str(inp_file),
        N,
        seed=args.seed,
        extreme_event_prob=args.extreme_event_prob,
        num_workers=args.num_workers,
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    demand_mults: List[float] = []
    pump_speeds: List[float] = []
    for _sim, scale_dict, pump_ctrl in results:
        for arr in scale_dict.values():
            demand_mults.extend(arr.ravel().tolist())
        for speeds in pump_ctrl.values():
            pump_speeds.extend(list(speeds))

    plot_dataset_distributions(demand_mults, pump_speeds, run_ts)
    train_res, val_res, test_res = split_results(results)

    wn_template = wntr.network.WaterNetworkModel(str(inp_file))
    if args.sequence_length > 1:
        X_train, Y_train, train_labels = build_sequence_dataset(
            train_res, wn_template, args.sequence_length
        )
        X_val, Y_val, val_labels = build_sequence_dataset(
            val_res, wn_template, args.sequence_length
        )
        X_test, Y_test, test_labels = build_sequence_dataset(
            test_res, wn_template, args.sequence_length
        )
    else:
        X_train, Y_train = build_dataset(train_res, wn_template)
        X_val, Y_val = build_dataset(val_res, wn_template)
        X_test, Y_test = build_dataset(test_res, wn_template)

    edge_index, edge_attr, edge_type = build_edge_index(wn_template)

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "Y_val.npy"), Y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "Y_test.npy"), Y_test)
    if args.sequence_length > 1:
        np.save(os.path.join(out_dir, "scenario_train.npy"), train_labels)
        np.save(os.path.join(out_dir, "scenario_val.npy"), val_labels)
        np.save(os.path.join(out_dir, "scenario_test.npy"), test_labels)
    np.save(os.path.join(out_dir, "edge_index.npy"), edge_index)
    np.save(os.path.join(out_dir, "edge_attr.npy"), edge_attr)
    np.save(os.path.join(out_dir, "edge_type.npy"), edge_type)

    with open(os.path.join(out_dir, "train_results_list.pkl"), "wb") as f:
        pickle.dump(train_res, f)
    with open(os.path.join(out_dir, "val_results_list.pkl"), "wb") as f:
        pickle.dump(val_res, f)
    with open(os.path.join(out_dir, "test_results_list.pkl"), "wb") as f:
        pickle.dump(test_res, f)


if __name__ == "__main__":
    main()
