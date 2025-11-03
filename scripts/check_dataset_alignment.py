"""Audit sequence dataset alignment and pump feature consistency.

This utility cross-checks the raw EPANET simulation outputs saved during
data generation against the numpy arrays consumed by training.  It verifies
that node ordering matches ``CTown.inp``, compares per-timestep pressures in
``X_<split>.npy`` and ``Y_<split>.npy`` with the simulator results, and
confirms that pump-related edge attributes encode the same commands, head
rise and unit head loss that EPANET produced.  The script also reports extra
per-pump feature blocks that are present in the node feature tensors but not
documented in ``manifest.json``.

Example usage:

```
python scripts/check_dataset_alignment.py --data-dir data/seq96_head_15m
```
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wntr

MIN_PRESSURE = 0.0


@dataclass
class PumpEdgeMeta:
    forward_idx: int
    reverse_idx: int
    start_node_idx: int
    end_node_idx: int
    length_m: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing X_<split>.npy, manifest.json and *_results_list.pkl",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to inspect",
    )
    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Only inspect the first N scenarios per split (defaults to all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs"),
        help="Directory where CSV/JSON reports will be written",
    )
    return parser.parse_args()


def _load_split_array(data_dir: Path, prefix: str, split: str, **kwargs):
    path = data_dir / f"{prefix}_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return np.load(path, **kwargs)


def _iter_valid_results(
    results: Sequence[
        Tuple[
            wntr.sim.results.SimulationResults,
            Dict[str, np.ndarray],
            Mapping[str, Sequence[float]],
        ]
    ],
    seq_len: int,
) -> Iterator[
    Tuple[int, Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Mapping[str, Sequence[float]]]]
]:
    """Yield ``(dataset_idx, entry)`` for scenarios retained in the dataset."""
    dataset_idx = 0
    for entry in results:
        sim_results = entry[0]
        times = sim_results.node["pressure"].index
        if seq_len > 1 and len(times) <= seq_len:
            continue
        yield dataset_idx, entry
        dataset_idx += 1


def _build_pump_edge_lookup(
    wn: wntr.network.WaterNetworkModel,
    edge_index: np.ndarray,
) -> Dict[str, PumpEdgeMeta]:
    node_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    lookup: Dict[str, PumpEdgeMeta] = {}
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        start_node = pump.start_node.name if hasattr(pump.start_node, "name") else pump.start_node
        end_node = pump.end_node.name if hasattr(pump.end_node, "name") else pump.end_node
        start_idx = node_map[start_node]
        end_idx = node_map[end_node]
        mask_fwd = np.where((edge_index[0] == start_idx) & (edge_index[1] == end_idx))[0]
        mask_rev = np.where((edge_index[0] == end_idx) & (edge_index[1] == start_idx))[0]
        if mask_fwd.size == 0 or mask_rev.size == 0:
            raise ValueError(f"Pump {pump_name} not found in directed edge_index array")
        lookup[pump_name] = PumpEdgeMeta(
            forward_idx=int(mask_fwd[0]),
            reverse_idx=int(mask_rev[0]),
            start_node_idx=start_idx,
            end_node_idx=end_idx,
            length_m=float(getattr(pump, "length", 0.0) or 0.0),
        )
    return lookup


def _ensure_node_order(data_dir: Path, wn: wntr.network.WaterNetworkModel) -> None:
    node_path = data_dir / "node_names.npy"
    if not node_path.exists():
        raise FileNotFoundError(f"Missing node ordering file at {node_path}")
    node_names = np.load(node_path, allow_pickle=True).tolist()
    if node_names != wn.node_name_list:
        raise ValueError(
            "Node ordering mismatch: node_names.npy does not match CTown.inp. "
            "Re-run data generation to refresh the dataset."
        )


def _compute_duplicate_block_stats(
    X: np.ndarray,
    pump_offset: int,
    num_pumps: int,
    documented_blocks: int,
) -> Optional[Dict[str, float]]:
    if num_pumps == 0:
        return None
    feature_dim = X.shape[-1]
    total_documented = pump_offset + num_pumps * documented_blocks
    if feature_dim <= total_documented:
        return None
    remaining = feature_dim - total_documented
    extra_blocks = remaining // num_pumps
    if extra_blocks <= 0:
        return None
    block_start = pump_offset + num_pumps * documented_blocks
    extra_slice = X[..., block_start : block_start + num_pumps]
    head_slice = X[..., pump_offset + num_pumps : pump_offset + 2 * num_pumps]
    diff = np.abs(extra_slice - head_slice)
    return {
        "extra_blocks": float(extra_blocks),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "documented_blocks": float(documented_blocks),
    }


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_dir}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    wn = wntr.network.WaterNetworkModel(str(Path(__file__).resolve().parents[1] / "CTown.inp"))
    _ensure_node_order(data_dir, wn)

    feature_layout: List[str] = manifest.get("node_feature_layout", [])
    pump_names: List[str] = manifest.get("pump_names", [])
    tank_names: List[str] = manifest.get("tank_names", [])
    pump_feature_repeats = int(manifest.get("pump_feature_repeats", 0))
    documented_blocks = pump_feature_repeats if pump_feature_repeats > 0 else (1 if pump_names else 0)
    if pump_names:
        first_pump = pump_names[0]
        try:
            pump_offset = feature_layout.index(f"pump_speed_cmd_{first_pump}")
        except ValueError as exc:  # pragma: no cover - manifest corruption is unexpected
            raise ValueError("Could not locate pump features in manifest layout") from exc
    else:
        pump_offset = 0

    X_any = None
    for split in args.splits:
        X_split = _load_split_array(data_dir, "X", split)
        if X_any is None:
            X_any = X_split
    if X_any is None:
        raise RuntimeError("No dataset arrays were loaded")
    feature_dim = X_any.shape[-1]
    documented_dim = len(feature_layout) if feature_layout else feature_dim
    expected_total = pump_offset + len(pump_names) * documented_blocks + len(tank_names)
    duplicate_stats = _compute_duplicate_block_stats(
        X_any,
        pump_offset=pump_offset,
        num_pumps=len(pump_names),
        documented_blocks=documented_blocks,
    )

    edge_index = np.load(data_dir / "edge_index.npy")
    pump_meta = _build_pump_edge_lookup(wn, edge_index)
    edge_dynamic_cols: Dict[str, int] = manifest.get("edge_dynamic_columns", {})
    speed_col = edge_dynamic_cols.get("pump_speed")
    head_col = edge_dynamic_cols.get("pump_head_rise")
    unit_col = edge_dynamic_cols.get("unit_head_loss")
    if speed_col is None or head_col is None or unit_col is None:
        raise ValueError("Manifest is missing pump dynamic column indices")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"dataset_alignment_{timestamp}"

    timestep_records: List[Dict[str, object]] = []
    pump_records: List[Dict[str, object]] = []

    summary = {
        "data_dir": str(data_dir),
        "feature_dim": feature_dim,
        "documented_feature_dim": documented_dim,
        "pump_offset": pump_offset,
        "num_pumps": len(pump_names),
        "documented_pump_blocks": documented_blocks,
        "expected_total_feature_dim": expected_total,
        "tank_features": len(tank_names),
    }
    if duplicate_stats is not None:
        summary["extra_pump_feature_blocks"] = duplicate_stats

    node_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    reservoir_mask = np.zeros(len(wn.node_name_list), dtype=bool)
    reservoir_base = np.zeros(len(wn.node_name_list), dtype=float)
    for name in wn.reservoir_name_list:
        idx = node_map[name]
        reservoir_mask[idx] = True
        reservoir_base[idx] = float(getattr(wn.get_node(name), "base_head", 0.0))

    for split in args.splits:
        X_split = _load_split_array(data_dir, "X", split)
        Y_split = _load_split_array(data_dir, "Y", split, allow_pickle=True)
        seq_mode = X_split.ndim == 4
        if not seq_mode:
            raise ValueError("Alignment checks currently expect sequence datasets (ndim == 4)")
        edge_attr_seq_path = data_dir / f"edge_attr_{split}_seq.npy"
        if not edge_attr_seq_path.exists():
            raise FileNotFoundError(f"Missing edge attribute sequence array for split '{split}'")
        edge_attr_seq = np.load(edge_attr_seq_path)
        scenario_path = data_dir / f"scenario_{split}.npy"
        scenario_labels = None
        if scenario_path.exists():
            scenario_labels = np.load(scenario_path, allow_pickle=True)

        results_path = data_dir / f"{split}_results_list.pkl"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing simulation results pickle for split '{split}'")
        with results_path.open("rb") as handle:
            results = pickle.load(handle)

        seq_len = X_split.shape[1]
        limit = args.limit_scenarios or X_split.shape[0]

        for dataset_idx, entry in _iter_valid_results(results, seq_len):
            if dataset_idx >= limit:
                break
            sim_results, _, pump_ctrl = entry
            pressures = sim_results.node["pressure"].clip(lower=MIN_PRESSURE)
            head_df = sim_results.node["head"]
            p_arr = pressures.to_numpy(dtype=np.float64)
            h_arr = head_df.to_numpy(dtype=np.float64)
            scenario_label = (
                scenario_labels[dataset_idx] if scenario_labels is not None and dataset_idx < len(scenario_labels) else ""
            )
            X_seq = X_split[dataset_idx]
            Y_entry = Y_split[dataset_idx]
            if isinstance(Y_entry, np.ndarray) and Y_split.dtype == object:
                Y_entry = Y_entry.item()
            if not isinstance(Y_entry, dict):
                raise TypeError("Expected Y array to contain dictionaries with 'node_outputs'")
            node_targets = np.asarray(Y_entry["node_outputs"])
            edge_seq = edge_attr_seq[dataset_idx]
            max_time = min(seq_len, p_arr.shape[0] - 1, node_targets.shape[0])
            for t in range(max_time):
                node_feat = X_seq[t, :, 1]
                node_target = node_targets[t, :, 0]
                actual_now = p_arr[t]
                actual_next = p_arr[t + 1]
                actual_now_adj = actual_now.copy()
                actual_next_adj = actual_next.copy()
                actual_now_adj[reservoir_mask] = reservoir_base[reservoir_mask]
                actual_next_adj[reservoir_mask] = reservoir_base[reservoir_mask]
                feat_diff = np.abs(node_feat - actual_now_adj)
                target_diff = np.abs(node_target - actual_next_adj)
                edge_step = edge_seq[t]
                max_speed_diff = 0.0
                max_head_diff = 0.0
                max_unit_diff = 0.0
                for pump_name in pump_names:
                    meta = pump_meta[pump_name]
                    ctrl_seq = np.asarray(pump_ctrl[pump_name], dtype=np.float64)
                    if ctrl_seq.size == 0:
                        continue
                    ctrl_idx = min(t + 1, ctrl_seq.size - 1)
                    speed_val = float(edge_step[meta.forward_idx, speed_col])
                    ctrl_val = float(ctrl_seq[ctrl_idx])
                    head_feature = float(edge_step[meta.forward_idx, head_col])
                    head_actual = float(h_arr[t, meta.end_node_idx] - h_arr[t, meta.start_node_idx])
                    length_km = meta.length_m / 1000.0 if meta.length_m > 0.0 else 0.0
                    if length_km > 0.0:
                        unit_actual = float(
                            (h_arr[t, meta.start_node_idx] - h_arr[t, meta.end_node_idx]) / max(length_km, 1e-6)
                        )
                    else:
                        unit_actual = 0.0
                    unit_feature = float(edge_step[meta.forward_idx, unit_col])
                    speed_diff = abs(speed_val - ctrl_val)
                    head_diff = abs(head_feature - head_actual)
                    unit_diff = abs(unit_feature - unit_actual)
                    max_speed_diff = max(max_speed_diff, speed_diff)
                    max_head_diff = max(max_head_diff, head_diff)
                    max_unit_diff = max(max_unit_diff, unit_diff)
                    pump_records.append(
                        {
                            "split": split,
                            "scenario_idx": dataset_idx,
                            "scenario_label": scenario_label,
                            "timestep": t,
                            "minutes": float(pressures.index[t]),
                            "pump": pump_name,
                            "speed_feature": speed_val,
                            "speed_command": ctrl_val,
                            "speed_abs_diff": speed_diff,
                            "head_feature": head_feature,
                            "head_actual": head_actual,
                            "head_abs_diff": head_diff,
                            "unit_headloss_feature": unit_feature,
                            "unit_headloss_actual": unit_actual,
                            "unit_headloss_abs_diff": unit_diff,
                        }
                    )
                timestep_records.append(
                    {
                        "split": split,
                        "scenario_idx": dataset_idx,
                        "scenario_label": scenario_label,
                        "timestep": t,
                        "minutes": float(pressures.index[t]),
                        "pressure_feat_max_abs_diff": float(feat_diff.max()),
                        "pressure_target_max_abs_diff": float(target_diff.max()),
                        "pressure_feat_mean_abs_diff": float(feat_diff.mean()),
                        "pressure_target_mean_abs_diff": float(target_diff.mean()),
                        "max_pump_speed_abs_diff": max_speed_diff,
                        "max_pump_head_abs_diff": max_head_diff,
                        "max_unit_headloss_abs_diff": max_unit_diff,
                    }
                )

    summary_path = output_dir / f"{prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if timestep_records:
        timestep_df = pd.DataFrame(timestep_records)
        timestep_df.to_csv(output_dir / f"{prefix}_timesteps.csv", index=False)
    else:
        timestep_df = pd.DataFrame()

    if pump_records:
        pump_df = pd.DataFrame(pump_records)
        pump_df.to_csv(output_dir / f"{prefix}_pump_edges.csv", index=False)
    else:
        pump_df = pd.DataFrame()

    print(f"Wrote dataset alignment summary to {summary_path}")
    if not timestep_df.empty:
        print(
            "Max per-timestep pressure feature diff:",
            float(timestep_df["pressure_feat_max_abs_diff"].max()),
        )
        print(
            "Max per-timestep pressure target diff:",
            float(timestep_df["pressure_target_max_abs_diff"].max()),
        )
        print(
            "Max per-timestep pump speed diff:",
            float(timestep_df["max_pump_speed_abs_diff"].max()),
        )
        print(
            "Max per-timestep pump head diff:",
            float(timestep_df["max_pump_head_abs_diff"].max()),
        )
    if duplicate_stats is not None:
        print(
            "Detected undocumented pump feature block: "
            f"max_abs_diff={duplicate_stats['max_abs_diff']:.6f}, "
            f"mean_abs_diff={duplicate_stats['mean_abs_diff']:.6f}"
        )


if __name__ == "__main__":
    main()
