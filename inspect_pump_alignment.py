"""Diagnose pump command alignment in generated datasets.

This utility compares the pump speed features stored in ``X_<split>.npy``
against the pump commands recorded during data generation.  It prints the
absolute differences between the feature values and the commands applied during
the same hydraulic interval as well as an intentionally misaligned (one step
shifted) comparison.  After fixing the temporal alignment bug the in-sync
differences should be near machine precision while the shifted comparison
remains large, proving that the dataset now exposes the correct commands.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import wntr

from scripts.feature_utils import build_pump_node_matrix


REPO_ROOT = Path(__file__).resolve().parent


def _load_dataset(data_dir: Path, split: str) -> np.ndarray:
    array_path = data_dir / f"X_{split}.npy"
    if not array_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {array_path}")
    return np.load(array_path)


def _iter_valid_results(
    results: Sequence[Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]],
    seq_len: int,
) -> Tuple[int, Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]]:
    """Yield ``(dataset_idx, result)`` for scenarios retained in the dataset."""

    dataset_idx = 0
    for entry in results:
        sim_results = entry[0]
        times = sim_results.node["pressure"].index
        if seq_len > 1 and len(times) <= seq_len:
            continue
        yield dataset_idx, entry
        dataset_idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=REPO_ROOT / "data",
        type=Path,
        help="Directory containing X_*.npy, manifest.json and results pickles",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to inspect",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    pump_names: List[str] = manifest.get("pump_names", [])
    if not pump_names:
        raise ValueError("Manifest does not list any pumps to inspect")
    feature_layout: List[str] = manifest.get("node_feature_layout", [])
    pump_columns = []
    for pump in pump_names:
        name = f"pump_speed_cmd_{pump}"
        try:
            pump_columns.append(feature_layout.index(name))
        except ValueError as exc:
            raise ValueError(
                f"Pump feature '{name}' not present in node_feature_layout"
            ) from exc

    X_split = _load_dataset(data_dir, args.split)
    seq_mode = X_split.ndim == 4
    if not seq_mode:
        raise ValueError(
            "Pump alignment inspection currently expects a sequence dataset (ndim==4)"
        )
    seq_len = X_split.shape[1]

    node_names = np.load(data_dir / "node_names.npy")
    wn = wntr.network.WaterNetworkModel(str(REPO_ROOT / "CTown.inp"))
    if list(node_names) != wn.node_name_list:
        raise ValueError("Node ordering in node_names.npy does not match CTown.inp")
    pump_layout = build_pump_node_matrix(wn, dtype=np.float64)

    discharge_nodes: List[int] = []
    scale_factors: List[float] = []
    for pump_idx in range(len(pump_names)):
        layout_column = pump_layout[:, pump_idx]
        positive = np.where(layout_column > 0)[0]
        if positive.size == 0:
            raise ValueError(f"Pump {pump_names[pump_idx]} has no downstream node in layout")
        node_idx = positive[np.argmax(layout_column[positive])]
        discharge_nodes.append(int(node_idx))
        scale_factors.append(float(layout_column[node_idx]))

    with (data_dir / f"{args.split}_results_list.pkl").open("rb") as handle:
        results = pickle.load(handle)

    abs_diff: List[float] = []
    abs_shift_diff: List[float] = []
    per_pump_max: Dict[str, float] = {pump: 0.0 for pump in pump_names}

    for dataset_idx, entry in _iter_valid_results(results, seq_len):
        sim_results, _scale_dict, pump_ctrl = entry
        X_scenario = X_split[dataset_idx]
        for pump_idx, pump in enumerate(pump_names):
            ctrl = np.asarray(pump_ctrl[pump], dtype=np.float64)
            ctrl = ctrl[:seq_len]
            if ctrl.size == 0:
                continue
            node_idx = discharge_nodes[pump_idx]
            scale = scale_factors[pump_idx]
            feature_series = X_scenario[:, node_idx, pump_columns[pump_idx]] / scale
            if feature_series.shape[0] != ctrl.shape[0]:
                min_len = min(feature_series.shape[0], ctrl.shape[0])
                feature_series = feature_series[:min_len]
                ctrl = ctrl[:min_len]
            diff = np.abs(feature_series - ctrl)
            abs_diff.extend(diff.tolist())
            per_pump_max[pump] = max(per_pump_max[pump], float(diff.max(initial=0.0)))
            if feature_series.shape[0] > 1 and ctrl.shape[0] > 1:
                shifted = np.abs(feature_series[1:] - ctrl[:-1])
                abs_shift_diff.extend(shifted.tolist())

    if not abs_diff:
        raise RuntimeError("No pump samples were collected – check dataset contents")

    diff_arr = np.asarray(abs_diff, dtype=np.float64)
    shift_arr = np.asarray(abs_shift_diff, dtype=np.float64) if abs_shift_diff else None

    print(f"Checked {diff_arr.size} pump feature samples across split '{args.split}'.")
    print(f"Alignment |diff(feature_t - command_t)|: mean={diff_arr.mean():.6f}, "
          f"max={diff_arr.max():.6f}, median={np.median(diff_arr):.6f}")
    if shift_arr is not None and shift_arr.size:
        print(f"One-step shift |feature_t - command_(t-1)|: mean={shift_arr.mean():.6f}, "
              f"max={shift_arr.max():.6f}")
    print("Per-pump max absolute misalignment:")
    for pump, value in per_pump_max.items():
        print(f"  {pump}: {value:.6f}")


if __name__ == "__main__":
    main()
