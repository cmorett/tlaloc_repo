import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wntr

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(REPO_ROOT))

from models.gnn_surrogate import MultiTaskGNNSurrogate, RecurrentGNNSurrogate
from scripts.feature_utils import (
    SequenceDataset,
    apply_sequence_normalization,
    build_node_type,
    build_edge_type,
)
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "logs"


DEFAULT_CLUSTER = {
    "J304",
    "J306",
    "J87",
    "J84",
    "J86",
    "J219",
    "J220",
    "J60",
    "J59",
    "J57",
    "J62",
    "J65",
    "J55",
    "J118",
    "J58",
    "J243",
    "J242",
    "J241",
    "J250",
    "J249",
    "J248",
    "J246",
    "J247",
    "J236",
    "J237",
    "J244",
    "J245",
    "J66",
    "J67",
    "J53",
    "J64",
    "J54",
    "J73",
    "J71",
    "J77",
    "J72",
    "J74",
    "J68",
    "J92",
    "J76",
    "J61",
    "J69",
    "J70",
    "J85",
    "J56",
    "T5",
}


TARGET_PUMPS = ["PU8", "PU9"]
TARGET_PIPES = ["P403", "P409", "P245"]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (bytes, np.bytes_)):
        return obj.decode()
    return str(obj)


def _load_split_arrays(
    split: str,
    sequence: bool,
    base_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    suffix = {
        "train": "_train",
        "val": "_val",
        "test": "_test",
    }[split]
    X = np.load(base_dir / f"X{suffix}.npy")
    Y = np.load(base_dir / f"Y{suffix}.npy", allow_pickle=True)
    edge_attr_seq = None
    scenarios = None
    if sequence:
        seq_path = base_dir / f"edge_attr{suffix}_seq.npy"
        if seq_path.exists():
            edge_attr_seq = np.load(seq_path)
        scen_path = base_dir / f"scenario{suffix}.npy"
        if scen_path.exists():
            scenarios = np.load(scen_path, allow_pickle=True)
    return X, Y, edge_attr_seq, scenarios


def _instantiate_model(
    state: Dict[str, torch.Tensor],
    meta: Dict[str, object],
    device: torch.device,
    num_pumps: int,
) -> torch.nn.Module:
    model_class = meta.get("model_class", "MultiTaskGNNSurrogate")
    pump_feature_offset = int(meta.get("pump_feature_offset", 4))
    pump_feature_repeats = int(meta.get("pump_feature_repeats", 1) or 1)
    in_channels = int(meta.get("in_channels", 44))
    hidden_channels = int(meta.get("hidden_dim", 256))
    edge_dim = int(meta.get("edge_dim", 0))
    if model_class == "MultiTaskGNNSurrogate":
        model = MultiTaskGNNSurrogate(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            node_output_dim=int(meta.get("node_output_dim", 1)),
            edge_output_dim=int(meta.get("edge_output_dim", 1)),
            num_layers=int(meta.get("num_layers", 4)),
            use_attention=bool(meta.get("use_attention", False)),
            gat_heads=int(meta.get("gat_heads", 4)),
            dropout=float(meta.get("dropout", 0.0)),
            residual=bool(meta.get("residual", True)),
            rnn_hidden_dim=int(meta.get("rnn_hidden_dim", 64)),
            share_weights=bool(meta.get("share_weights", False)),
            num_node_types=int(meta.get("num_node_types", 1)),
            num_edge_types=int(meta.get("num_edge_types", 1)),
            use_checkpoint=bool(meta.get("use_checkpoint", False)),
            pressure_feature_idx=int(meta.get("pressure_feature_idx", 1)),
            use_pressure_skip=bool(meta.get("use_pressure_skip", True)),
            num_pumps=num_pumps,
            pump_feature_offset=pump_feature_offset,
            pump_feature_repeats=pump_feature_repeats,
        )
    elif model_class == "RecurrentGNNSurrogate":
        model = RecurrentGNNSurrogate(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            output_dim=int(meta.get("output_dim", 1)),
            num_layers=int(meta.get("num_layers", 4)),
            use_attention=bool(meta.get("use_attention", False)),
            gat_heads=int(meta.get("gat_heads", 4)),
            dropout=float(meta.get("dropout", 0.0)),
            residual=bool(meta.get("residual", True)),
            rnn_hidden_dim=int(meta.get("rnn_hidden_dim", 64)),
            share_weights=bool(meta.get("share_weights", False)),
            num_node_types=int(meta.get("num_node_types", 1)),
            num_edge_types=int(meta.get("num_edge_types", 1)),
            pressure_feature_idx=int(meta.get("pressure_feature_idx", 1)),
            use_pressure_skip=bool(meta.get("use_pressure_skip", True)),
            num_pumps=num_pumps,
            pump_feature_offset=pump_feature_offset,
        )
    else:
        raise ValueError(f"Unsupported surrogate class '{model_class}' in checkpoint")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.pump_feature_repeats = pump_feature_repeats
    model.has_pump_head_features = bool(meta.get("has_pump_head_features", pump_feature_repeats >= 2))
    return model


def _attach_pump_indices(
    model: torch.nn.Module,
    wn: wntr.network.WaterNetworkModel,
    device: torch.device,
) -> None:
    if not wn.pump_name_list:
        model.pump_start_indices = torch.tensor([], dtype=torch.long, device=device)
        model.pump_end_indices = torch.tensor([], dtype=torch.long, device=device)
        return
    node_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    start_idx: List[int] = []
    end_idx: List[int] = []
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        start = pump.start_node.name if hasattr(pump.start_node, "name") else pump.start_node
        end = pump.end_node.name if hasattr(pump.end_node, "name") else pump.end_node
        start_idx.append(node_map[start])
        end_idx.append(node_map[end])
    model.pump_start_indices = torch.tensor(start_idx, dtype=torch.long, device=device)
    model.pump_end_indices = torch.tensor(end_idx, dtype=torch.long, device=device)


def _build_edge_lookup(
    wn: wntr.network.WaterNetworkModel,
    edge_index: np.ndarray,
    node_names: List[str],
) -> Dict[str, Tuple[int, int]]:
    node_map = {name: idx for idx, name in enumerate(node_names)}
    pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    mapping: Dict[str, Tuple[int, int]] = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        start = link.start_node.name if hasattr(link.start_node, "name") else link.start_node
        end = link.end_node.name if hasattr(link.end_node, "name") else link.end_node
        s_idx = node_map[start]
        t_idx = node_map[end]
        try:
            fwd = pairs.index((s_idx, t_idx))
            rev = pairs.index((t_idx, s_idx))
        except ValueError as exc:
            raise ValueError(f"Failed to locate edge indices for link {link_name}") from exc
        mapping[link_name] = (fwd, rev)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose pump-induced pressure discrepancies")
    parser.add_argument("--model", type=str, default=str(REPO_ROOT / "models" / "gnn_surrogate.pth"), help="Path to trained surrogate checkpoint")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to analyse")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Directory containing dataset arrays")
    parser.add_argument("--cluster", type=str, nargs="*", default=sorted(DEFAULT_CLUSTER), help="Node names forming the hotspot cluster")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--device", type=str, default="auto", help="Computation device: auto, cpu or cuda")
    parser.add_argument("--output-prefix", type=str, default="cluster_diagnostics", help="Prefix for generated logs")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalisation and assume inputs already scaled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_dir}")
    manifest = json.loads(manifest_path.read_text())

    cluster_nodes = set(args.cluster)
    device = (
        torch.device("cuda")
        if args.device == "cuda"
        else torch.device("cpu")
        if args.device == "cpu"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load arrays
    X_raw, Y_raw, edge_attr_seq_raw, scenario_labels = _load_split_arrays(
        args.split,
        sequence=True,
        base_dir=data_dir,
    )
    if edge_attr_seq_raw is None:
        raise ValueError("Sequence edge attributes are required for diagnostics; regenerate data with --sequence-length > 1")

    edge_index = np.load(data_dir / "edge_index.npy")
    node_names = np.load(data_dir / "node_names.npy", allow_pickle=True).tolist()
    wn = wntr.network.WaterNetworkModel(str(REPO_ROOT / "CTown.inp"))

    node_type = build_node_type(wn)
    edge_type = build_edge_type(wn, edge_index)

    dataset = SequenceDataset(
        X_raw,
        Y_raw,
        edge_index,
        np.load(data_dir / "edge_attr.npy"),
        node_type=node_type,
        edge_type=edge_type,
        edge_attr_seq=edge_attr_seq_raw,
    )

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint or "model_meta" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict/model_meta")
    model = _instantiate_model(
        checkpoint["model_state_dict"],
        checkpoint["model_meta"],
        device,
        num_pumps=len(wn.pump_name_list),
    )
    _attach_pump_indices(model, wn, device)

    norm_stats = checkpoint.get("norm_stats")
    if norm_stats is None:
        raise ValueError("Checkpoint missing norm_stats; rerun training with --normalize enabled")

    if not args.no_normalize:
        x_mean = torch.tensor(norm_stats["x_mean"], dtype=torch.float32)
        x_std = torch.tensor(norm_stats["x_std"], dtype=torch.float32)
        edge_mean = torch.tensor(norm_stats["edge_mean"], dtype=torch.float32)
        edge_std = torch.tensor(norm_stats["edge_std"], dtype=torch.float32)
        y_mean = {k: torch.tensor(v, dtype=torch.float32) for k, v in norm_stats["y_mean"].items()}
        y_std = {k: torch.tensor(v, dtype=torch.float32) for k, v in norm_stats["y_std"].items()}
        apply_sequence_normalization(
            dataset,
            x_mean,
            x_std,
            y_mean,
            y_std,
            edge_mean,
            edge_std,
            per_node=True,
        )

    model.x_mean = torch.tensor(norm_stats["x_mean"], dtype=torch.float32, device=device)
    model.x_std = torch.tensor(norm_stats["x_std"], dtype=torch.float32, device=device)
    model.y_mean = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in norm_stats["y_mean"].items()}
    model.y_std = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in norm_stats["y_std"].items()}
    if "edge_mean" in norm_stats and "edge_std" in norm_stats:
        model.edge_mean = torch.tensor(norm_stats["edge_mean"], dtype=torch.float32, device=device)
        model.edge_std = torch.tensor(norm_stats["edge_std"], dtype=torch.float32, device=device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    node_type_t = torch.tensor(node_type, dtype=torch.long, device=device)
    edge_type_t = torch.tensor(edge_type, dtype=torch.long, device=device)

    edge_lookup = _build_edge_lookup(wn, edge_index, node_names)
    pump_head_col = int(manifest["edge_dynamic_columns"]["pump_head_rise"])
    unit_loss_col = int(manifest["edge_dynamic_columns"]["unit_head_loss"])
    pump_speed_col = int(manifest["edge_dynamic_columns"]["pump_speed"])

    timestep_seconds = float(getattr(wn.options.time, "hydraulic_timestep", 3600.0))

    records: List[Dict[str, object]] = []
    mae_records: Dict[str, List[float]] = {name: [] for name in node_names}

    if scenario_labels is not None:
        scenario_types = scenario_labels.tolist()
    else:
        scenario_types = ["unknown"] * len(dataset)

    y_mean_node = model.y_mean["node_outputs"].detach()
    y_std_node = model.y_std["node_outputs"].detach()

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            X_seq, edge_attr_seq, target = batch
        else:
            raise RuntimeError("Unexpected dataset output; expected sequence with edge attributes")
        seq_len = X_seq.size(1)
        scenario_index = batch_idx
        raw_target = Y_raw[scenario_index]
        if not isinstance(raw_target, dict):
            raise TypeError("Expected multi-task target dict in sequence dataset")

        X_seq = X_seq.to(device)
        edge_attr_seq = edge_attr_seq.to(device)
        with torch.no_grad():
            outputs = model(
                X_seq,
                edge_index_t,
                edge_attr_seq,
                node_type_t,
                edge_type_t,
            )
        node_pred = outputs["node_outputs"][0].cpu()
        node_true = target["node_outputs"][0].cpu()

        # Un-normalise predictions
        y_mean_local = y_mean_node.clone()
        y_std_local = y_std_node.clone()
        while y_mean_local.dim() < node_pred.dim():
            y_mean_local = y_mean_local.unsqueeze(0)
            y_std_local = y_std_local.unsqueeze(0)
        node_pred = node_pred * y_std_local.cpu() + y_mean_local.cpu()
        node_true = node_true * y_std_local.cpu() + y_mean_local.cpu()

        edge_attr_seq_np = edge_attr_seq_raw[scenario_index]
        edge_outputs_true = raw_target["edge_outputs"]

        for t in range(seq_len):
            minutes = t * timestep_seconds / 60.0
            pred_t = node_pred[t, :, 0].numpy()
            true_t = node_true[t, :, 0].numpy()
            residual = pred_t - true_t
            abs_residual = np.abs(residual)
            for idx in range(len(node_names)):
                mae_records[node_names[idx]].append(abs_residual[idx])
                records.append(
                    {
                        "scenario_index": scenario_index,
                        "scenario_label": str(scenario_types[scenario_index]),
                        "timestep": t,
                        "minutes": minutes,
                        "node_index": idx,
                        "node_name": node_names[idx],
                        "predicted_pressure": float(pred_t[idx]),
                        "actual_pressure": float(true_t[idx]),
                        "abs_error": float(abs_residual[idx]),
                        "error": float(residual[idx]),
                    }
                )

            pump_metrics = {}
            for pump in TARGET_PUMPS:
                if pump not in edge_lookup:
                    continue
                fwd_idx, _ = edge_lookup[pump]
                pump_metrics[f"pump_{pump}_headrise"] = float(edge_attr_seq_np[t, fwd_idx, pump_head_col])
                pump_metrics[f"pump_{pump}_speed"] = float(edge_attr_seq_np[t, fwd_idx, pump_speed_col])

            for pipe in TARGET_PIPES:
                if pipe not in edge_lookup:
                    continue
                fwd_idx, _ = edge_lookup[pipe]
                pump_metrics[f"pipe_{pipe}_unit_headloss"] = float(edge_attr_seq_np[t, fwd_idx, unit_loss_col])
                if edge_outputs_true.ndim == 3:
                    flow_val = float(edge_outputs_true[t, fwd_idx, 0])
                else:
                    flow_val = float(edge_outputs_true[t, fwd_idx])
                pump_metrics[f"pipe_{pipe}_flow"] = flow_val

            for rec in records[-len(node_names) :]:
                rec.update(pump_metrics)

    df = pd.DataFrame.from_records(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{args.output_prefix}_{args.split}_{timestamp}"
    node_detail_path = LOG_DIR / f"{output_prefix}.csv"
    df.to_csv(node_detail_path, index=False)

    mae_series = {node: float(np.mean(vals)) if vals else 0.0 for node, vals in mae_records.items()}
    mae_df = pd.DataFrame(
        {
            "node_name": list(mae_series.keys()),
            "mae": list(mae_series.values()),
            "cluster": [name in cluster_nodes for name in mae_series.keys()],
        }
    )
    mae_df.sort_values("mae", ascending=False, inplace=True)
    mae_path = LOG_DIR / f"{output_prefix}_mae.csv"
    mae_df.to_csv(mae_path, index=False)

    cluster_mask = df["node_name"].isin(cluster_nodes)
    cluster_time = (
        df[cluster_mask]
        .groupby(["scenario_index", "timestep"])
        .agg(cluster_mae=("abs_error", "mean"), minutes=("minutes", "first"))
        .reset_index()
    )
    worst_windows = cluster_time.nlargest(5, "cluster_mae").to_dict(orient="records")

    summary = {
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "mean_cluster_mae": float(mae_df[mae_df["cluster"]]["mae"].mean()),
        "mean_noncluster_mae": float(mae_df[~mae_df["cluster"]]["mae"].mean()),
        "peak_cluster_records": df[df["node_name"].isin(cluster_nodes)].nlargest(10, "abs_error").to_dict(orient="records"),
        "worst_cluster_windows": worst_windows,
    }

    cluster_subset = df[cluster_mask]
    for pump in TARGET_PUMPS:
        col = f"pump_{pump}_headrise"
        if col in cluster_subset.columns:
            corr = cluster_subset["abs_error"].corr(cluster_subset[col])
            summary[f"corr_abs_error_{col}"] = None if pd.isna(corr) else float(corr)

    (LOG_DIR / f"{output_prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    print(f"Detailed node diagnostics written to {node_detail_path}")
    print(f"Per-node MAE summary written to {mae_path}")


if __name__ == "__main__":
    main()
