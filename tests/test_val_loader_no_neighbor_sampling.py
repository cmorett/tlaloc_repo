import numpy as np
import subprocess
from pathlib import Path
import sys
import wntr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import build_edge_attr


def test_train_with_val_without_neighbor_sampling(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    log_file = data_dir / "training_unit_val.log"
    if log_file.exists():
        log_file.unlink()

    wn = wntr.network.WaterNetworkModel(repo / "CTown.inp")
    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    link = wn.get_link(wn.link_name_list[0])
    edge_index = np.array(
        [
            [node_map[link.start_node.name], node_map[link.end_node.name]],
            [node_map[link.end_node.name], node_map[link.start_node.name]],
        ],
        dtype=np.int64,
    )
    edge_attr = build_edge_attr(wn, edge_index)

    np.save(tmp_path / "edge_index.npy", edge_index)
    np.save(tmp_path / "edge_attr.npy", edge_attr)

    F = 4 + len(wn.pump_name_list)
    N = len(wn.node_name_list)
    X = np.ones((1, N, F), dtype=np.float32)
    Y = np.zeros((1, N, 2), dtype=np.float32)

    np.save(tmp_path / "X_train.npy", X)
    np.save(tmp_path / "Y_train.npy", Y)
    np.save(tmp_path / "X_val.npy", X)
    np.save(tmp_path / "Y_val.npy", Y)

    cmd = [
        "python",
        str(repo / "scripts/train_gnn.py"),
        "--x-path",
        str(tmp_path / "X_train.npy"),
        "--y-path",
        str(tmp_path / "Y_train.npy"),
        "--x-val-path",
        str(tmp_path / "X_val.npy"),
        "--y-val-path",
        str(tmp_path / "Y_val.npy"),
        "--edge-index-path",
        str(tmp_path / "edge_index.npy"),
        "--edge-attr-path",
        str(tmp_path / "edge_attr.npy"),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--run-name",
        "unit_val",
        "--output",
        str(tmp_path / "model.pth"),
        "--workers",
        "0",
    ]

    subprocess.run(cmd, check=True)

    assert log_file.exists()
