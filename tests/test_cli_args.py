import numpy as np
import subprocess
from pathlib import Path
import sys
import wntr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import build_edge_attr

def test_cli_no_pressure_loss(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    log_file = data_dir / "training_unit.log"
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
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "Y.npy", Y)

    cmd = [
        "python",
        str(repo / "scripts/train_gnn.py"),
        "--x-path",
        str(tmp_path / "X.npy"),
        "--y-path",
        str(tmp_path / "Y.npy"),
        "--edge-index-path",
        str(tmp_path / "edge_index.npy"),
        "--edge-attr-path",
        str(tmp_path / "edge_attr.npy"),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--run-name",
        "unit",
        "--output",
        str(tmp_path / "model.pth"),
        "--no-pressure-loss",
    ]

    subprocess.run(cmd, check=True)

    assert log_file.exists()
    log_text = log_file.read_text()
    assert "'pressure_loss': False" in log_text


def test_cli_loss_weights(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    log_file = data_dir / "training_unit_weights.log"
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
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "Y.npy", Y)

    cmd = [
        "python",
        str(repo / "scripts/train_gnn.py"),
        "--x-path",
        str(tmp_path / "X.npy"),
        "--y-path",
        str(tmp_path / "Y.npy"),
        "--edge-index-path",
        str(tmp_path / "edge_index.npy"),
        "--edge-attr-path",
        str(tmp_path / "edge_attr.npy"),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--run-name",
        "unit_weights",
        "--output",
        str(tmp_path / "model.pth"),
        "--w-press",
        "2.5",
        "--w-cl",
        "0.5",
        "--w-flow",
        "1.5",
    ]

    subprocess.run(cmd, check=True)

    assert log_file.exists()
    log_text = log_file.read_text()
    assert "'w_press': 2.5" in log_text
    assert "'w_cl': 0.5" in log_text
    assert "'w_flow': 1.5" in log_text


def test_cli_loss_scales(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    log_file = data_dir / "training_unit_scales.log"
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
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "Y.npy", Y)

    cmd = [
        "python",
        str(repo / "scripts/train_gnn.py"),
        "--x-path",
        str(tmp_path / "X.npy"),
        "--y-path",
        str(tmp_path / "Y.npy"),
        "--edge-index-path",
        str(tmp_path / "edge_index.npy"),
        "--edge-attr-path",
        str(tmp_path / "edge_attr.npy"),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--run-name",
        "unit_scales",
        "--output",
        str(tmp_path / "model.pth"),
        "--mass-scale",
        "0.5",
        "--head-scale",
        "0.25",
        "--pump-scale",
        "0.75",
        "--pressure_loss",
        "--pump-loss",
    ]

    subprocess.run(cmd, check=True)

    assert log_file.exists()
    log_text = log_file.read_text()
    assert "scales: mass=0.5, head=0.25, pump=0.75" in log_text
