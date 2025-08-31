import subprocess
import re
from pathlib import Path

import numpy as np
import pytest
import wntr

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.train_gnn import build_edge_attr, apply_warmup


def test_apply_warmup_basic():
    assert apply_warmup(2.0, 0, 3) == pytest.approx(2.0 / 3)
    assert apply_warmup(2.0, 1, 3) == pytest.approx(4.0 / 3)
    assert apply_warmup(2.0, 2, 3) == pytest.approx(2.0)
    assert apply_warmup(2.0, 5, 3) == pytest.approx(2.0)


def test_cli_weight_schedule(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
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
    F = 3 + len(wn.pump_name_list)
    N = len(wn.node_name_list)
    X = np.ones((1, N, F), dtype=np.float32)
    Y = np.zeros((1, N, 1), dtype=np.float32)
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
        "3",
        "--batch-size",
        "1",
        "--run-name",
        "warmup_test",
        "--output",
        str(tmp_path / "model.pth"),
        "--w_mass",
        "1",
        "--w_head",
        "1",
        "--w_pump",
        "1",
        "--mass-warmup",
        "3",
        "--head-warmup",
        "3",
        "--pump-warmup",
        "3",
        "--no-progress",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if "w_mass" in l]
    masses = []
    for l in lines:
        m = re.search(r"w_mass=([0-9.]+)", l)
        masses.append(float(m.group(1)))
    assert masses == pytest.approx([1/3, 2/3, 1.0], rel=1e-2)
