import ast
import numpy as np
import subprocess
from pathlib import Path
import sys
import wntr
import torch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import build_edge_attr


def test_auto_w_flow_scales(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    log_file = data_dir / "training_auto_w_flow.log"
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

    F = 3 + len(wn.pump_name_list)
    N = len(wn.node_name_list)
    X = np.ones((2, N, F), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((N, 1), dtype=np.float32),
                "edge_outputs": np.zeros(2, dtype=np.float32),
            },
            {
                "node_outputs": 2 * np.ones((N, 1), dtype=np.float32),
                "edge_outputs": 4 * np.ones(2, dtype=np.float32),
            },
        ],
        dtype=object,
    )
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
        "auto_w_flow",
        "--output",
        str(tmp_path / "model.pth"),
        "--auto-w-flow",
        "--no-physics-loss",
        "--no-pressure-loss",
        "--no-pump-loss",
    ]

    subprocess.run(cmd, check=True)

    assert log_file.exists()
    args_line = next(
        line for line in log_file.read_text().splitlines() if line.startswith("args:")
    )
    args_dict = ast.literal_eval(args_line.split("args: ")[1])
    w_flow = args_dict["w_flow"]

    flow_vals = torch.tensor([0.0, 0.0, 4.0, 4.0])
    flow_std = flow_vals.std(unbiased=True) + 1e-8
    press_vals = torch.cat(
        [torch.zeros(N), 2 * torch.ones(N)], dim=0
    )
    press_std = press_vals.std(unbiased=True) + 1e-8
    expected = args_dict["w_press"] * (press_std.item() / flow_std.item())
    assert w_flow == pytest.approx(expected, rel=1e-3)

