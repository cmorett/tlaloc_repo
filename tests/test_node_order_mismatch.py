import numpy as np
import wntr
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import load_dataset


def test_load_dataset_detects_node_order_mismatch(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    wn = wntr.network.WaterNetworkModel(repo / "CTown.inp")

    # Build a minimal edge index
    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    link = wn.get_link(wn.link_name_list[0])
    edge_index = np.array(
        [
            [node_map[link.start_node.name], node_map[link.end_node.name]],
            [node_map[link.end_node.name], node_map[link.start_node.name]],
        ],
        dtype=np.int64,
    )
    np.save(tmp_path / "edge_index.npy", edge_index)

    F = 4 + len(wn.pump_name_list)
    N = len(wn.node_name_list)
    X = np.ones((1, N, F), dtype=np.float32)
    Y = np.zeros((1, N, 1), dtype=np.float32)

    # Permute node order and save the mismatching list
    perm = np.arange(N)
    perm[[0, 1]] = perm[[1, 0]]
    X = X[:, perm, :]
    node_names = np.array(wn.node_name_list)[perm]

    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "Y.npy", Y)
    np.save(tmp_path / "node_names.npy", node_names)

    with pytest.raises(ValueError):
        load_dataset(
            str(tmp_path / "X.npy"),
            str(tmp_path / "Y.npy"),
            str(tmp_path / "edge_index.npy"),
            expected_node_names=wn.node_name_list,
            node_names_path=str(tmp_path / "node_names.npy"),
        )
