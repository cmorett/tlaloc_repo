import sys
from pathlib import Path

import numpy as np
import pytest
import wntr
from wntr.network.base import LinkStatus

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.feature_utils import build_edge_attr  # noqa: E402
from scripts.data_generation import build_edge_index  # noqa: E402


def test_valve_edges_expose_setpoint_and_status():
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    edge_index, _, _, _ = build_edge_index(wn)
    edge_attr = build_edge_attr(wn, edge_index)

    node_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    valve = wn.get_link('v1')
    start = node_map[valve.start_node.name]
    end = node_map[valve.end_node.name]

    forward_idx = np.where((edge_index[0] == start) & (edge_index[1] == end))[0]
    assert forward_idx.size == 1
    forward_attr = edge_attr[forward_idx[0]]

    assert forward_attr[3] == pytest.approx(float(valve.initial_setting or 0.0))
    expected_flags = np.array([
        1.0 if valve.status == LinkStatus.Open else 0.0,
        1.0 if valve.status == LinkStatus.Closed else 0.0,
        1.0 if valve.status == LinkStatus.Active else 0.0,
    ], dtype=np.float32)
    assert np.allclose(forward_attr[4:7], expected_flags)
    assert forward_attr[7] == pytest.approx(float(valve.minor_loss or 0.0))

    reverse_idx = np.where((edge_index[0] == end) & (edge_index[1] == start))[0]
    assert reverse_idx.size == 1
    reverse_attr = edge_attr[reverse_idx[0]]
    assert np.allclose(reverse_attr[3:8], forward_attr[3:8])
