import sys
from pathlib import Path
import random
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.data_generation as data_generation
from scripts.data_generation import (
    _run_single_scenario,
    build_dataset,
    build_sequence_dataset,
)


def _run_scenario():
    inp = Path(__file__).resolve().parents[1] / 'CTown.inp'
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        return _run_single_scenario((0, str(inp), 42))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def test_at_least_one_pump_active_per_hour():
    res, scale, controls = _run_scenario()
    hours = len(next(iter(controls.values())))
    for h in range(hours):
        assert any(controls[p][h] > 0.05 for p in controls)


def test_pump_speeds_continuous_and_correlated():
    res, scale, controls = _run_scenario()
    for speeds in controls.values():
        arr = np.array(speeds, dtype=float)
        # Speeds span more than the discrete {0.0, 0.5, 1.0} set
        assert len(np.unique(np.round(arr, 2))) > 3
        # Adjacent hours should be positively correlated when the pump remains open
        if len(arr) > 1:
            prev = arr[:-1]
            nxt = arr[1:]
            active_mask = (prev > 0.05) & (nxt > 0.05)
            if np.any(active_mask):
                corr = np.corrcoef(prev[active_mask], nxt[active_mask])[0, 1]
                assert corr > 0.1
                diffs = np.diff(arr)
                assert np.max(np.abs(diffs[active_mask])) < 0.35


def test_closed_pumps_reflected_in_features(monkeypatch):
    import pandas as pd
    from types import SimpleNamespace

    class DummyNode:
        def __init__(self, elevation=5.0, base_head=None):
            self.elevation = elevation
            self.base_head = base_head

    class DummyLink:
        length = 100.0
        diameter = 0.5
        roughness = 100.0

        def get_head_curve_coefficients(self):
            return (0.0, 0.0, 0.0)

    class DummyWN:
        def __init__(self):
            quality = SimpleNamespace(parameter="CHEMICAL")
            self.options = SimpleNamespace(quality=quality)
            self.pump_name_list = ["P1"]
            self.link_name_list = ["P1"]
            self.junction_name_list = ["J1"]
            self.tank_name_list = []
            self.reservoir_name_list = []
            self.valve_name_list = []
            self.pipe_name_list = []
            self.node_name_list = ["J1"]
            self._nodes = {"J1": DummyNode(elevation=10.0)}
            self._links = {"P1": DummyLink()}

        def get_node(self, name):
            return self._nodes[name]

        def get_link(self, name):
            return self._links[name]

    wn_instance = DummyWN()

    def fake_build(*_args, **_kwargs):
        return wn_instance, {}, {"P1": [0.8, 0.8, 0.8]}

    def fake_pump_energy(flows, heads, wn):
        index = flows.index if hasattr(flows, "index") else range(len(flows))
        data = {pump: np.zeros(len(index), dtype=float) for pump in wn.pump_name_list}
        return pd.DataFrame(data, index=index)

    class DummySim:
        def __init__(self, wn):
            self.wn = wn

        def run_sim(self, file_prefix=None):
            index = pd.Index([0, 3600, 7200], name="time")
            link = {
                "flowrate": pd.DataFrame({"P1": [0.1, 0.2, 0.3]}, index=index),
                "status": pd.DataFrame({"P1": [1, 0, 1]}, index=index),
                "setting": pd.DataFrame({"P1": [0.8, 0.0, 0.7]}, index=index),
            }
            node = {
                "head": pd.DataFrame({"J1": [60.0, 60.0, 60.0]}, index=index),
                "pressure": pd.DataFrame({"J1": [52.0, 51.0, 50.0]}, index=index),
                "quality": pd.DataFrame({"J1": [0.2, 0.2, 0.2]}, index=index),
                "demand": pd.DataFrame({"J1": [1.0, 1.1, 1.2]}, index=index),
            }
            return SimpleNamespace(link=link, node=node)

    monkeypatch.setattr(data_generation, "_build_randomized_network", fake_build)
    monkeypatch.setattr(data_generation, "pump_energy", fake_pump_energy)
    monkeypatch.setattr(data_generation.wntr.sim, "EpanetSimulator", DummySim)

    result = data_generation._run_single_scenario((0, "fake.inp", None))
    assert result is not None
    sim_results, scale_dict, pump_controls = result
    assert np.allclose(pump_controls["P1"], [0.8, 0.0, 0.7])

    X_seq, Y_seq, scenario_types, edge_attr_seq = build_sequence_dataset(
        [(sim_results, scale_dict, pump_controls)],
        wn_instance,
        seq_len=2,
    )

    pump_feature_start = X_seq.shape[-1] - len(wn_instance.pump_name_list)
    assert np.isclose(X_seq[0, 0, 0, pump_feature_start], 0.8)
    assert np.allclose(X_seq[0, 1, :, pump_feature_start:], 0.0)
    assert np.allclose(edge_attr_seq[0, 1, :, -1], 0.0)

    X_single, Y_single = build_dataset(
        [(sim_results, scale_dict, pump_controls)],
        wn_instance,
    )

    pump_feature_start_single = X_single.shape[-1] - len(wn_instance.pump_name_list)
    assert np.isclose(X_single[0, 0, pump_feature_start_single], 0.8)
    assert np.allclose(X_single[1, :, pump_feature_start_single:], 0.0)

