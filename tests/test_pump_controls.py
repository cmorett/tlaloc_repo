import sys
from pathlib import Path
import random
import numpy as np
import torch
import wntr
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

import scripts.data_generation as data_generation
from scripts.data_generation import (
    _run_single_scenario,
    _extract_applied_pump_speeds,
    build_dataset,
    build_sequence_dataset,
    build_edge_index,
)
from scripts.feature_utils import (
    SequenceDataset,
    compute_sequence_norm_stats,
    apply_sequence_normalization,
    build_edge_attr,
    build_edge_pairs,
    build_edge_type,
    build_node_type,
)
from scripts.train_gnn import build_loss_mask, train_sequence


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


def test_pump_speeds_align_with_epanet_outputs():
    sim_results, _scale, controls = _run_scenario()
    link_outputs = getattr(sim_results, "link", {})
    setting_df = link_outputs.get("setting")
    status_df = link_outputs.get("status")
    for pump, values in controls.items():
        ctrl = np.asarray(values, dtype=float)
        reference = None
        if setting_df is not None and pump in setting_df:
            reference = np.nan_to_num(setting_df[pump].to_numpy(dtype=float))
        elif status_df is not None and pump in status_df:
            reference = np.nan_to_num(status_df[pump].to_numpy(dtype=float))
        if reference is None or reference.size == 0:
            continue
        length = min(len(ctrl), len(reference))
        assert length > 0
        assert np.allclose(ctrl[:length], reference[:length], atol=1e-6)
        assert np.all((ctrl[:length] >= 0.0) & (ctrl[:length] <= 1.5))


def test_extract_applied_pump_speeds_fallback_to_status():
    import pandas as pd

    status = pd.DataFrame({"P1": ["OPEN", "CLOSED", "OPEN"]})
    extracted = _extract_applied_pump_speeds(["P1"], status, None, fallback=None)
    assert np.allclose(extracted["P1"], [1.0, 0.0, 1.0])

    fallback = {"P1": [0.9, 0.9, 0.9, 0.7]}
    extracted = _extract_applied_pump_speeds(["P1"], None, None, fallback=fallback)
    assert np.allclose(extracted["P1"], fallback["P1"])


def test_closed_pumps_reflected_in_features(monkeypatch):
    import pandas as pd
    from types import SimpleNamespace

    class DummyNode:
        def __init__(self, elevation=5.0, base_head=None):
            self.elevation = elevation
            self.base_head = base_head

    class DummyLink:
        def __init__(self):
            self.length = 100.0
            self.diameter = 0.5
            self.roughness = 100.0
            self.start_node = SimpleNamespace(name="J1")
            self.end_node = SimpleNamespace(name="J2")

        def get_head_curve_coefficients(self):
            return (0.0, 0.0, 0.0)

    class DummyWN:
        def __init__(self):
            quality = SimpleNamespace(parameter="CHEMICAL")
            self.options = SimpleNamespace(quality=quality)
            self.pump_name_list = ["P1"]
            self.link_name_list = ["P1"]
            self.junction_name_list = ["J1", "J2"]
            self.tank_name_list = []
            self.reservoir_name_list = []
            self.valve_name_list = []
            self.pipe_name_list = []
            self.node_name_list = ["J1", "J2"]
            self._nodes = {
                "J1": DummyNode(elevation=10.0),
                "J2": DummyNode(elevation=12.0),
            }
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
                "head": pd.DataFrame(
                    {"J1": [60.0, 60.0, 60.0], "J2": [55.0, 55.0, 55.0]}, index=index
                ),
                "pressure": pd.DataFrame(
                    {"J1": [52.0, 51.0, 50.0], "J2": [45.0, 44.0, 43.0]}, index=index
                ),
                "quality": pd.DataFrame(
                    {"J1": [0.2, 0.2, 0.2], "J2": [0.1, 0.1, 0.1]}, index=index
                ),
                "demand": pd.DataFrame(
                    {"J1": [1.0, 1.1, 1.2], "J2": [0.0, 0.0, 0.0]}, index=index
                ),
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
    assert np.isclose(X_seq[0, 0, 0, pump_feature_start], -0.8)
    assert np.isclose(X_seq[0, 0, 1, pump_feature_start], 0.8)
    assert np.allclose(X_seq[0, 1, :, pump_feature_start:], 0.0)
    assert np.allclose(edge_attr_seq[0, 1, :, -1], 0.0)

    X_single, Y_single = build_dataset(
        [(sim_results, scale_dict, pump_controls)],
        wn_instance,
    )

    pump_feature_start_single = X_single.shape[-1] - len(wn_instance.pump_name_list)
    assert np.isclose(X_single[0, 0, pump_feature_start_single], -0.8)
    assert np.isclose(X_single[0, 1, pump_feature_start_single], 0.8)
    assert np.allclose(X_single[1, :, pump_feature_start_single:], 0.0)


def test_tank_rollout_matches_epanet_when_pump_toggles(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    inp_path = repo_root / "CTown.inp"
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(str(tmp_path / "toggle"))

    pump_name = "PU7"
    status_series = results.link["status"][pump_name]
    status = status_series.to_numpy()
    toggle_indices = np.where(np.diff(status) != 0)[0]
    assert toggle_indices.size > 0, "expected pump toggle from controls"
    toggle_idx = int(toggle_indices[0])
    assert status[toggle_idx] != status[toggle_idx + 1]

    seq_len = min(toggle_idx + 3, len(status) - 1)
    pump_controls = {name: results.link["setting"][name].tolist() for name in wn.pump_name_list}
    X_seq_raw, Y_seq_raw, _, edge_attr_seq = build_sequence_dataset(
        [(results, {}, pump_controls)],
        wn,
        seq_len=seq_len,
    )

    edge_index_np, edge_attr_np, edge_type_np, _ = build_edge_index(wn)
    node_type_np = build_node_type(wn)
    edge_pairs = build_edge_pairs(edge_index_np, edge_type_np)

    dataset = SequenceDataset(
        X_seq_raw,
        Y_seq_raw,
        edge_index_np,
        edge_attr_np,
        node_type=node_type_np,
        edge_type=edge_type_np,
        edge_attr_seq=edge_attr_seq,
    )

    tank_name = "T4"
    tank_idx = wn.node_name_list.index(tank_name)
    loss_mask = build_loss_mask(wn)
    assert loss_mask[tank_idx], "tank nodes must participate in the loss mask"
    norm_mask_np = loss_mask.cpu().numpy()
    x_mean, x_std, y_mean, y_std = compute_sequence_norm_stats(
        X_seq_raw,
        Y_seq_raw,
        per_node=False,
        node_mask=norm_mask_np,
    )
    apply_sequence_normalization(
        dataset,
        x_mean,
        x_std,
        y_mean,
        y_std,
        per_node=False,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cpu")
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = dataset.edge_attr
    edge_attr_phys = torch.tensor(build_edge_attr(wn, edge_index_np).astype(np.float32))
    node_type_tensor = dataset.node_type
    edge_type_tensor = dataset.edge_type
    loss_mask_tensor = loss_mask.to(device)

    tank_node = wn.get_node(tank_name)
    tank_area = np.pi * (float(tank_node.diameter) ** 2) / 4.0
    target_nodes_norm = dataset.Y["node_outputs"][0]
    target_edges_norm = dataset.Y["edge_outputs"][0]
    raw_initial_pressure = float(X_seq_raw[0, 0, tank_idx, 1])
    expected_level = raw_initial_pressure * tank_area

    class SimpleTankModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("target_nodes", target_nodes_norm.clone())
            self.register_buffer("target_edges", target_edges_norm.clone())
            tank_idx_tensor = torch.tensor([tank_idx], dtype=torch.long)
            self.register_buffer("tank_indices", tank_idx_tensor)
            self.tank_index = int(tank_idx_tensor.item())
            self.register_buffer("tank_areas", torch.tensor([tank_area], dtype=torch.float32))
            self.seq_len = target_nodes_norm.shape[0]
            self.tank_bias = nn.Parameter(torch.full((self.seq_len,), -5.0))
            self.reset_history = []
            self.register_buffer("y_mean_node", y_mean["node_outputs"].clone())
            self.register_buffer("y_std_node", y_std["node_outputs"].clone())
            self.register_buffer("y_mean_edge", y_mean["edge_outputs"].clone())
            self.register_buffer("y_std_edge", y_std["edge_outputs"].clone())
            self.y_mean = {
                "node_outputs": self.y_mean_node,
                "edge_outputs": self.y_mean_edge,
            }
            self.y_std = {
                "node_outputs": self.y_std_node,
                "edge_outputs": self.y_std_edge,
            }
            self.register_buffer("x_mean", x_mean.clone())
            self.register_buffer("x_std", x_std.clone())

        def reset_tank_levels(self, init_levels=None, batch_size=None, device=None):
            if init_levels is None:
                if batch_size is None:
                    raise ValueError("batch_size required when init_levels is None")
                self.current_levels = torch.zeros(
                    batch_size, self.tank_indices.numel(), device=device
                )
            else:
                self.current_levels = init_levels.clone()
                self.reset_history.append(self.current_levels.cpu())

        def forward(self, X_seq, edge_index, edge_attr=None, node_type=None, edge_type=None):
            batch_size, seq_len, num_nodes, _ = X_seq.shape
            node_out = (
                self.target_nodes[:seq_len]
                .unsqueeze(0)
                .expand(batch_size, seq_len, num_nodes, -1)
                .clone()
            )
            bias = self.tank_bias[:seq_len].view(1, seq_len).expand(batch_size, -1)
            node_out[:, :, self.tank_index, 0] = bias
            edge_out = (
                self.target_edges[:seq_len]
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(batch_size, seq_len, -1, 1)
                .clone()
            )
            return {"node_outputs": node_out, "edge_outputs": edge_out}

    model = SimpleTankModel().to(device)
    model.reset_tank_levels(batch_size=1, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    for _ in range(200):
        train_sequence(
            model,
            loader,
            edge_index,
            edge_attr,
            edge_attr_phys,
            node_type_tensor,
            edge_type_tensor,
            edge_pairs,
            optimizer,
            device,
            pump_coeffs=None,
            loss_fn="mse",
            physics_loss=False,
            pressure_loss=False,
            pump_loss=False,
            node_mask=loss_mask_tensor,
            mass_scale=1.0,
            head_scale=1.0,
            pump_scale=1.0,
            w_mass=0.0,
            w_head=0.0,
            w_pump=0.0,
            w_press=100.0,
            w_cl=0.0,
            w_flow=0.0,
            amp=False,
            progress=False,
            head_sign_weight=0.5,
            has_chlorine=True,
            use_head=False,
        )

    assert model.reset_history, "tank reset levels should be recorded"
    recorded_level = model.reset_history[0].numpy()
    assert np.allclose(recorded_level, [[expected_level]], rtol=1e-5, atol=1e-5)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            X_batch, edge_attr_seq_batch, _ = batch
        else:
            X_batch, edge_attr_seq_batch = batch
        preds = model(
            X_batch.to(device),
            edge_index.to(device),
            edge_attr_seq_batch.to(device),
            node_type_tensor.to(device) if node_type_tensor is not None else None,
            edge_type_tensor.to(device) if edge_type_tensor is not None else None,
        )
        node_pred_norm = preds["node_outputs"][0]
        y_mean_node = model.y_mean["node_outputs"].to(node_pred_norm.device)
        y_std_node = model.y_std["node_outputs"].to(node_pred_norm.device)
        node_pred = (
            node_pred_norm * y_std_node.view(1, 1, -1)
            + y_mean_node.view(1, 1, -1)
        )

    true_nodes = torch.tensor(Y_seq_raw[0]["node_outputs"], dtype=torch.float32)
    diff = node_pred[:, tank_idx, 0] - true_nodes[:, tank_idx, 0]
    assert torch.max(torch.abs(diff)).item() < 0.05
    assert abs(diff[toggle_idx + 1].item()) < 0.05

