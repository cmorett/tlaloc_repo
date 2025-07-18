import os
import torch
import wntr
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "scripts"))
TEMP_DIR = REPO_ROOT / "data" / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
from scripts.mpc_control import load_network
from scripts.experiments_validation import validate_surrogate

class DummyModel(torch.nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.out_dim = out_dim
        self.x_mean = None
        self.x_std = torch.ones(1)
        self.y_mean = None
        self.y_std = torch.ones(1)

    def forward(self, x, edge_index, edge_attr=None, node_types=None, edge_types=None):
        return torch.zeros(x.size(0), self.out_dim, device=x.device)

def test_validate_surrogate_accepts_tuple():
    device = torch.device('cpu')
    wn, node_to_index, pump_names, edge_index, node_types, edge_types = load_network('CTown.inp')
    wn.options.time.duration = 2 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim(str(TEMP_DIR / "temp"))
    model = DummyModel().to(device)
    metrics, arr, times = validate_surrogate(
        model,
        edge_index,
        None,
        wn,
        [(res, {})],
        device,
        "test",
        torch.tensor(node_types, dtype=torch.long),
        torch.tensor(edge_types, dtype=torch.long),
    )
    assert "pressure_rmse" in metrics
    assert arr.shape[1] == len(wn.node_name_list)


def test_validate_surrogate_clips_low_pressure():
    device = torch.device('cpu')
    wn, node_to_index, pump_names, edge_index, node_types, edge_types = load_network('CTown.inp')
    wn.options.time.duration = 2 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim(str(TEMP_DIR / "temp_clip"))
    # create an unrealistic negative pressure for the next timestep
    res.node["pressure"].iloc[1] = -1.0
    model = DummyModel().to(device)
    metrics, arr, times = validate_surrogate(
        model,
        edge_index,
        None,
        wn,
        [res],
        device,
        "test_clip",
        torch.tensor(node_types, dtype=torch.long),
        torch.tensor(edge_types, dtype=torch.long),
    )
    # clipped to 5 m -> prediction (0) minus 5 yields -5
    assert arr.shape[0] >= 1
    assert abs(arr[0, 0] + 5.0) < 1e-6


def test_validate_surrogate_respects_node_mask():
    """Metrics should ignore nodes marked as tanks/reservoirs."""
    device = torch.device('cpu')
    wn, node_to_index, pump_names, edge_index, node_types, edge_types = load_network('CTown.inp')
    wn.options.time.duration = 2 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim(str(TEMP_DIR / "temp_mask"))
    model = DummyModel().to(device)
    custom_types = [0] + [1] * (len(node_types) - 1)
    metrics, arr, times = validate_surrogate(
        model,
        edge_index,
        None,
        wn,
        [res],
        device,
        "test_mask",
        torch.tensor(custom_types, dtype=torch.long),
        torch.tensor(edge_types, dtype=torch.long),
    )
    p_df = res.node["pressure"].clip(lower=5.0)
    c_df = res.node["quality"]
    vals_p = [p_df.iloc[i + 1, 0] for i in range(len(p_df.index) - 1)]
    vals_c = [c_df.iloc[i + 1, 0] for i in range(len(c_df.index) - 1)]
    expected_rmse_p = np.sqrt(np.mean(np.square(vals_p)))
    expected_mae_p = np.mean(np.abs(vals_p))
    expected_max_p = np.max(np.abs(vals_p))
    expected_rmse_c = np.sqrt(np.mean(np.square(vals_c)))
    expected_mae_c = np.mean(np.abs(vals_c))
    expected_max_c = np.max(np.abs(vals_c))
    assert abs(metrics["pressure_rmse"] - expected_rmse_p) < 1e-6
    assert abs(metrics["pressure_mae"] - expected_mae_p) < 1e-6
    assert abs(metrics["pressure_max_error"] - expected_max_p) < 1e-6
    assert abs(metrics["chlorine_rmse"] - expected_rmse_c) < 1e-6
    assert abs(metrics["chlorine_mae"] - expected_mae_c) < 1e-6
    assert abs(metrics["chlorine_max_error"] - expected_max_c) < 1e-6

def test_validate_surrogate_dict_stats():
    """Validation should unnormalize predictions using dict stats."""
    device = torch.device('cpu')
    wn, node_to_index, pump_names, edge_index, node_types, edge_types = load_network('CTown.inp')
    wn.options.time.duration = 2 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim(str(TEMP_DIR / "temp_dict"))
    model = DummyModel().to(device)
    model.y_mean = {"node_outputs": torch.tensor([1.0, 0.1])}
    model.y_std = {"node_outputs": torch.tensor([2.0, 0.2])}
    metrics, arr, times = validate_surrogate(
        model,
        edge_index,
        None,
        wn,
        [res],
        device,
        "test_dict",
        torch.tensor(node_types, dtype=torch.long),
        torch.tensor(edge_types, dtype=torch.long),
    )
    p_df = res.node["pressure"].clip(lower=5.0)
    c_df = res.node["quality"]
    true_p = p_df.iloc[1, 0]
    true_c = c_df.iloc[1, 0]
    expected_diff_p = 1.0 - true_p
    assert arr.shape[0] >= 1
    assert abs(arr[0, 0] - expected_diff_p) < 1e-6


def test_validate_surrogate_edge_dim_check():
    device = torch.device('cpu')
    (
        wn,
        node_to_index,
        pump_names,
        edge_index,
        edge_attr,
        node_types,
        edge_types,
    ) = load_network('CTown.inp', return_edge_attr=True)
    wn.options.time.duration = 2 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim(str(TEMP_DIR / 'temp_edge'))

    class EdgeModel(DummyModel):
        def __init__(self):
            super().__init__()
            self.edge_dim = 2

    model = EdgeModel().to(device)
    with pytest.raises(ValueError):
        validate_surrogate(
            model,
            edge_index,
            edge_attr,
            wn,
            [res],
            device,
            'edge_check',
            torch.tensor(node_types, dtype=torch.long),
            torch.tensor(edge_types, dtype=torch.long),
        )

