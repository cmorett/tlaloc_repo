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
from scripts.experiments_validation import rollout_surrogate


class ZeroModel(torch.nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.x_mean = None
        self.x_std = torch.ones(1)
        self.y_mean = None
        self.y_std = torch.ones(1)

    def forward(self, x, edge_index, edge_attr=None, node_types=None, edge_types=None):
        return torch.zeros(self.num_nodes, 2, device=x.device)


class PerfectModel(torch.nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.i = 0
        self.x_mean = None
        self.x_std = torch.ones(1)
        self.y_mean = None
        self.y_std = torch.ones(1)

    def forward(self, x, edge_index, edge_attr=None, node_types=None, edge_types=None):
        out = self.seq[self.i].to(x.device)
        self.i += 1
        return out


def test_rollout_rmse_decreases_with_perfect_model():
    device = torch.device('cpu')
    wn, node_to_index, pump_names, edge_index, node_types, edge_types = load_network('CTown.inp')
    wn.options.time.duration = 6 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim(str(TEMP_DIR / 'rollout'))

    p_df = res.node['pressure'].clip(lower=5.0)
    c_df = res.node['quality']
    steps = 6
    seq = []
    for i in range(steps):
        p = p_df.iloc[i + 1].to_numpy()
        c = np.log1p(c_df.iloc[i + 1].to_numpy() / 1000.0)
        seq.append(np.column_stack([p, c]))
    seq = np.stack(seq)

    model_bad = ZeroModel(len(wn.node_name_list)).to(device)
    model_good = PerfectModel(seq).to(device)
    rmse_bad_p, rmse_bad_c = rollout_surrogate(
        model_bad,
        edge_index,
        None,
        wn,
        [res],
        device,
        steps,
        torch.tensor(node_types, dtype=torch.long),
        torch.tensor(edge_types, dtype=torch.long),
    )
    rmse_good_p, rmse_good_c = rollout_surrogate(
        model_good,
        edge_index,
        None,
        wn,
        [res],
        device,
        steps,
        torch.tensor(node_types, dtype=torch.long),
        torch.tensor(edge_types, dtype=torch.long),
    )
    assert np.all(rmse_good_p <= rmse_bad_p + 1e-8)
    assert np.all(rmse_good_c <= rmse_bad_c + 1e-8)
    assert not np.any(np.isnan(rmse_bad_p))
    assert not np.any(np.isnan(rmse_bad_c))
    assert not np.any(np.isnan(rmse_good_p))
    assert not np.any(np.isnan(rmse_good_c))
