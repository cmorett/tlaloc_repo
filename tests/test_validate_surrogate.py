import os
import torch
import wntr
import sys
from pathlib import Path

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

    def forward(self, x, edge_index, edge_attr=None):
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
        model, edge_index, None, wn, [(res, {})], device, "test"
    )
    assert "pressure_rmse" in metrics
    assert arr.shape[1] == len(wn.node_name_list)
