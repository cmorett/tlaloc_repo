import os
import os
import torch
import wntr
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "scripts"))
TEMP_DIR = REPO_ROOT / "data" / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
from scripts.mpc_control import load_network
from scripts.experiments_validation import rollout_surrogate


class ExtraOutputModel(torch.nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.i = 0
        self.x_mean = None
        self.x_std = torch.ones(1)
        self.y_mean = torch.zeros(2)
        self.y_std = torch.ones(2)

    def forward(self, x, edge_index, edge_attr=None, node_types=None, edge_types=None):
        out = self.seq[self.i].to(x.device)
        self.i += 1
        extra = torch.full((out.size(0), 1), 123.0, device=x.device)
        return torch.cat([out, extra], dim=1)


def test_rollout_surrogate_ignores_excess_outputs():
    device = torch.device("cpu")
    wn, node_to_index, pump_names, edge_index, node_types, edge_types = load_network("CTown.inp")
    wn.options.time.duration = 3 * 3600
    wn.options.time.hydraulic_timestep = wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim(str(TEMP_DIR / "extra_outputs"))

    p_df = res.node["pressure"].clip(lower=5.0)
    c_df = res.node["quality"]
    steps = 3
    seq = []
    for i in range(steps):
        p = p_df.iloc[i + 1].to_numpy()
        c = np.log1p(c_df.iloc[i + 1].to_numpy() / 1000.0)
        seq.append(np.column_stack([p, c]))
    seq = np.stack(seq)

    model = ExtraOutputModel(seq).to(device)
    rmse_p = rollout_surrogate(
        model,
        edge_index,
        None,
        wn,
        [res],
        device,
        steps,
        torch.tensor(node_types, dtype=torch.long),
        torch.tensor(edge_types, dtype=torch.long),
    )
    assert np.allclose(rmse_p, 0.0, atol=1e-6)
