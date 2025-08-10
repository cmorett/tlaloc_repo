import subprocess
import sys
from pathlib import Path


def _dummy_run(args, **kwargs):
    """Return a trivial result tuple for fast testing."""
    return (None, {}, {})


def test_run_scenarios_show_progress(monkeypatch):
    import scripts.data_generation as dg

    monkeypatch.setattr(dg, "_run_single_scenario", _dummy_run)
    results = dg.run_scenarios("foo", 3, num_workers=2, show_progress=True)
    assert len(results) == 3


def test_run_scenarios_show_progress_no_tqdm(monkeypatch):
    import scripts.data_generation as dg

    monkeypatch.setattr(dg, "_run_single_scenario", _dummy_run)
    monkeypatch.setattr(dg, "tqdm", None)
    results = dg.run_scenarios("foo", 2, num_workers=2, show_progress=True)
    assert len(results) == 2


def test_data_generation_cli_show_progress(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo / "scripts/data_generation.py"),
        "--num-scenarios",
        "0",
        "--output-dir",
        str(tmp_path),
        "--show-progress",
    ]
    subprocess.run(cmd, check=True)


def test_train_progress_flag(monkeypatch):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import scripts.train_gnn as tg

    calls = []

    class DummyTqdm:
        def __init__(self, it, disable=False):
            calls.append(disable)
            self.it = it

        def __iter__(self):
            return iter(self.it)

    monkeypatch.setattr(tg, "tqdm", DummyTqdm)
    model = tg.GCNEncoder(1, 1, 1, num_layers=1)
    data = Data(x=torch.zeros((1, 1)), edge_index=torch.tensor([[0], [0]]), y=torch.zeros((1, 1)))
    loader = DataLoader([data], batch_size=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    tg.train(model, loader, opt, torch.device("cpu"), check_negative=False, progress=True)
    tg.train(model, loader, opt, torch.device("cpu"), check_negative=False, progress=False)
    assert calls == [False, True]

