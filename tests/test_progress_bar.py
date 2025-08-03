import subprocess
import sys
from pathlib import Path


def _dummy_run(args, extreme_event_prob: float = 0.0):
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

