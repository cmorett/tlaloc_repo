import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import scripts.data_generation as dg  # noqa: E402


def test_temp_files_cleanup_on_failure(tmp_path, monkeypatch):
    # Redirect temporary files to the pytest-provided directory
    monkeypatch.setattr(dg, "TEMP_DIR", tmp_path)

    class FailingSim:
        def __init__(self, wn):
            pass

        def run_sim(self, file_prefix):
            Path(f"{file_prefix}.rpt").touch()
            raise dg.wntr.epanet.exceptions.EpanetException("fail")

    monkeypatch.setattr(dg.wntr.sim, "EpanetSimulator", FailingSim)

    res = dg._run_single_scenario((0, str(dg.REPO_ROOT / "CTown.inp"), 42))
    assert res is None
    assert list(tmp_path.iterdir()) == []
