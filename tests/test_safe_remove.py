import os
from scripts import data_generation


def test_safe_remove_retry(tmp_path, monkeypatch):
    test_file = tmp_path / "dummy.txt"
    test_file.write_text("x")

    call_count = {"n": 0}

    def fake_remove(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise PermissionError
        os.unlink(path)

    monkeypatch.setattr(os, "remove", fake_remove)
    data_generation._safe_remove(str(test_file), retries=2, delay=0)

    assert call_count["n"] == 2
    assert not test_file.exists()
