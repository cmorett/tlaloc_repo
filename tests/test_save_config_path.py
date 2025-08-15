import sys
from pathlib import Path

import yaml

# Ensure repository root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.reproducibility import save_config


def test_save_config_converts_path(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    some_path = tmp_path / "subdir" / "file.txt"

    save_config(cfg_file, {"my_path": some_path})

    data = yaml.safe_load(cfg_file.read_text())
    assert data["my_path"] == some_path.as_posix()
    assert isinstance(data["my_path"], str)
