import csv
from argparse import Namespace
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import append_pressure_stats


def test_append_pressure_stats(tmp_path):
    args = Namespace(
        seed=123,
        deterministic=True,
        num_workers=2,
        sequence_length=4,
        fixed_pump_speed=1.0,
        demand_scale_range=(0.5, 1.5),
        extreme_rate=0.05,
        pump_outage_rate=0.1,
        local_surge_rate=0.2,
        tank_level_range=(0.2, 0.8),
        no_demand_scaling=False,
        show_progress=False,
        output_dir=tmp_path,
    )

    stats_path = tmp_path / "pressure_stats.csv"
    append_pressure_stats(
        stats_path,
        args,
        num_scenarios=10,
        mean_pressure=50.0,
        std_pressure=5.0,
        timestamp="20240101_000000",
    )

    with open(stats_path) as f:
        row = list(csv.DictReader(f))[0]

    assert row["sequence_length"] == "4"
    assert row["deterministic"] == "True"
    assert row["num_workers"] == "2"
    assert row["seed"] == "123"
    assert row["tank_min"] == "0.2"
    assert row["tank_max"] == "0.8"


def test_append_pressure_stats_readonly_file(tmp_path):
    args = Namespace(
        seed=123,
        deterministic=True,
        num_workers=2,
        sequence_length=4,
        fixed_pump_speed=1.0,
        demand_scale_range=(0.5, 1.5),
        extreme_rate=0.05,
        pump_outage_rate=0.1,
        local_surge_rate=0.2,
        tank_level_range=(0.2, 0.8),
        no_demand_scaling=False,
        show_progress=False,
        output_dir=tmp_path,
    )

    stats_path = tmp_path / "pressure_stats.csv"
    stats_path.write_text(
        ",".join(
            [
                "timestamp",
                "num_scenarios",
                "seed",
                "deterministic",
                "num_workers",
                "sequence_length",
                "fixed_pump_speed",
                "demand_min",
                "demand_max",
                "extreme_rate",
                "pump_outage_rate",
                "local_surge_rate",
                "tank_min",
                "tank_max",
                "no_demand_scaling",
                "show_progress",
                "output_dir",
                "mean_pressure",
                "std_pressure",
            ]
        )
        + "\n"
    )
    stats_path.chmod(0o444)

    append_pressure_stats(
        stats_path,
        args,
        num_scenarios=10,
        mean_pressure=50.0,
        std_pressure=5.0,
        timestamp="20240101_000000",
    )

    with open(stats_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1


def test_append_pressure_stats_permission_error(tmp_path, monkeypatch):
    args = Namespace(
        seed=123,
        deterministic=True,
        num_workers=2,
        sequence_length=4,
        fixed_pump_speed=1.0,
        demand_scale_range=(0.5, 1.5),
        extreme_rate=0.05,
        pump_outage_rate=0.1,
        local_surge_rate=0.2,
        tank_level_range=(0.2, 0.8),
        no_demand_scaling=False,
        show_progress=False,
        output_dir=tmp_path,
    )

    stats_path = tmp_path / "pressure_stats.csv"

    def _raise(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr("builtins.open", _raise)

    with pytest.raises(RuntimeError):
        append_pressure_stats(
            stats_path,
            args,
            num_scenarios=10,
            mean_pressure=50.0,
            std_pressure=5.0,
            timestamp="20240101_000000",
        )

    assert stats_path.exists()
