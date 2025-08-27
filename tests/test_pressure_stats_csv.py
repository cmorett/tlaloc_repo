import csv
from argparse import Namespace
from pathlib import Path
import sys

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
