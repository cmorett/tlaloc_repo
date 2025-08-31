import sys
from pathlib import Path
import random
import numpy as np
import wntr

sys.path.append(str(Path(__file__).resolve().parents[1]))
import scripts.data_generation as dg


def test_demand_multiplier_range():
    inp = Path(__file__).resolve().parents[1] / "CTown.inp"

    base_wn = wntr.network.WaterNetworkModel(str(inp))
    base_patterns = {}
    for jname in base_wn.junction_name_list:
        ts = base_wn.get_node(jname).demand_timeseries_list[0]
        if ts.pattern is None:
            base_mult = np.ones(168, dtype=float)
        else:
            base_mult = np.array(ts.pattern.multipliers, dtype=float)
        base_patterns[jname] = base_mult

    random.seed(42)
    np.random.seed(42)
    _, scale_dict, _ = dg._build_randomized_network(str(inp), idx=0)

    for jname, scaled in scale_dict.items():
        base_mult = base_patterns[jname][: len(scaled)]
        ratio = scaled * base_mult.mean() / base_mult
        assert np.all(ratio >= 0.8)
        assert np.all(ratio <= 1.2)


def test_no_demand_scaling_arg(monkeypatch, tmp_path):
    captured = {}

    def fake_run_scenarios(*args, **kwargs):
        captured["demand_scale_range"] = kwargs["demand_scale_range"]
        return []

    monkeypatch.setattr(dg, "run_scenarios", fake_run_scenarios)
    monkeypatch.setattr(dg, "plot_dataset_distributions", lambda *a, **k: None)
    monkeypatch.setattr(dg, "plot_pressure_histogram", lambda *a, **k: None)
    monkeypatch.setattr(dg, "split_results", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(
        dg,
        "build_dataset",
        lambda *a, **k: (np.zeros((1, 1, 1)), np.zeros((1, 1, 1))),
    )
    monkeypatch.setattr(
        dg,
        "build_edge_index",
        lambda wn: (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,)),
            np.zeros((0,)),
            np.zeros((0,)),
        ),
    )

    argv = [
        "prog",
        "--num-scenarios",
        "1",
        "--no-demand-scaling",
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    dg.main()

    assert captured["demand_scale_range"] == (1.0, 1.0)

