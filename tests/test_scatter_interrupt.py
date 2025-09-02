import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import scripts.train_gnn as train_gnn


def test_scatter_plots_generated_when_interrupted(tmp_path):
    train_gnn.interrupted = True
    try:
        p_true = np.array([1.0, 2.0])
        p_pred = np.array([1.1, 2.1])
        c_true = np.array([0.1, 0.2])
        c_pred = np.array([0.11, 0.19])
        train_gnn.save_scatter_plots(p_true, p_pred, c_true, c_pred, "unit", plots_dir=tmp_path)
        assert (tmp_path / "pred_vs_actual_pressure_unit.png").exists()
        assert (tmp_path / "pred_vs_actual_chlorine_unit.png").exists()
        assert (tmp_path / "residual_scatter_pressure_unit.png").exists()
        assert (tmp_path / "residual_scatter_chlorine_unit.png").exists()
    finally:
        train_gnn.interrupted = False
