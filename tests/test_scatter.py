import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import save_scatter_plots, predicted_vs_actual_scatter


def test_save_scatter_plots(tmp_path):
    p_true = np.array([1.0, 2.0, 3.0])
    p_pred = np.array([1.1, 2.1, 2.9])
    c_true = np.array([0.1, 0.2, 0.3])
    c_pred = np.array([0.1, 0.19, 0.31])
    save_scatter_plots(p_true, p_pred, c_true, c_pred, "unit", plots_dir=tmp_path)
    assert (tmp_path / "pred_vs_actual_pressure_unit.png").exists()
    assert (tmp_path / "pred_vs_actual_chlorine_unit.png").exists()
    assert (tmp_path / "residual_scatter_pressure_unit.png").exists()
    assert (tmp_path / "residual_scatter_chlorine_unit.png").exists()


def test_save_scatter_plots_mask(tmp_path):
    p_true = np.array([1.0, 2.0, 3.0, 4.0])
    p_pred = np.array([1.0, 2.1, 3.0, 4.2])
    c_true = np.array([0.1, 0.2, 0.3, 0.4])
    c_pred = np.array([0.1, 0.21, 0.31, 0.41])
    mask = [True, False, True, False]
    save_scatter_plots(
        p_true,
        p_pred,
        c_true,
        c_pred,
        "unit",
        plots_dir=tmp_path,
        mask=mask,
    )
    assert (tmp_path / "pred_vs_actual_pressure_unit.png").exists()
    assert (tmp_path / "pred_vs_actual_chlorine_unit.png").exists()
    assert (tmp_path / "residual_scatter_pressure_unit.png").exists()
    assert (tmp_path / "residual_scatter_chlorine_unit.png").exists()
