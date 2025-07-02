from pathlib import Path

from scripts.train_gnn import predicted_vs_actual_scatter
from scripts.mpc_control import plot_convergence_curve


def test_predicted_vs_actual_scatter(tmp_path: Path):
    fig = predicted_vs_actual_scatter(
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 2.9],
        [0.1, 0.2, 0.3],
        [0.1, 0.19, 0.31],
        "unit",
        plots_dir=tmp_path,
        return_fig=True,
    )
    assert (tmp_path / "pred_vs_actual_unit.png").exists()
    # verify axis labels
    ax0 = fig.axes[0]
    assert ax0.get_xlabel() == "Actual Pressure (m)"
    plt = None


def test_predicted_vs_actual_scatter_mask(tmp_path: Path):
    fig = predicted_vs_actual_scatter(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.1, 2.9, 4.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.21, 0.29, 0.39],
        "unit",
        plots_dir=tmp_path,
        return_fig=True,
        mask=[True, False, True, False],
    )
    assert (tmp_path / "pred_vs_actual_unit.png").exists()
    ax0 = fig.axes[0]
    assert ax0.collections[0].get_offsets().shape[0] == 2


def test_convergence_curve(tmp_path: Path):
    fig = plot_convergence_curve([5, 4, 3, 2, 1], "unit", plots_dir=tmp_path, return_fig=True)
    assert (tmp_path / "mpc_convergence_unit.png").exists()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Optimization Iteration"

