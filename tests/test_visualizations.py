from pathlib import Path
import numpy as np
import torch

from scripts.train_gnn import (
    plot_loss_components,
    SequenceDataset,
    plot_sequence_prediction,
    correlation_heatmap,
    pred_vs_actual_scatter,
    plot_error_histogram,
)
from scripts.mpc_control import plot_convergence_curve




def test_convergence_curve(tmp_path: Path):
    fig = plot_convergence_curve([5, 4, 3, 2, 1], "unit", plots_dir=tmp_path, return_fig=True)
    assert (tmp_path / "mpc_convergence_unit.png").exists()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Optimization Iteration"


def test_plot_loss_components(tmp_path: Path):
    comps = [
        (1.0, 2.0, 3.0, 4.0, 5.0),
        (0.5, 1.5, 2.5, 3.5, 4.5),
    ]
    plot_loss_components(comps, "unit", plots_dir=tmp_path)
    assert (tmp_path / "loss_components_unit.png").exists()




def test_plot_sequence_prediction(tmp_path: Path):
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 3), dtype=torch.float32)
    X = np.zeros((1, 2, 2, 4), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((2, 2, 2), dtype=np.float32),
                "edge_outputs": np.zeros((2, 2), dtype=np.float32),
            }
        ],
        dtype=object,
    )
    ds = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())

    class Dummy(torch.nn.Module):
        def forward(self, X_seq, ei, ea, nt, et):
            B, T, N, _ = X_seq.size()
            return {"node_outputs": torch.zeros(B, T, N, 2) + self.dummy}

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

    model = Dummy()
    model.y_mean = {"node_outputs": torch.zeros(2)}
    model.y_std = {"node_outputs": torch.ones(2)}

    plot_sequence_prediction(model, ds, "unit", plots_dir=tmp_path)
    assert (tmp_path / "time_series_example_unit.png").exists()


def test_plot_sequence_prediction_single_step(tmp_path: Path):
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 3), dtype=torch.float32)
    X = np.zeros((1, 1, 2, 4), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((1, 2, 2), dtype=np.float32),
                "edge_outputs": np.zeros((1, 2), dtype=np.float32),
            }
        ],
        dtype=object,
    )
    ds = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())

    class Dummy(torch.nn.Module):
        def forward(self, X_seq, ei, ea, nt, et):
            B, T, N, _ = X_seq.size()
            return {"node_outputs": torch.zeros(B, T, N, 2) + self.dummy}

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

    model = Dummy()
    model.y_mean = {"node_outputs": torch.zeros(2)}
    model.y_std = {"node_outputs": torch.ones(2)}

    plot_sequence_prediction(model, ds, "unit1", plots_dir=tmp_path)
    assert (tmp_path / "time_series_example_unit1.png").exists()


def test_pred_vs_actual_scatter(tmp_path: Path):
    pred = np.linspace(0, 1, 10)
    true = np.linspace(0, 1, 10)
    pred_vs_actual_scatter(pred, true, "unit", plots_dir=tmp_path)
    assert (tmp_path / "pred_vs_actual_pressure_unit.png").exists()
    plot_error_histogram(pred - true, "unit", plots_dir=tmp_path)
    assert (tmp_path / "error_histograms_unit.png").exists()


def test_correlation_heatmap(tmp_path: Path):
    mat = np.random.randn(8, 3)
    labels = ["a", "b", "c"]
    correlation_heatmap(mat, labels, "unit", plots_dir=tmp_path)
    assert (tmp_path / "correlation_heatmap_unit.png").exists()

