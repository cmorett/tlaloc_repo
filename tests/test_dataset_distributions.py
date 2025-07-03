from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import plot_dataset_distributions


def test_plot_dataset_distributions(tmp_path: Path):
    plot_dataset_distributions([0.8, 1.0, 1.2], [0.0, 0.5, 1.0], "unit", plots_dir=tmp_path)
    assert (tmp_path / "dataset_distributions_unit.png").exists()
