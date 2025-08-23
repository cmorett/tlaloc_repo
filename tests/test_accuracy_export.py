import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import RunningStats, save_accuracy_metrics


def test_save_accuracy_metrics(tmp_path):
    true_p = np.array([10.0, 12.0])
    pred_p = np.array([9.5, 12.5])

    p_stats = RunningStats()
    p_stats.update(pred_p, true_p)

    save_accuracy_metrics(p_stats, "unit", logs_dir=tmp_path)
    f = tmp_path / "accuracy_unit.csv"
    assert f.exists()
    df = pd.read_csv(f, index_col=0)
    assert "Pressure (m)" in df.columns
    assert df.shape[1] == 1
