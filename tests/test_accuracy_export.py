import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import save_accuracy_metrics

def test_save_accuracy_metrics(tmp_path):
    true_p = np.array([10.0, 12.0])
    pred_p = np.array([9.5, 12.5])
    true_c = np.log1p(np.array([0.5, 0.4]))
    pred_c = np.log1p(np.array([0.45, 0.6]))
    save_accuracy_metrics(true_p, pred_p, true_c, pred_c, "unit", logs_dir=tmp_path)
    f = tmp_path / "accuracy_unit.csv"
    assert f.exists()
    df = pd.read_csv(f, index_col=0)
    assert "Pressure (m)" in df.columns
    assert "Chlorine (mg/L)" in df.columns
