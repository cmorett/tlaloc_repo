import csv
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import SequenceDataset, train_sequence, evaluate_sequence


class DummyModel(torch.nn.Module):
    def __init__(self, node_pred, edge_pred, y_mean, y_std):
        super().__init__()
        self.node_pred = node_pred
        self.edge_pred = edge_pred
        self.y_mean = y_mean
        self.y_std = y_std
        # parameter to keep optimizer non-empty
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X_seq, edge_index, edge_attr, nt, et):
        node_out = self.node_pred + 0 * self.dummy
        edge_out = self.edge_pred + 0 * self.dummy
        return {"node_outputs": node_out, "edge_outputs": edge_out}


def test_pressure_mae_logging(tmp_path):
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 3), dtype=torch.float32)
    T, N, E = 1, 2, 2
    X = np.zeros((1, T, N, 4), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((T, N, 2), dtype=np.float32),
                "edge_outputs": np.zeros((T, E), dtype=np.float32),
            }
        ],
        dtype=object,
    )
    dataset = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())
    loader = TorchLoader(dataset, batch_size=1)

    node_pred = torch.zeros(1, T, N, 2)
    node_pred[..., 0] = 0.5  # normalized pressure prediction
    edge_pred = torch.zeros(1, T, E, 1)
    y_mean = {"node_outputs": torch.tensor([[10.0, 0.0], [20.0, 0.0]]), "edge_outputs": torch.zeros(E)}
    y_std = {"node_outputs": torch.tensor([[2.0, 1.0], [2.0, 1.0]]), "edge_outputs": torch.ones(E)}
    model = DummyModel(node_pred, edge_pred, y_mean, y_std)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    device = torch.device("cpu")
    loss_tuple = train_sequence(
        model,
        loader,
        dataset.edge_index,
        dataset.edge_attr,
        dataset.edge_attr,
        None,
        None,
        [],
        optimizer,
        device,
        w_flow=0.0,
        progress=False,
    )
    eval_tuple = evaluate_sequence(
        model,
        loader,
        dataset.edge_index,
        dataset.edge_attr,
        dataset.edge_attr,
        None,
        None,
        [],
        device,
        w_flow=0.0,
        progress=False,
    )

    press_mae = loss_tuple[9]
    assert abs(press_mae - 1.0) < 1e-6
    assert abs(eval_tuple[9] - 1.0) < 1e-6

    log_path = tmp_path / "log.csv"
    with open(log_path, "w") as f:
        f.write(
            "epoch,train_loss,val_loss,press_loss,flow_loss,mass_imbalance,head_violation,press_mae,val_press_loss,val_flow_loss,val_mass_imbalance,val_head_violation,val_press_mae,lr\n"
        )
        f.write(
            f"0,{loss_tuple[0]:.6f},{eval_tuple[0]:.6f},{loss_tuple[1]:.6f},{loss_tuple[2]:.6f},0,0,{loss_tuple[9]:.6f},{eval_tuple[1]:.6f},{eval_tuple[2]:.6f},0,0,{eval_tuple[9]:.6f},0\n"
        )
    with open(log_path) as f:
        row = list(csv.DictReader(f))[0]
    assert "press_mae" in row and "val_press_mae" in row
    assert abs(float(row["press_mae"]) - 1.0) < 1e-6
    assert abs(float(row["val_press_mae"]) - 1.0) < 1e-6
