import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import (
    SequenceDataset,
    MultiTaskGNNSurrogate,
    train_sequence,
    build_loss_mask,
)
import wntr


def test_reservoir_node_excluded_from_loss():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0, 0.5, 100.0, 1.0], [1.0, 0.5, 100.0, 1.0]], dtype=torch.float32)
    T = 1
    N = 2
    X = np.ones((1, T, N, 4), dtype=np.float32)
    Y = np.array([
        {
            "node_outputs": np.array([[[1.0, 0.0], [3.0, 0.0]]], dtype=np.float32),
            "edge_outputs": np.zeros((T, edge_index.size(1)), dtype=np.float32),
        }
    ], dtype=object)
    ds = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())
    loader = TorchLoader(ds, batch_size=1)
    model = MultiTaskGNNSurrogate(
        in_channels=4,
        hidden_channels=4,
        edge_dim=4,
        node_output_dim=2,
        edge_output_dim=1,
        num_layers=1,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=4,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    mask = torch.tensor([False, True])

    X_seq, Y_seq = ds[0]
    with torch.no_grad():
        pred = model(X_seq.unsqueeze(0), ds.edge_index, ds.edge_attr, None, None)
    expected_p = F.mse_loss(
        pred["node_outputs"][:, :, mask, 0],
        Y_seq["node_outputs"].unsqueeze(0)[:, :, mask, 0],
    ).item()
    loss_tuple = train_sequence(
        model,
        loader,
        ds.edge_index,
        ds.edge_attr,
        edge_attr,
        None,
        None,
        [(0, 1)],
        opt,
        torch.device("cpu"),
        physics_loss=False,
        pressure_loss=False,
        node_mask=mask,
        loss_fn="mse",
    )
    assert abs(loss_tuple[1] - expected_p) < 1e-6


def test_build_loss_mask_ctown():
    wn = wntr.network.WaterNetworkModel('CTown.inp')
    mask = build_loss_mask(wn)
    res_idx = [i for i, n in enumerate(wn.node_name_list) if n in wn.reservoir_name_list]
    for idx in res_idx:
        assert not mask[idx]
    tank_idx = [i for i, n in enumerate(wn.node_name_list) if n in wn.tank_name_list]
    for idx in tank_idx:
        assert not mask[idx]
    assert mask.dtype == torch.bool

