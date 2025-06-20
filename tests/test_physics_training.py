import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import (
    SequenceDataset,
    MultiTaskGNNSurrogate,
    train_sequence,
)


def test_train_sequence_with_physics_losses():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1.0, 0.5, 100.0], [1.0, 0.5, 100.0]], dtype=torch.float32
    )
    T = 2
    N = 2
    E = 2
    X = np.ones((1, T, N, 4), dtype=np.float32)
    Y = np.array(
        [
            {
                "node_outputs": np.zeros((T, N, 2), dtype=np.float32),
                "edge_outputs": np.zeros((T, E), dtype=np.float32),
            }
        ],
        dtype=object,
    )
    ds = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())
    loader = TorchLoader(ds, batch_size=1)
    model = MultiTaskGNNSurrogate(
        in_channels=4,
        hidden_channels=8,
        edge_dim=3,
        node_output_dim=2,
        edge_output_dim=1,
        num_layers=2,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=4,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_tuple = train_sequence(
        model,
        loader,
        ds.edge_index,
        ds.edge_attr,
        edge_attr,
        None,
        None,
        opt,
        torch.device("cpu"),
        physics_loss=True,
        pressure_loss=True,
        node_mask=None,
    )
    # mass and head losses should be finite numbers
    assert torch.isfinite(torch.tensor(loss_tuple[3]))
    assert torch.isfinite(torch.tensor(loss_tuple[4]))
