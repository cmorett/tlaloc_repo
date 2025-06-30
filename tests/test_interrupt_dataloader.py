import numpy as np
import torch
from torch.utils.data import DataLoader as TorchLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import train_gnn
from scripts.train_gnn import SequenceDataset, MultiTaskGNNSurrogate, train_sequence


def test_train_sequence_dataloader_interrupt():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0, 0.5, 100.0], [1.0, 0.5, 100.0]], dtype=torch.float32)
    T, N, E = 1, 2, 2
    X = np.ones((1, T, N, 4), dtype=np.float32)
    Y = np.array([
        {
            "node_outputs": np.zeros((T, N, 2), dtype=np.float32),
            "edge_outputs": np.zeros((T, E), dtype=np.float32),
        }
    ], dtype=object)
    ds = SequenceDataset(X, Y, edge_index.numpy(), edge_attr.numpy())

    class FailingLoader(TorchLoader):
        def __iter__(self):
            it = super().__iter__()

            def gen():
                yield next(it)
                raise RuntimeError("DataLoader worker (pid(s) 1) exited unexpectedly")

            return gen()

    loader = FailingLoader(ds, batch_size=1)
    model = MultiTaskGNNSurrogate(
        in_channels=4,
        hidden_channels=4,
        edge_dim=3,
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

    train_gnn.interrupted = True
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
        node_mask=None,
    )
    assert isinstance(loss_tuple, tuple)

