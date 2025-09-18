import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.feature_utils import SequenceDataset
from scripts.train_gnn import evaluate_sequence


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # two nodes with individual stats
        self.y_mean = {
            "node_outputs": torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32),
            "edge_outputs": torch.zeros(1, dtype=torch.float32),
        }
        self.y_std = {
            "node_outputs": torch.ones((2, 2), dtype=torch.float32),
            "edge_outputs": torch.ones(1, dtype=torch.float32),
        }

    def forward(self, X_seq, edge_index, edge_attr, node_type, edge_type):
        batch_size, T, num_nodes, _ = X_seq.shape
        edge_count = edge_index.size(1)
        node_out = torch.zeros(batch_size, T, num_nodes, 2, device=X_seq.device)
        edge_out = torch.zeros(batch_size, T, edge_count, 1, device=X_seq.device)
        return {"node_outputs": node_out, "edge_outputs": edge_out}


def test_evaluate_sequence_applies_node_mask_to_stats():
    X = np.zeros((1, 1, 2, 2), dtype=np.float32)
    Y = np.array([
        {
            "node_outputs": np.zeros((1, 2, 2), dtype=np.float32),
            "edge_outputs": np.zeros((1, 1), dtype=np.float32),
        }
    ], dtype=object)
    edge_index = np.array([[0], [1]])
    edge_attr = np.zeros((1, 10), dtype=np.float32)
    dataset = SequenceDataset(X, Y, edge_index=edge_index, edge_attr=edge_attr)
    loader = DataLoader(dataset, batch_size=1)

    model = DummyModel()
    device = torch.device("cpu")
    edge_attr_phys = torch.ones_like(dataset.edge_attr)

    node_mask = torch.tensor([True, False])
    result = evaluate_sequence(
        model,
        loader,
        dataset.edge_index,
        dataset.edge_attr,
        edge_attr_phys,
        None,
        None,
        [],
        device,
        node_mask=node_mask,
        pressure_loss=True,
        progress=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 11
