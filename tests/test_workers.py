import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def test_workers_dataloader():
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    d = Data(x=torch.ones(2, 2), edge_index=edge, y=torch.zeros(2, 1))
    loader = DataLoader(
        [d, d],
        batch_size=1,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    for _ in loader:
        pass
