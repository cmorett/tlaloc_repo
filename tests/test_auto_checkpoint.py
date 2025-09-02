import torch
import pytest
from scripts.train_gnn import _forward_with_auto_checkpoint

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_checkpoint = False
        self.called = 0

    def forward(self):
        self.called += 1
        if self.called == 1 and not self.use_checkpoint:
            raise RuntimeError("CUDA out of memory")
        return torch.tensor(1.0)

def test_forward_enables_checkpoint_on_oom():
    model = DummyModel()
    def fn():
        return model()
    out = _forward_with_auto_checkpoint(model, fn)
    assert model.use_checkpoint
    assert out.item() == 1.0
    assert model.called == 2
