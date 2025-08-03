import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import handle_keyboard_interrupt


def test_handle_keyboard_interrupt(tmp_path, capsys):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer)
    ckpt = tmp_path / "ckpt.pth"
    handle_keyboard_interrupt(str(ckpt), model, optimizer, scheduler, 5)
    captured = capsys.readouterr()
    assert "Training interrupted" in captured.out
    assert str(ckpt) in captured.out
    state = torch.load(ckpt)
    assert state["epoch"] == 5
    assert "model_state_dict" in state
    assert "optimizer_state_dict" in state
    assert "scheduler_state_dict" in state
