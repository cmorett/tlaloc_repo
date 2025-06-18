import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import handle_keyboard_interrupt


def test_handle_keyboard_interrupt(capsys):
    handle_keyboard_interrupt("foo/bar.pth")
    captured = capsys.readouterr()
    assert "Training interrupted" in captured.out
    assert "foo/bar.pth" in captured.out
