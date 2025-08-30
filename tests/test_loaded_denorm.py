import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import load_surrogate_model
from models.gnn_surrogate import MultiTaskGNNSurrogate

def test_loaded_model_denormalizes(tmp_path):
    model = MultiTaskGNNSurrogate(
        in_channels=1,
        hidden_channels=4,
        edge_dim=1,
        node_output_dim=1,
        edge_output_dim=1,
        num_layers=1,
        use_attention=False,
        gat_heads=1,
        dropout=0.0,
        residual=False,
        rnn_hidden_dim=4,
    )
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_meta": {
            "model_class": "MultiTaskGNNSurrogate",
            "in_channels": 1,
            "hidden_dim": 4,
            "num_layers": 1,
            "use_attention": False,
            "gat_heads": 1,
            "residual": False,
            "dropout": 0.0,
            "activation": "relu",
            "node_output_dim": 1,
            "edge_output_dim": 1,
            "rnn_hidden_dim": 4,
            "edge_dim": 1,
        },
    }
    ckpt_path = tmp_path / "model.pth"
    torch.save(ckpt, ckpt_path)
    np.savez(
        tmp_path / "model_norm.npz",
        x_mean=np.zeros(1),
        x_std=np.ones(1),
        y_mean_node=np.array([2.0]),
        y_std_node=np.array([4.0]),
        y_mean_edge=np.array([1.0]),
        y_std_edge=np.array([3.0]),
    )
    loaded = load_surrogate_model(torch.device("cpu"), path=str(ckpt_path), use_jit=False)
    assert torch.allclose(loaded.y_mean, torch.tensor([2.0]))
    assert torch.allclose(loaded.y_std, torch.tensor([4.0]))
    assert torch.allclose(loaded.y_mean_edge, torch.tensor([1.0]))
    assert torch.allclose(loaded.y_std_edge, torch.tensor([3.0]))
    X_seq = torch.zeros(1, 1, 2, 1)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros(2, 1)
    out = loaded(X_seq, edge_index, edge_attr)
    node_pred = out["node_outputs"] * loaded.y_std + loaded.y_mean
    edge_pred = out["edge_outputs"] * loaded.y_std_edge + loaded.y_mean_edge
    assert torch.allclose(node_pred, torch.full_like(node_pred, 2.0))
    assert torch.allclose(edge_pred, torch.full_like(edge_pred, 1.0))
