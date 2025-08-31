import warnings
import subprocess
import numpy as np
import sys
from pathlib import Path
import torch
from torch.nn import MultiheadAttention
from torch_geometric.nn import GATConv
import wntr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import build_edge_attr
from models.gnn_surrogate import EnhancedGNNEncoder, HydroConv


def test_attention_keeps_hydroconv_by_default():
    model = EnhancedGNNEncoder(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_layers=1,
        edge_dim=2,
        use_attention=True,
        gat_heads=1,
    )
    assert isinstance(model.convs[0], HydroConv)
    assert isinstance(model.attentions[0], MultiheadAttention)


def test_attention_warns_when_disabling_hydro():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = EnhancedGNNEncoder(
            in_channels=3,
            hidden_channels=4,
            out_channels=2,
            num_layers=1,
            edge_dim=2,
            use_attention=True,
            attention_after_hydro=False,
            gat_heads=1,
        )
        assert any("HydroConv disabled" in str(wi.message) for wi in w)
        assert isinstance(model.convs[0], GATConv)


def test_cli_use_attention_keeps_hydro(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    log_file = data_dir / "training_attn.log"
    if log_file.exists():
        log_file.unlink()

    wn = wntr.network.WaterNetworkModel(repo / "CTown.inp")
    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    link = wn.get_link(wn.link_name_list[0])
    edge_index = np.array(
        [[node_map[link.start_node.name], node_map[link.end_node.name]],
         [node_map[link.end_node.name], node_map[link.start_node.name]]],
        dtype=np.int64,
    )
    edge_attr = build_edge_attr(wn, edge_index)

    np.save(tmp_path / "edge_index.npy", edge_index)
    np.save(tmp_path / "edge_attr.npy", edge_attr)

    F = 3 + len(wn.pump_name_list)
    N = len(wn.node_name_list)
    X = np.ones((1, N, F), dtype=np.float32)
    Y = np.zeros((1, N, 1), dtype=np.float32)
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "Y.npy", Y)

    cmd = [
        "python",
        str(repo / "scripts/train_gnn.py"),
        "--x-path",
        str(tmp_path / "X.npy"),
        "--y-path",
        str(tmp_path / "Y.npy"),
        "--edge-index-path",
        str(tmp_path / "edge_index.npy"),
        "--edge-attr-path",
        str(tmp_path / "edge_attr.npy"),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--run-name",
        "attn",
        "--output",
        str(tmp_path / "model.pth"),
        "--use-attention",
    ]

    subprocess.run(cmd, check=True)

    assert log_file.exists()
    log_text = log_file.read_text()
    assert "'attention_after_hydro': True" in log_text
