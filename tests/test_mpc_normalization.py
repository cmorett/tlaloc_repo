import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import GNNSurrogate, prepare_node_features, EPS
from torch_geometric.nn import GCNConv


def test_mpc_normalization_round_trip_and_consistency():
    torch.manual_seed(0)
    num_nodes = 2
    num_pumps = 1
    template = torch.zeros(num_nodes, 4 + num_pumps)
    pressures = torch.tensor([1.0, 2.0])
    chlorine = torch.tensor([10.0, 20.0])
    pump_speed = torch.tensor([0.5])

    conv = GCNConv(5, 2)
    model = GNNSurrogate([conv]).eval()

    model.x_mean = torch.randn(5)
    model.x_std = torch.rand(5) + 0.1
    model.y_mean = torch.randn(2)
    model.y_std = torch.rand(2) + 0.1

    feats = template.clone()
    feats[:, 1] = pressures
    feats[:, 2] = torch.log1p(chlorine / 1000.0)
    feats[:, 4] = pump_speed

    x_norm = prepare_node_features(template, pressures, chlorine, pump_speed, model)
    x_round = x_norm * (model.x_std + EPS) + model.x_mean
    assert torch.allclose(x_round, feats, atol=1e-6)

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with torch.no_grad():
        out_norm = model(x_norm, edge_index)
        out = out_norm * (model.y_std + EPS) + model.y_mean
        x_manual = (feats - model.x_mean) / (model.x_std + EPS)
        out_manual = model(x_manual, edge_index)
        out_manual = out_manual * (model.y_std + EPS) + model.y_mean
    assert torch.allclose(out, out_manual, atol=1e-6)
