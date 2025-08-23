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
    template = torch.zeros(num_nodes, 3 + num_pumps)
    pressures = torch.tensor([1.0, 2.0])
    pump_speed = torch.tensor([0.5])

    conv = GCNConv(4, 1)
    model = GNNSurrogate([conv]).eval()
    model.x_mean = torch.randn(4)
    model.x_std = torch.rand(4) + 0.1
    model.y_mean = torch.randn(1)
    model.y_std = torch.rand(1) + 0.1

    feats = template.clone()
    feats[:, 1] = pressures
    feats[:, 3] = pump_speed

    x_norm = prepare_node_features(template, pressures, pump_speed, model)
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


def test_prepare_node_features_per_node_batch_norm():
    torch.manual_seed(0)
    batch_size = 3
    num_nodes = 2
    num_pumps = 1
    template = torch.zeros(num_nodes, 3 + num_pumps)
    pressures = torch.randn(batch_size, num_nodes)
    pump_speed = torch.rand(batch_size, num_pumps)

    conv = GCNConv(4, 1)
    model = GNNSurrogate([conv]).eval()
    model.x_mean = torch.randn(num_nodes, 4)
    model.x_std = torch.rand(num_nodes, 4) + 0.1

    feats = template.expand(batch_size, num_nodes, template.size(1)).clone()
    feats[:, :, 1] = pressures
    feats[:, :, 3:3 + num_pumps] = pump_speed.view(batch_size, 1, -1).expand(
        batch_size, num_nodes, num_pumps
    )

    x_norm = prepare_node_features(template, pressures, pump_speed, model)
    x_manual = (feats - model.x_mean.view(1, num_nodes, -1)) / (
        model.x_std.view(1, num_nodes, -1) + EPS
    )
    assert torch.allclose(x_norm.view(batch_size, num_nodes, -1), x_manual, atol=1e-6)


def test_prepare_node_features_flattened_stats():
    torch.manual_seed(0)
    batch_size = 1
    num_nodes = 2
    num_pumps = 1
    template = torch.zeros(num_nodes, 3 + num_pumps)
    pressures = torch.randn(batch_size, num_nodes)
    pump_speed = torch.rand(batch_size, num_pumps)

    conv = GCNConv(4, 1)
    model = GNNSurrogate([conv]).eval()

    mean2d = torch.randn(num_nodes, 4)
    std2d = torch.rand(num_nodes, 4) + 0.1
    model.x_mean = mean2d.flatten()
    model.x_std = std2d.flatten()

    feats = template.expand(batch_size, num_nodes, template.size(1)).clone()
    feats[:, :, 1] = pressures
    feats[:, :, 3:3 + num_pumps] = pump_speed.view(batch_size, 1, -1).expand(
        batch_size, num_nodes, num_pumps
    )
    expected = (feats - mean2d.view(1, num_nodes, -1)) / (
        std2d.view(1, num_nodes, -1) + EPS
    )

    x_norm = prepare_node_features(template, pressures, pump_speed, model)
    assert torch.allclose(
        x_norm.view(batch_size, num_nodes, -1), expected, atol=1e-6
    )
