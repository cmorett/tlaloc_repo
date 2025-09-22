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
    template = torch.zeros(num_nodes, 5 + num_pumps)
    template[:, 4] = torch.tensor([1.0, 2.0])
    template[:, 5] = torch.tensor([-1.0, 1.0])
    pressures = torch.tensor([1.0, 2.0])
    chlorine = torch.tensor([10.0, 20.0])
    pump_speed = torch.tensor([0.5])
    node_type = torch.zeros(num_nodes, dtype=torch.long)

    conv = GCNConv(6, 2)
    model = GNNSurrogate([conv]).eval()

    model.x_mean = torch.randn(6)
    model.x_std = torch.rand(6) + 0.1
    model.y_mean = torch.randn(2)
    model.y_std = torch.rand(2) + 0.1

    feats = template.clone()
    feats[:, 1] = pressures
    feats[:, 2] = torch.log1p(chlorine / 1000.0)
    feats[:, 3] = pressures + template[:, 4]
    feats[:, 5] = template[:, 5] * pump_speed

    x_norm = prepare_node_features(
        template, pressures, chlorine, pump_speed, model, node_type=node_type
    )
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
    template = torch.zeros(num_nodes, 5 + num_pumps)
    template[:, 4] = torch.tensor([1.0, 2.0])
    template[:, 5] = torch.tensor([-1.0, 1.0])
    pressures = torch.randn(batch_size, num_nodes)
    chlorine = torch.rand(batch_size, num_nodes)
    pump_speed = torch.rand(batch_size, num_pumps)
    node_type = torch.zeros(num_nodes, dtype=torch.long)

    conv = GCNConv(6, 2)
    model = GNNSurrogate([conv]).eval()
    model.x_mean = torch.randn(num_nodes, 6)
    model.x_std = torch.rand(num_nodes, 6) + 0.1

    feats = template.expand(batch_size, num_nodes, template.size(1)).clone()
    feats[:, :, 1] = pressures
    feats[:, :, 2] = torch.log1p(chlorine / 1000.0)
    feats[:, :, 3] = pressures + template[:, 4]
    pump_layout = template[:, 5 : 5 + num_pumps]
    feats[:, :, 5 : 5 + num_pumps] = pump_layout.unsqueeze(0) * pump_speed.view(
        batch_size, 1, -1
    )

    x_norm = prepare_node_features(
        template, pressures, chlorine, pump_speed, model, node_type=node_type
    )
    x_manual = (feats - model.x_mean.view(1, num_nodes, -1)) / (
        model.x_std.view(1, num_nodes, -1) + EPS
    )
    assert torch.allclose(x_norm.view(batch_size, num_nodes, -1), x_manual, atol=1e-6)


def test_prepare_node_features_flattened_stats():
    torch.manual_seed(0)
    batch_size = 1
    num_nodes = 2
    num_pumps = 1
    template = torch.zeros(num_nodes, 5 + num_pumps)
    template[:, 4] = torch.tensor([1.0, 2.0])
    template[:, 5] = torch.tensor([-1.0, 1.0])
    pressures = torch.randn(batch_size, num_nodes)
    chlorine = torch.rand(batch_size, num_nodes)
    pump_speed = torch.rand(batch_size, num_pumps)
    node_type = torch.zeros(num_nodes, dtype=torch.long)

    conv = GCNConv(6, 2)
    model = GNNSurrogate([conv]).eval()

    mean2d = torch.randn(num_nodes, 6)
    std2d = torch.rand(num_nodes, 6) + 0.1
    model.x_mean = mean2d.flatten()
    model.x_std = std2d.flatten()

    feats = template.expand(batch_size, num_nodes, template.size(1)).clone()
    feats[:, :, 1] = pressures
    feats[:, :, 2] = torch.log1p(chlorine / 1000.0)
    feats[:, :, 3] = pressures + template[:, 4]
    pump_layout = template[:, 5 : 5 + num_pumps]
    feats[:, :, 5 : 5 + num_pumps] = pump_layout.unsqueeze(0) * pump_speed.view(
        batch_size, 1, -1
    )
    expected = (feats - mean2d.view(1, num_nodes, -1)) / (
        std2d.view(1, num_nodes, -1) + EPS
    )

    x_norm = prepare_node_features(
        template, pressures, chlorine, pump_speed, model, node_type=node_type
    )
    assert torch.allclose(
        x_norm.view(batch_size, num_nodes, -1), expected, atol=1e-6
    )
