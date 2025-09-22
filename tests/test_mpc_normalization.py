import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.mpc_control import GNNSurrogate, prepare_node_features, EPS
from scripts.train_gnn import evaluate_sequence, train_sequence
from models.losses import pressure_headloss_consistency_loss
from scripts.feature_utils import SequenceDataset
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


def test_sequence_head_loss_uses_elevation_column():
    head_idx = 3
    elev_idx = 4
    has_chlorine = True
    device = torch.device("cpu")
    node_count = 2

    demand = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    pressure = torch.tensor([[40.0, 45.0]], dtype=torch.float32)
    chlorine = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    elevation = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    head = pressure + elevation

    X_single = torch.stack((demand, pressure, chlorine, head, elevation), dim=-1)
    node_outputs = torch.stack((pressure, chlorine), dim=-1)
    edge_outputs = torch.tensor([[30.0, -30.0]], dtype=torch.float32)

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr_phys = torch.tensor(
        [
            [100.0, 0.5, 130.0, 0.0, 0.0],
            [100.0, 0.5, 130.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    X_np = X_single.unsqueeze(0).numpy()
    Y_np = np.array(
        [
            {
                "node_outputs": node_outputs.numpy(),
                "edge_outputs": edge_outputs.numpy(),
            }
        ],
        dtype=object,
    )
    seq_dataset = SequenceDataset(X_np, Y_np, edge_index.numpy(), None)
    loader = DataLoader(seq_dataset, batch_size=1)

    node_pred_template = node_outputs.unsqueeze(0)
    edge_pred_template = edge_outputs.unsqueeze(0).unsqueeze(-1)

    class DummySequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("node_template", node_pred_template.clone())
            self.register_buffer("edge_template", edge_pred_template.clone())
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, X_seq, edge_index, edge_attr, node_type, edge_type):
            batch = X_seq.size(0)
            node_out = self.node_template.expand(batch, -1, -1, -1).clone() + self.dummy
            edge_out = self.edge_template.expand(batch, -1, -1, -1).clone() + self.dummy
            return {"node_outputs": node_out, "edge_outputs": edge_out}

    model = DummySequenceModel().to(device)
    model.x_mean = model.x_std = model.y_mean = model.y_std = None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model.train()

    press_manual = node_pred_template[..., 0]
    flow_manual = edge_pred_template.squeeze(-1)
    elev_manual = X_single.unsqueeze(0)[..., elev_idx]
    expected_head_loss, _ = pressure_headloss_consistency_loss(
        press_manual,
        flow_manual,
        edge_index,
        edge_attr_phys,
        elevation=elev_manual,
        edge_type=None,
        node_type=None,
        return_violation=True,
        sign_weight=0.0,
        use_head=True,
    )
    expected_head_loss = expected_head_loss.item()

    wrong_head_loss, _ = pressure_headloss_consistency_loss(
        press_manual,
        flow_manual,
        edge_index,
        edge_attr_phys,
        elevation=X_single.unsqueeze(0)[..., head_idx],
        edge_type=None,
        node_type=None,
        return_violation=True,
        sign_weight=0.0,
        use_head=True,
    )
    wrong_head_loss = wrong_head_loss.item()
    assert not math.isclose(expected_head_loss, wrong_head_loss, rel_tol=0.0, abs_tol=1e-6)

    train_result = train_sequence(
        model,
        loader,
        edge_index,
        None,
        edge_attr_phys,
        None,
        None,
        [],
        optimizer,
        device,
        physics_loss=False,
        pressure_loss=True,
        pump_loss=False,
        node_mask=None,
        amp=False,
        progress=False,
        head_sign_weight=0.0,
        has_chlorine=has_chlorine,
        use_head=True,
        head_idx=head_idx,
        elev_idx=elev_idx,
    )
    train_head_loss = train_result[5]

    eval_loader = DataLoader(seq_dataset, batch_size=1)
    eval_result = evaluate_sequence(
        model,
        eval_loader,
        edge_index,
        None,
        edge_attr_phys,
        None,
        None,
        [],
        device,
        pump_coeffs=None,
        loss_fn="mae",
        physics_loss=False,
        pressure_loss=True,
        pump_loss=False,
        node_mask=None,
        mass_scale=0.0,
        head_scale=0.0,
        pump_scale=0.0,
        w_mass=2.0,
        w_head=1.0,
        w_pump=1.0,
        w_press=3.0,
        w_cl=1.0,
        w_flow=1.0,
        amp=False,
        progress=False,
        head_sign_weight=0.0,
        has_chlorine=has_chlorine,
        use_head=True,
        head_idx=head_idx,
        elev_idx=elev_idx,
    )
    eval_head_loss = eval_result[5]

    assert math.isclose(train_head_loss, expected_head_loss, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(eval_head_loss, expected_head_loss, rel_tol=0.0, abs_tol=1e-6)
