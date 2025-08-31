import sys
from pathlib import Path
import types
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_gnn import compute_loss_scales
from models.losses import scale_physics_losses
from models.loss_utils import pressure_headloss_consistency_loss


def _make_args(**kwargs):
    return types.SimpleNamespace(**kwargs)


def test_compute_loss_scales_with_edge_flows():
    X_raw = np.array([[[[5.0]]]], dtype=np.float32)
    Y_raw = np.array([
        {"node_outputs": np.zeros((1, 1, 2), dtype=np.float32), "edge_outputs": np.array([[0.03]], dtype=np.float32)}
    ], dtype=object)
    edge_attr_phys_np = np.array([[1000.0, 0.1, 1.0]], dtype=np.float32)
    edge_types = np.array([0])
    pump_coeffs_np = np.zeros((1, 3), dtype=np.float32)
    args = _make_args(
        mass_scale=0.0,
        head_scale=0.0,
        pump_scale=0.0,
        physics_loss=True,
        pressure_loss=True,
        pump_loss=False,
    )
    mass_scale, head_scale, _ = compute_loss_scales(
        X_raw,
        Y_raw,
        edge_attr_phys_np,
        edge_types,
        pump_coeffs_np,
        args,
        seq_mode=True,
    )
    assert args.pressure_loss
    assert mass_scale > 1.0
    assert head_scale > 1.0


def test_compute_loss_scales_without_edge_flows():
    X_raw = np.array([[[[5.0]]]], dtype=np.float32)
    Y_raw = np.array([
        {"node_outputs": np.zeros((1, 1, 2), dtype=np.float32)}
    ], dtype=object)
    edge_attr_phys_np = np.array([[1.0, 0.1, 1.0]], dtype=np.float32)
    edge_types = np.array([0])
    pump_coeffs_np = np.zeros((1, 3), dtype=np.float32)
    args = _make_args(
        mass_scale=0.0,
        head_scale=0.0,
        pump_scale=0.0,
        physics_loss=True,
        pressure_loss=True,
        pump_loss=False,
    )
    with pytest.warns(UserWarning):
        compute_loss_scales(
            X_raw,
            Y_raw,
            edge_attr_phys_np,
            edge_types,
            pump_coeffs_np,
            args,
            seq_mode=True,
        )
    assert not args.pressure_loss


def test_head_loss_scaling_vs_pressure_mae():
    X_raw = np.array([[[[5.0]]]], dtype=np.float32)
    Y_raw = np.array([
        {"node_outputs": np.zeros((1, 1, 2), dtype=np.float32), "edge_outputs": np.array([[0.03]], dtype=np.float32)}
    ], dtype=object)
    edge_attr_phys_np = np.array([[1000.0, 0.1, 1.0]], dtype=np.float32)
    edge_types = np.array([0])
    pump_coeffs_np = np.zeros((1, 3), dtype=np.float32)
    args = _make_args(
        mass_scale=0.0,
        head_scale=0.0,
        pump_scale=0.0,
        physics_loss=True,
        pressure_loss=True,
        pump_loss=False,
    )
    mass_scale, head_scale, _ = compute_loss_scales(
        X_raw,
        Y_raw,
        edge_attr_phys_np,
        edge_types,
        pump_coeffs_np,
        args,
        seq_mode=True,
    )

    pred_pressures = torch.tensor([100.0, 95.0], dtype=torch.float32)
    true_pressures = torch.tensor([100.0, 98.0], dtype=torch.float32)
    flow = torch.tensor([0.03], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    head_loss = pressure_headloss_consistency_loss(
        pred_pressures, flow, edge_index, torch.tensor(edge_attr_phys_np)
    )
    pressure_mae = F.l1_loss(pred_pressures, true_pressures)
    _, scaled_head_loss, _ = scale_physics_losses(
        torch.tensor(0.0), head_loss, torch.tensor(0.0),
        mass_scale=mass_scale, head_scale=head_scale, pump_scale=1.0
    )
    assert scaled_head_loss < pressure_mae
