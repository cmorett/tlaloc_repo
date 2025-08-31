import warnings
import torch
from torch.nn import MultiheadAttention
from torch_geometric.nn import GATConv

from models.gnn_surrogate import EnhancedGNNEncoder, HydroConv


def test_attention_keeps_hydroconv():
    model = EnhancedGNNEncoder(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_layers=1,
        edge_dim=2,
        use_attention=True,
        attention_after_hydro=True,
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
