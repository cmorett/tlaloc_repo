import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, GATConv, LayerNorm, MessagePassing
from typing import Optional, Sequence
import torch.utils.checkpoint
import warnings


class HydroConv(MessagePassing):
    """Mass-conserving convolution supporting heterogeneous components."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ) -> None:
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.lin = nn.ModuleList(
            [nn.Linear(in_channels, out_channels) for _ in range(num_node_types)]
        )

        self.edge_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(edge_dim, 1), nn.Softplus()) for _ in range(num_edge_types)]
        )

        self.edge_type_emb = (
            nn.Embedding(num_edge_types, edge_dim) if num_edge_types > 1 else None
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr cannot be None for HydroConv")

        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_type=edge_type)
        out = torch.zeros((x.size(0), self.out_channels), device=x.device, dtype=aggr.dtype)
        for t, lin in enumerate(self.lin):
            idx = node_type == t
            if torch.any(idx):
                out[idx] = lin(aggr[idx]).to(out.dtype)
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if self.edge_type_emb is not None:
            edge_attr = edge_attr + self.edge_type_emb(edge_type)
        weight = torch.zeros(edge_attr.size(0), 1, device=edge_attr.device)
        for t, mlp in enumerate(self.edge_mlps):
            idx = edge_type == t
            if torch.any(idx):
                weight[idx] = mlp(edge_attr[idx]).to(weight.dtype)
        return weight * (x_j - x_i)


class EnhancedGNNEncoder(nn.Module):
    """Flexible GNN encoder supporting heterogeneous graphs and attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
        residual: bool = False,
        edge_dim: Optional[int] = None,
        use_attention: bool = False,
        attention_after_hydro: bool = True,
        gat_heads: int = 4,
        share_weights: bool = False,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ) -> None:
        super().__init__()

        self.use_attention = use_attention
        self.attention_after_hydro = attention_after_hydro
        self.dropout = dropout
        self.residual = residual
        self.act_fn = getattr(F, activation)
        self.share_weights = share_weights
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        def make_conv(in_c: int, out_c: int):
            if use_attention and not attention_after_hydro:
                warnings.warn(
                    "HydroConv disabled because attention_after_hydro=False.",
                    stacklevel=2,
                )
                return GATConv(in_c, out_c // gat_heads, heads=gat_heads, edge_dim=edge_dim)
            if edge_dim is not None:
                return HydroConv(
                    in_c,
                    out_c,
                    edge_dim,
                    num_node_types=self.num_node_types,
                    num_edge_types=self.num_edge_types,
                )
            if use_attention:
                return GATConv(in_c, out_c // gat_heads, heads=gat_heads, edge_dim=edge_dim)
            return GCNConv(in_c, out_c)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.attentions = nn.ModuleList()

        def make_att_module(conv: nn.Module):
            if use_attention and isinstance(conv, HydroConv):
                return MultiheadAttention(hidden_channels, gat_heads, batch_first=True)
            return nn.Identity()

        if share_weights:
            first = make_conv(in_channels, hidden_channels)
            self.convs.append(first)
            self.attentions.append(make_att_module(first))
            shared = first if in_channels == hidden_channels else make_conv(hidden_channels, hidden_channels)
            shared_att = self.attentions[0] if in_channels == hidden_channels else make_att_module(shared)
            for _ in range(num_layers - 1):
                self.convs.append(shared)
                self.attentions.append(shared_att)
        else:
            for i in range(num_layers):
                in_c = in_channels if i == 0 else hidden_channels
                conv = make_conv(in_c, hidden_channels)
                self.convs.append(conv)
                self.attentions.append(make_att_module(conv))

        for _ in range(num_layers):
            self.norms.append(LayerNorm(hidden_channels))

        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv, attn, norm in zip(self.convs, self.attentions, self.norms):
            identity = x
            if isinstance(conv, HydroConv):
                x = conv(x, edge_index, edge_attr, node_type, edge_type)
            elif conv.__class__.__name__ == "GATConv":
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if not isinstance(attn, nn.Identity):
                x, _ = attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
                x = x.squeeze(0)
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = norm(x)
            if self.residual and identity.shape == x.shape:
                x = x + identity
        return self.fc_out(x)


class RecurrentGNNSurrogate(nn.Module):
    """GNN encoder followed by an LSTM for sequence prediction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        output_dim: int,
        num_layers: int,
        use_attention: bool,
        gat_heads: int,
        dropout: float,
        residual: bool,
        rnn_hidden_dim: int,
        share_weights: bool = False,
        num_node_types: int = 1,
        num_edge_types: int = 1,
        attention_after_hydro: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EnhancedGNNEncoder(
            in_channels,
            hidden_channels,
            hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            edge_dim=edge_dim,
            use_attention=use_attention,
            attention_after_hydro=attention_after_hydro,
            gat_heads=gat_heads,
            share_weights=share_weights,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        )
        self.use_checkpoint = use_checkpoint
        self.rnn = nn.LSTM(hidden_channels, rnn_hidden_dim, batch_first=True)
        self.decoder = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(
        self,
        X_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        edge_attr_rep = edge_attr.repeat(batch_size, 1) if edge_attr is not None else None
        node_type_rep = (
            node_type.repeat(batch_size) if node_type is not None else None
        )
        edge_type_rep = (
            edge_type.repeat(batch_size) if edge_type is not None else None
        )

        node_embeddings = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)

            def encode(x):
                return self.encoder(
                    x,
                    batch_edge_index,
                    edge_attr_rep,
                    node_type_rep,
                    edge_type_rep,
                )

            if self.use_checkpoint and self.training:
                x_t = x_t.requires_grad_()
                gnn_out = torch.utils.checkpoint.checkpoint(encode, x_t, use_reentrant=False)
            else:
                gnn_out = encode(x_t)
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)
            node_embeddings.append(gnn_out)

        emb = torch.stack(node_embeddings, dim=1)
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)
        node_pred = self.decoder(rnn_out)
        node_pred = node_pred.clone()
        min_p = min_c = 0.0
        if getattr(self, "y_mean", None) is not None:
            if isinstance(self.y_mean, dict):
                node_mean = self.y_mean["node_outputs"].to(node_pred.device)
                node_std = self.y_std["node_outputs"].to(node_pred.device)
                p_mean = node_mean[..., 0]
                p_std = node_std[..., 0]
                min_p = -p_mean / p_std
                if node_mean.shape[-1] > 1:
                    c_mean = node_mean[..., 1]
                    c_std = node_std[..., 1]
                    min_c = -c_mean / c_std
                else:
                    min_c = torch.zeros_like(min_p)
            else:
                if self.y_mean.ndim == 2:
                    node_mean = self.y_mean.to(node_pred.device)
                    node_std = self.y_std.to(node_pred.device)
                    p_mean = node_mean[:, 0]
                    p_std = node_std[:, 0]
                    min_p = -p_mean / p_std
                    if node_mean.shape[1] > 1:
                        c_mean = node_mean[:, 1]
                        c_std = node_std[:, 1]
                        min_c = -c_mean / c_std
                    else:
                        min_c = torch.zeros_like(min_p)
                else:
                    p_mean = self.y_mean[0].to(node_pred.device)
                    p_std = self.y_std[0].to(node_pred.device)
                    min_p = -p_mean / p_std
                    if self.y_mean.numel() > 1:
                        c_mean = self.y_mean[1].to(node_pred.device)
                        c_std = self.y_std[1].to(node_pred.device)
                        min_c = -c_mean / c_std
                    else:
                        min_c = torch.zeros_like(min_p)

        if node_pred.size(-1) >= 1:
            comps = [torch.clamp(node_pred[..., 0], min=min_p).unsqueeze(-1)]
            if node_pred.size(-1) >= 2:
                comps.append(torch.clamp(node_pred[..., 1], min=min_c).unsqueeze(-1))
                if node_pred.size(-1) > 2:
                    comps.append(node_pred[..., 2:])
            node_pred = torch.cat(comps, dim=-1)
        return node_pred


class MultiTaskGNNSurrogate(nn.Module):
    """Recurrent GNN surrogate predicting node and edge level targets."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        node_output_dim: int,
        edge_output_dim: int,
        num_layers: int,
        use_attention: bool,
        gat_heads: int,
        dropout: float,
        residual: bool,
        rnn_hidden_dim: int,
        share_weights: bool = False,
        num_node_types: int = 1,
        num_edge_types: int = 1,
        attention_after_hydro: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EnhancedGNNEncoder(
            in_channels,
            hidden_channels,
            hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            edge_dim=edge_dim,
            use_attention=use_attention,
            attention_after_hydro=attention_after_hydro,
            gat_heads=gat_heads,
            share_weights=share_weights,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        )
        self.use_checkpoint = use_checkpoint
        self.rnn = nn.LSTM(hidden_channels, rnn_hidden_dim, batch_first=True)
        self.time_att = MultiheadAttention(rnn_hidden_dim, num_heads=4, batch_first=True)
        self.node_decoder = nn.Linear(rnn_hidden_dim, node_output_dim)
        self.edge_decoder = nn.Linear(rnn_hidden_dim * 3, edge_output_dim)

    def reset_tank_levels(
        self,
        init_levels: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        if not hasattr(self, "tank_indices"):
            return
        if device is None:
            device = next(self.parameters()).device
        if init_levels is None:
            init_levels = torch.zeros(batch_size, len(self.tank_indices), device=device)
        else:
            init_levels = init_levels.to(device)
        self.tank_levels = init_levels

    def forward(
        self,
        X_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ):
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        edge_attr_rep = edge_attr.repeat(batch_size, 1) if edge_attr is not None else None
        node_type_rep = (
            node_type.repeat(batch_size) if node_type is not None else None
        )
        edge_type_rep = (
            edge_type.repeat(batch_size) if edge_type is not None else None
        )

        node_embeddings = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)

            def encode(x):
                return self.encoder(
                    x,
                    batch_edge_index,
                    edge_attr_rep,
                    node_type_rep,
                    edge_type_rep,
                )

            if self.use_checkpoint and self.training:
                x_t = x_t.requires_grad_()
                gnn_out = torch.utils.checkpoint.checkpoint(encode, x_t, use_reentrant=False)
            else:
                gnn_out = encode(x_t)
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)
            node_embeddings.append(gnn_out)

        emb = torch.stack(node_embeddings, dim=1)
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)

        att_in = rnn_out.reshape(batch_size * num_nodes, T, -1)
        att_out, _ = self.time_att(att_in, att_in, att_in)
        att_out = att_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)

        node_pred = self.node_decoder(att_out)
        node_pred = node_pred.clone()

        src = edge_index[0]
        tgt = edge_index[1]
        h_src = att_out[:, :, src, :]
        h_tgt = att_out[:, :, tgt, :]
        h_diff = h_src - h_tgt
        edge_emb = torch.cat([h_src, h_tgt, h_diff], dim=-1)
        edge_pred = self.edge_decoder(edge_emb)

        if hasattr(self, "tank_indices") and len(getattr(self, "tank_indices")) > 0:
            if not hasattr(self, "tank_levels") or self.tank_levels.size(0) != batch_size:
                self.tank_levels = torch.zeros(batch_size, len(self.tank_indices), device=device)
            flows = edge_pred[..., 0]
            updates = torch.zeros(batch_size, T, num_nodes, device=device)
            for t in range(T):
                net = []
                for edges, signs in zip(self.tank_edges, self.tank_signs):
                    if edges.numel() == 0:
                        net.append(torch.zeros(batch_size, device=device))
                    else:
                        net.append((flows[:, t, edges] * signs).sum(dim=1))
                net_flow = torch.stack(net, dim=1)
                delta_vol = net_flow * 3600.0  # flows already in m^3/s
                self.tank_levels += delta_vol
                self.tank_levels = self.tank_levels.clamp(min=0.0)
                delta_h = delta_vol / self.tank_areas
                for i, tank_idx in enumerate(self.tank_indices):
                    updates[:, t, tank_idx] = delta_h[:, i]
            update_tensor = torch.zeros_like(node_pred)
            update_tensor[..., 0] = updates
            node_pred = node_pred + update_tensor

        min_p = min_c = 0.0
        if getattr(self, "y_mean", None) is not None:
            if isinstance(self.y_mean, dict):
                node_mean = self.y_mean["node_outputs"]
                node_std = self.y_std["node_outputs"]
                p_mean = node_mean[..., 0].to(node_pred.device)
                p_std = node_std[..., 0].to(node_pred.device)
                if node_mean.shape[-1] > 1:
                    c_mean = node_mean[..., 1].to(node_pred.device)
                    c_std = node_std[..., 1].to(node_pred.device)
                else:
                    c_mean = torch.tensor(0.0, device=node_pred.device)
                    c_std = torch.tensor(1.0, device=node_pred.device)
            else:
                p_mean = self.y_mean[0].to(node_pred.device)
                p_std = self.y_std[0].to(node_pred.device)
                if self.y_mean.shape[0] > 1:
                    c_mean = self.y_mean[1].to(node_pred.device)
                    c_std = self.y_std[1].to(node_pred.device)
                else:
                    c_mean = torch.tensor(0.0, device=node_pred.device)
                    c_std = torch.tensor(1.0, device=node_pred.device)
            min_p = -p_mean / p_std
            min_c = -c_mean / c_std

        if node_pred.size(-1) >= 1:
            comps = [torch.clamp(node_pred[..., 0], min=min_p).unsqueeze(-1)]
            if node_pred.size(-1) >= 2:
                comps.append(torch.clamp(node_pred[..., 1], min=min_c).unsqueeze(-1))
                if node_pred.size(-1) > 2:
                    comps.append(node_pred[..., 2:])
            node_pred = torch.cat(comps, dim=-1)

        return {
            "node_outputs": node_pred,
            "edge_outputs": edge_pred,
        }


GCNEncoder = EnhancedGNNEncoder

__all__ = [
    "HydroConv",
    "EnhancedGNNEncoder",
    "GCNEncoder",
    "RecurrentGNNSurrogate",
    "MultiTaskGNNSurrogate",
]
