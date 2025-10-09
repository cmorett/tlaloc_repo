import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv, GATConv, LayerNorm, MessagePassing
from typing import Optional, Sequence
import torch.utils.checkpoint


class HydroConv(MessagePassing):
    """Mass-conserving convolution supporting heterogeneous components."""

    PUMP_EDGE_TYPE = 1

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

        self.edge_mlps = nn.ModuleList()
        for t in range(num_edge_types):
            out_dim = 2 if t == self.PUMP_EDGE_TYPE else 1
            mlp = nn.Linear(edge_dim, out_dim)
            if out_dim > 1:
                with torch.no_grad():
                    mlp.weight[out_dim - 1].zero_()
                    mlp.bias[out_dim - 1].zero_()
            self.edge_mlps.append(mlp)

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
        direction = edge_attr[:, -2:-1].clone()
        pump_speed = edge_attr[:, -1:].clone()
        if self.edge_type_emb is not None:
            edge_attr = edge_attr + self.edge_type_emb(edge_type)
        gain = edge_attr.new_zeros(edge_attr.size(0), 1)
        bias = edge_attr.new_zeros(edge_attr.size(0), 1)
        for t, mlp in enumerate(self.edge_mlps):
            idx = torch.where(edge_type == t)[0]
            if idx.numel() == 0:
                continue
            raw = mlp(edge_attr.index_select(0, idx))
            if t == self.PUMP_EDGE_TYPE and raw.size(1) >= 2:
                dir_local = direction.index_select(0, idx)
                speed_mag = pump_speed.index_select(0, idx)
                speed_scale = torch.where(
                    dir_local > 0, dir_local, torch.ones_like(dir_local)
                )
                speed_local = speed_mag * speed_scale
                gain_t = F.softplus(raw[:, :1]).to(gain.dtype) * speed_local
                bias_t = raw[:, 1:2].to(bias.dtype) * speed_local
                gain.index_copy_(0, idx, gain_t)
                bias.index_copy_(0, idx, bias_t)
            else:
                gain_t = F.softplus(raw).to(gain.dtype)
                gain.index_copy_(0, idx, gain_t)
        sign = direction * 2.0 - 1.0
        return sign * (gain * (x_j - x_i) + bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for idx in range(self.num_edge_types):
            new_weight_key = f"{prefix}edge_mlps.{idx}.weight"
            new_bias_key = f"{prefix}edge_mlps.{idx}.bias"
            for legacy_prefix in ("edge_mlps", "edge_mlp"):
                old_weight_key = f"{prefix}{legacy_prefix}.{idx}.0.weight"
                old_bias_key = f"{prefix}{legacy_prefix}.{idx}.0.bias"
                if old_weight_key in state_dict:
                    value = state_dict.pop(old_weight_key)
                    if new_weight_key not in state_dict:
                        state_dict[new_weight_key] = value
                if old_bias_key in state_dict:
                    value = state_dict.pop(old_bias_key)
                    if new_bias_key not in state_dict:
                        state_dict[new_bias_key] = value

        pump_idx = self.PUMP_EDGE_TYPE
        if pump_idx < len(self.edge_mlps):
            pump_weight_key = f"{prefix}edge_mlps.{pump_idx}.weight"
            pump_bias_key = f"{prefix}edge_mlps.{pump_idx}.bias"
            expected_out = self.edge_mlps[pump_idx].out_features
            if pump_weight_key in state_dict:
                weight = state_dict[pump_weight_key]
                if (
                    weight.dim() == 2
                    and weight.size(0) < expected_out
                    and expected_out > weight.size(0)
                ):
                    pad = weight.new_zeros((expected_out - weight.size(0), weight.size(1)))
                    state_dict[pump_weight_key] = torch.cat([weight, pad], dim=0)
            if pump_bias_key in state_dict:
                bias_param = state_dict[pump_bias_key]
                if (
                    bias_param.dim() == 1
                    and bias_param.size(0) < expected_out
                    and expected_out > bias_param.size(0)
                ):
                    pad = bias_param.new_zeros(expected_out - bias_param.size(0))
                    state_dict[pump_bias_key] = torch.cat([bias_param, pad], dim=0)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


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
        gat_heads: int = 4,
        share_weights: bool = False,
        num_node_types: int = 1,
        num_edge_types: int = 1,
    ) -> None:
        super().__init__()

        self.use_attention = use_attention
        self.dropout = dropout
        self.residual = residual
        self.act_fn = getattr(F, activation)
        self.share_weights = share_weights
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        def make_conv(in_c: int, out_c: int):
            if use_attention:
                return GATConv(in_c, out_c // gat_heads, heads=gat_heads, edge_dim=edge_dim)
            if edge_dim is not None:
                return HydroConv(
                    in_c,
                    out_c,
                    edge_dim,
                    num_node_types=self.num_node_types,
                    num_edge_types=self.num_edge_types,
                )
            return GCNConv(in_c, out_c)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if share_weights:
            first = make_conv(in_channels, hidden_channels)
            self.convs.append(first)
            shared = first if in_channels == hidden_channels else make_conv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1):
                self.convs.append(shared)
        else:
            for i in range(num_layers):
                in_c = in_channels if i == 0 else hidden_channels
                self.convs.append(make_conv(in_c, hidden_channels))

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
        for conv, norm in zip(self.convs, self.norms):
            identity = x
            if isinstance(conv, HydroConv):
                x = conv(x, edge_index, edge_attr, node_type, edge_type)
            elif conv.__class__.__name__ == "GATConv":
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
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
        use_checkpoint: bool = False,
        pressure_feature_idx: int = 1,
        use_pressure_skip: bool = True,
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
            gat_heads=gat_heads,
            share_weights=share_weights,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        )
        self.use_checkpoint = use_checkpoint
        self.rnn = nn.LSTM(hidden_channels, rnn_hidden_dim, batch_first=True)
        self.decoder = nn.Linear(rnn_hidden_dim, output_dim)
        self.pressure_feature_idx = int(pressure_feature_idx)
        self.use_pressure_skip = bool(use_pressure_skip)
        self.num_pumps = int(num_pumps)
        self.pump_feature_offset = int(pump_feature_offset)
        if self.num_pumps > 0:
            self.pump_gain = nn.Parameter(torch.zeros(self.num_pumps))
        else:
            self.register_parameter("pump_gain", None)

    def forward(
        self,
        X_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        edge_attr_seq = None
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
            if edge_attr.dim() == 2:
                edge_attr_seq = edge_attr.view(1, 1, E, -1).expand(batch_size, T, E, -1)
            elif edge_attr.dim() == 3:
                if edge_attr.size(0) != T:
                    raise IndexError("edge_attr sequence shorter than input sequence")
                edge_attr_seq = edge_attr.view(1, edge_attr.size(0), E, -1).expand(batch_size, -1, -1, -1)
            elif edge_attr.dim() == 4:
                edge_attr_seq = edge_attr
            else:
                raise ValueError("Unsupported edge_attr shape for sequence model")
            if edge_attr_seq.size(0) == 1 and batch_size > 1:
                edge_attr_seq = edge_attr_seq.expand(batch_size, -1, -1, -1)
        node_type_rep = (
            node_type.repeat(batch_size) if node_type is not None else None
        )
        edge_type_rep = (
            edge_type.repeat(batch_size) if edge_type is not None else None
        )

        emb = None
        press_inputs: List[torch.Tensor] = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)
            edge_attr_t = None
            if edge_attr_seq is not None:
                edge_attr_t = edge_attr_seq[:, t].reshape(batch_size * E, -1)

            if self.use_pressure_skip and self.pressure_feature_idx < in_dim:
                press_inputs.append(
                    X_seq[:, t, :, self.pressure_feature_idx]
                )

            if edge_attr_t is not None:

                def encode(x, attr):
                    return self.encoder(
                        x,
                        batch_edge_index,
                        attr,
                        node_type_rep,
                        edge_type_rep,
                    )

                if self.use_checkpoint and self.training:
                    x_t = x_t.requires_grad_()
                    gnn_out = torch.utils.checkpoint.checkpoint(
                        encode,
                        x_t,
                        edge_attr_t,
                        use_reentrant=False,
                    )
                else:
                    gnn_out = encode(x_t, edge_attr_t)
            else:

                def encode_no_attr(x):
                    return self.encoder(
                        x,
                        batch_edge_index,
                        None,
                        node_type_rep,
                        edge_type_rep,
                    )

                if self.use_checkpoint and self.training:
                    x_t = x_t.requires_grad_()
                    gnn_out = torch.utils.checkpoint.checkpoint(
                        encode_no_attr,
                        x_t,
                        use_reentrant=False,
                    )
                else:
                    gnn_out = encode_no_attr(x_t)
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)

            if emb is None:
                emb = X_seq.new_empty(batch_size, T, num_nodes, gnn_out.size(-1))
            emb[:, t] = gnn_out

        assert emb is not None
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)
        out = self.decoder(rnn_out)
        if self.use_pressure_skip and press_inputs and out.size(-1) >= 1:
            press_stack = torch.stack(press_inputs, dim=1).to(out.device, out.dtype)
            if press_stack.shape != out[..., 0].shape:
                press_stack = press_stack.view_as(out[..., 0])
            out = out.clone()
            out[..., 0] = out[..., 0] + press_stack
        out = out.clone()
        min_p = min_c = 0.0
        if getattr(self, "y_mean", None) is not None:
            if isinstance(self.y_mean, dict):
                p_mean = self.y_mean["node_outputs"][0].to(out.device)
                p_std = self.y_std["node_outputs"][0].to(out.device)
                if self.y_mean["node_outputs"].numel() > 1:
                    c_mean = self.y_mean["node_outputs"][1].to(out.device)
                    c_std = self.y_std["node_outputs"][1].to(out.device)
                else:
                    c_mean = torch.tensor(0.0, device=out.device)
                    c_std = torch.tensor(1.0, device=out.device)
            else:
                p_mean = self.y_mean[0].to(out.device)
                p_std = self.y_std[0].to(out.device)
                if self.y_mean.numel() > 1:
                    c_mean = self.y_mean[1].to(out.device)
                    c_std = self.y_std[1].to(out.device)
                else:
                    c_mean = torch.tensor(0.0, device=out.device)
                    c_std = torch.tensor(1.0, device=out.device)
            min_p = -p_mean / p_std
            min_c = -c_mean / c_std

        if out.size(-1) >= 1:
            comps = [torch.clamp(out[..., 0], min=min_p).unsqueeze(-1)]
            if out.size(-1) >= 2:
                comps.append(torch.clamp(out[..., 1], min=min_c).unsqueeze(-1))
                if out.size(-1) > 2:
                    comps.append(out[..., 2:])
            out = torch.cat(comps, dim=-1)
        return out


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
        use_checkpoint: bool = False,
        pressure_feature_idx: int = 1,
        use_pressure_skip: bool = True,
        num_pumps: int = 0,
        pump_feature_offset: int = 4,
        pump_feature_repeats: int = 1,
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
        self.pressure_feature_idx = int(pressure_feature_idx)
        self.use_pressure_skip = bool(use_pressure_skip)
        self.num_pumps = int(num_pumps)
        self.pump_feature_offset = int(pump_feature_offset)
        # Default physics metadata for tank integration; training populates
        # these with the simulator timestep and flow unit conversion.
        self.timestep_seconds = 3600.0
        self.flow_unit_scale = 1.0
        if self.num_pumps > 0:
            repeats = int(pump_feature_repeats) if pump_feature_repeats else 1
            if repeats < 1:
                repeats = 1
            self.pump_feature_repeats = repeats
            self.pump_gain = nn.Parameter(torch.zeros(self.num_pumps))
            if self.pump_feature_repeats >= 2:
                self.pump_head_gain = nn.Parameter(torch.zeros(self.num_pumps))
            else:
                self.register_parameter("pump_head_gain", None)
        else:
            self.pump_feature_repeats = 0
            self.register_parameter("pump_gain", None)
            self.register_parameter("pump_head_gain", None)

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
        edge_attr: Optional[torch.Tensor],
        node_type: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ):
        batch_size, T, num_nodes, in_dim = X_seq.size()
        device = X_seq.device
        E = edge_index.size(1)
        batch_edge_index = edge_index.repeat(1, batch_size) + (
            torch.arange(batch_size, device=device).repeat_interleave(E) * num_nodes
        )
        edge_attr_seq = None
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
            if edge_attr.dim() == 2:
                edge_attr_seq = edge_attr.view(1, 1, E, -1).expand(batch_size, T, E, -1)
            elif edge_attr.dim() == 3:
                if edge_attr.size(0) != T:
                    raise IndexError("edge_attr sequence shorter than input sequence")
                edge_attr_seq = edge_attr.view(1, edge_attr.size(0), E, -1).expand(batch_size, -1, -1, -1)
            elif edge_attr.dim() == 4:
                edge_attr_seq = edge_attr
            else:
                raise ValueError("Unsupported edge_attr shape for sequence model")
            if edge_attr_seq.size(0) == 1 and batch_size > 1:
                edge_attr_seq = edge_attr_seq.expand(batch_size, -1, -1, -1)
        node_type_rep = (
            node_type.repeat(batch_size) if node_type is not None else None
        )
        edge_type_rep = (
            edge_type.repeat(batch_size) if edge_type is not None else None
        )

        emb = None
        press_inputs: List[torch.Tensor] = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(batch_size * num_nodes, in_dim)
            edge_attr_t = None
            if edge_attr_seq is not None:
                edge_attr_t = edge_attr_seq[:, t].reshape(batch_size * E, -1)

            if self.use_pressure_skip and self.pressure_feature_idx < in_dim:
                press_inputs.append(
                    X_seq[:, t, :, self.pressure_feature_idx]
                )

            if edge_attr_t is not None:

                def encode(x, attr):
                    return self.encoder(
                        x,
                        batch_edge_index,
                        attr,
                        node_type_rep,
                        edge_type_rep,
                    )

                if self.use_checkpoint and self.training:
                    x_t = x_t.requires_grad_()
                    gnn_out = torch.utils.checkpoint.checkpoint(
                        encode,
                        x_t,
                        edge_attr_t,
                        use_reentrant=False,
                    )
                else:
                    gnn_out = encode(x_t, edge_attr_t)
            else:

                def encode_no_attr(x):
                    return self.encoder(
                        x,
                        batch_edge_index,
                        None,
                        node_type_rep,
                        edge_type_rep,
                    )

                if self.use_checkpoint and self.training:
                    x_t = x_t.requires_grad_()
                    gnn_out = torch.utils.checkpoint.checkpoint(
                        encode_no_attr,
                        x_t,
                        use_reentrant=False,
                    )
                else:
                    gnn_out = encode_no_attr(x_t)
            gnn_out = gnn_out.view(batch_size, num_nodes, -1)

            if emb is None:
                emb = X_seq.new_empty(batch_size, T, num_nodes, gnn_out.size(-1))
            emb[:, t] = gnn_out

        assert emb is not None
        rnn_in = emb.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, T, -1)
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)

        att_in = rnn_out.reshape(batch_size * num_nodes, T, -1)
        att_out, _ = self.time_att(att_in, att_in, att_in)
        att_out = att_out.reshape(batch_size, num_nodes, T, -1).permute(0, 2, 1, 3)

        node_pred = self.node_decoder(att_out)
        pump_corr = None
        if self.num_pumps > 0 and self.pump_gain is not None:
            pump_start = self.pump_feature_offset
            repeats = self.pump_feature_repeats if self.pump_feature_repeats else 1
            pump_end = pump_start + self.num_pumps * repeats
            pump_feats = X_seq[:, :, :, pump_start:pump_end]
            if self.x_mean is not None and self.x_std is not None:
                x_mean = self.x_mean
                x_std = self.x_std
                if not isinstance(x_mean, torch.Tensor):
                    x_mean = torch.as_tensor(x_mean, device=pump_feats.device, dtype=pump_feats.dtype)
                    x_std = torch.as_tensor(x_std, device=pump_feats.device, dtype=pump_feats.dtype)
                else:
                    x_mean = x_mean.to(device=pump_feats.device, dtype=pump_feats.dtype)
                    x_std = x_std.to(device=pump_feats.device, dtype=pump_feats.dtype)
                if x_mean.ndim == 2:
                    pump_mean = x_mean[:, pump_start:pump_end].unsqueeze(0).unsqueeze(0)
                    pump_std = x_std[:, pump_start:pump_end].unsqueeze(0).unsqueeze(0)
                else:
                    pump_mean = x_mean[..., pump_start:pump_end]
                    pump_std = x_std[..., pump_start:pump_end]
                    while pump_mean.dim() < pump_feats.dim():
                        pump_mean = pump_mean.unsqueeze(0)
                        pump_std = pump_std.unsqueeze(0)
                pump_std = pump_std.clamp_min(1e-6)
                pump_feats = pump_feats * pump_std + pump_mean
            speed_feats = pump_feats[..., : self.num_pumps]
            gain_speed = self.pump_gain.to(device=speed_feats.device, dtype=speed_feats.dtype)
            pump_corr = torch.tensordot(speed_feats, gain_speed, dims=([-1], [0]))
            if self.pump_head_gain is not None and pump_feats.size(-1) >= self.num_pumps * 2:
                head_feats = pump_feats[..., self.num_pumps : self.num_pumps * 2]
                gain_head = self.pump_head_gain.to(device=head_feats.device, dtype=head_feats.dtype)
                pump_corr = pump_corr + torch.tensordot(head_feats, gain_head, dims=([-1], [0]))
            node_pred = node_pred.clone()
            node_pred[..., 0] = node_pred[..., 0] + pump_corr
        else:
            node_pred = node_pred.clone()
        if self.use_pressure_skip and press_inputs and node_pred.size(-1) >= 1:
            press_stack = torch.stack(press_inputs, dim=1).to(node_pred.device, node_pred.dtype)
            if press_stack.shape != node_pred[..., 0].shape:
                press_stack = press_stack.view_as(node_pred[..., 0])
            node_pred[..., 0] = node_pred[..., 0] + press_stack

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
            edge_mean = edge_std = None
            if getattr(self, "y_mean", None) is not None and getattr(self, "y_std", None) is not None:
                if isinstance(self.y_mean, dict):
                    edge_mean = self.y_mean.get("edge_outputs")
                    edge_std = (
                        self.y_std.get("edge_outputs")
                        if isinstance(self.y_std, dict)
                        else None
                    )
                else:
                    edge_mean = self.y_mean
                    edge_std = self.y_std
            if edge_mean is not None and edge_std is not None:
                if not isinstance(edge_mean, torch.Tensor):
                    edge_mean_t = torch.as_tensor(edge_mean, device=device, dtype=flows.dtype)
                else:
                    edge_mean_t = edge_mean.to(device=device, dtype=flows.dtype)
                if not isinstance(edge_std, torch.Tensor):
                    edge_std_t = torch.as_tensor(edge_std, device=device, dtype=flows.dtype)
                else:
                    edge_std_t = edge_std.to(device=device, dtype=flows.dtype)
                if edge_mean_t.ndim > 1 and edge_mean_t.shape[-1] != flows.shape[-1]:
                    edge_mean_t = edge_mean_t.select(-1, 0)
                    edge_std_t = edge_std_t.select(-1, 0)
                mean_flat = edge_mean_t.reshape(-1)
                std_flat = edge_std_t.reshape(-1)
                if mean_flat.numel() == flows.shape[-1]:
                    view_shape = [1] * (flows.dim() - 1) + [flows.shape[-1]]
                    flows = flows * std_flat.view(*view_shape) + mean_flat.view(*view_shape)
                else:
                    flows = flows * std_flat.reshape(-1)[0] + mean_flat.reshape(-1)[0]

            tank_std = None
            if getattr(self, "y_std", None) is not None:
                if isinstance(self.y_std, dict):
                    node_std = self.y_std.get("node_outputs")
                else:
                    node_std = self.y_std
                if node_std is not None:
                    if not isinstance(node_std, torch.Tensor):
                        node_std_t = torch.as_tensor(node_std, device=device, dtype=node_pred.dtype)
                    else:
                        node_std_t = node_std.to(device=device, dtype=node_pred.dtype)
                    if node_std_t.ndim == 0 or node_std_t.numel() == 1:
                        tank_std = node_std_t.reshape(())
                    elif node_std_t.ndim == 1:
                        if node_std_t.shape[0] == 1:
                            tank_std = node_std_t.reshape(())
                        else:
                            tank_std = node_std_t[0]
                    else:
                        press_std = node_std_t.select(-1, 0)
                        if press_std.ndim == 0 or press_std.numel() == 1:
                            tank_std = press_std.reshape(())
                        else:
                            idx = self.tank_indices.to(device=device, dtype=torch.long)
                            if press_std.size(0) >= idx.numel():
                                tank_std = press_std.index_select(0, idx)
                            else:
                                tank_std = press_std.reshape(-1)[0]
            updates = torch.zeros(batch_size, T, num_nodes, device=device)
            for t in range(T):
                net = []
                for edges, signs in zip(self.tank_edges, self.tank_signs):
                    if edges.numel() == 0:
                        net.append(torch.zeros(batch_size, device=device))
                    else:
                        net.append((flows[:, t, edges] * signs).sum(dim=1))
                net_flow = torch.stack(net, dim=1) / 2.0
                timestep = getattr(self, "timestep_seconds", 3600.0)
                flow_scale = getattr(self, "flow_unit_scale", 1.0)
                if isinstance(flow_scale, torch.Tensor):
                    scale_tensor = flow_scale.to(device=device, dtype=net_flow.dtype)
                else:
                    scale_tensor = torch.tensor(
                        flow_scale, device=device, dtype=net_flow.dtype
                    )
                while scale_tensor.dim() < net_flow.dim():
                    scale_tensor = scale_tensor.unsqueeze(0)
                delta_vol = net_flow * scale_tensor * timestep
                self.tank_levels += delta_vol
                self.tank_levels = self.tank_levels.clamp(min=0.0)
                delta_h = delta_vol / self.tank_areas
                delta_update = delta_h
                if tank_std is not None:
                    if not isinstance(tank_std, torch.Tensor):
                        std_local = torch.as_tensor(
                            tank_std, device=device, dtype=delta_h.dtype
                        )
                    else:
                        std_local = tank_std.to(device=device, dtype=delta_h.dtype)
                    if std_local.ndim == 0:
                        std_local = std_local.clamp_min(1e-6)
                        delta_update = delta_h / std_local
                    else:
                        while std_local.dim() < delta_h.dim():
                            std_local = std_local.unsqueeze(0)
                        std_local = std_local.clamp_min(1e-6)
                        delta_update = delta_h / std_local
                else:
                    delta_update = delta_h
                for i, tank_idx in enumerate(self.tank_indices):
                    updates[:, t, tank_idx] = delta_update[:, i]
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


