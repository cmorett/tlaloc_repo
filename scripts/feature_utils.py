import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Sequence, Union
from collections import deque
import wntr
from wntr.network.base import LinkStatus

EPS = 1e-8


def sanitize_edge_attr_stats(
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    skip_cols: Optional[Sequence[int]] = None,
):
    """Return copies of ``mean`` and ``std`` with ``skip_cols`` set to (0, 1).

    Parameters
    ----------
    mean, std:
        Edge attribute statistics. ``None`` values are returned unchanged.
    skip_cols:
        Column indices that should remain in raw units. These indices are
        normalised relative to the last dimension of the tensors.
    """

    if mean is None or std is None or not skip_cols:
        return mean, std

    mean_tensor = torch.as_tensor(mean).clone()
    std_tensor = torch.as_tensor(std).clone()
    if mean_tensor.ndim == 0:
        mean_tensor = mean_tensor.unsqueeze(0)
    if std_tensor.ndim == 0:
        std_tensor = std_tensor.unsqueeze(0)

    num_cols = mean_tensor.shape[-1]
    valid_idx: List[int] = []
    for col in skip_cols:
        idx = int(col)
        if idx < 0:
            idx += num_cols
        if 0 <= idx < num_cols:
            valid_idx.append(idx)
    if not valid_idx:
        return mean_tensor, std_tensor

    idx_tensor = torch.tensor(sorted(set(valid_idx)), dtype=torch.long, device=mean_tensor.device)
    dim = mean_tensor.dim() - 1
    mean_tensor = mean_tensor.clone()
    std_tensor = std_tensor.clone()
    mean_tensor.index_fill_(dim, idx_tensor, 0.0)
    std_tensor.index_fill_(dim, idx_tensor, 1.0)
    return mean_tensor, std_tensor


def compute_norm_stats(
    data_list,
    per_node: bool = False,
    static_cols: Optional[Sequence[int]] = None,
    node_mask: Optional[torch.Tensor] = None,
):
    """Compute mean and std per feature/target dimension from ``data_list``.

    When ``per_node`` is ``True`` statistics are computed for each node index
    separately, otherwise they are aggregated across all nodes.  Columns listed
    in ``static_cols`` will always use global (across-node) statistics even when
    ``per_node`` is ``True``.  This is useful for features that are repeated for
    every node such as pump speeds.
    ``node_mask`` can be used to exclude specific node indices (e.g. tanks or
    reservoirs) from the aggregation.
    """
    mask = None
    if node_mask is not None:
        mask = node_mask.to(dtype=torch.bool)
    if per_node:
        all_x = torch.stack([d.x.float() for d in data_list], dim=0)
        x_mean = all_x.mean(dim=0)
        x_std = all_x.std(dim=0) + EPS
        if static_cols:
            if mask is not None:
                x_flat = all_x[:, mask].reshape(-1, all_x.shape[-1])
            else:
                x_flat = all_x.reshape(-1, all_x.shape[-1])
            global_mean = x_flat.mean(dim=0)
            global_std = x_flat.std(dim=0) + EPS
            for col in static_cols:
                x_mean[:, col] = global_mean[col]
                x_std[:, col] = global_std[col]
    else:
        if mask is not None:
            all_x = torch.cat([d.x.float()[mask] for d in data_list], dim=0)
        else:
            all_x = torch.cat([d.x.float() for d in data_list], dim=0)
        x_mean = all_x.mean(dim=0)
        x_std = all_x.std(dim=0) + EPS

    if any(getattr(d, "edge_y", None) is not None for d in data_list):
        if per_node:
            all_y_node = torch.stack([d.y.float() for d in data_list], dim=0)
        else:
            if mask is not None:
                all_y_node = torch.cat([d.y.float()[mask] for d in data_list], dim=0)
            else:
                all_y_node = torch.cat([d.y.float() for d in data_list], dim=0)
        all_y_edge = torch.cat([d.edge_y.float() for d in data_list], dim=0)
        y_mean = {
            "node_outputs": all_y_node.mean(dim=0),
            "edge_outputs": all_y_edge.mean(dim=0),
        }
        y_std = {
            "node_outputs": all_y_node.std(dim=0) + EPS,
            "edge_outputs": all_y_edge.std(dim=0) + EPS,
        }
    else:
        if per_node:
            all_y = torch.stack([d.y.float() for d in data_list], dim=0)
            y_mean = all_y.mean(dim=0)
            y_std = all_y.std(dim=0) + EPS
        else:
            if mask is not None:
                all_y = torch.cat([d.y.float()[mask] for d in data_list], dim=0)
            else:
                all_y = torch.cat([d.y.float() for d in data_list], dim=0)
            y_mean = all_y.mean(dim=0)
            y_std = all_y.std(dim=0) + EPS
    return x_mean, x_std, y_mean, y_std


def apply_normalization(
    data_list,
    x_mean,
    x_std,
    y_mean,
    y_std,
    edge_attr_mean=None,
    edge_attr_std=None,
    per_node: bool = False,
    skip_edge_attr_cols: Optional[Sequence[int]] = None,
):
    del per_node  # unused but kept for API symmetry
    sanitized_edge_mean = sanitized_edge_std = None
    if edge_attr_mean is not None and edge_attr_std is not None:
        sanitized_edge_mean, sanitized_edge_std = sanitize_edge_attr_stats(
            edge_attr_mean, edge_attr_std, skip_edge_attr_cols
        )
    else:
        sanitized_edge_mean = edge_attr_mean
        sanitized_edge_std = edge_attr_std
    for d in data_list:
        d.x = (d.x - x_mean) / x_std
        if isinstance(y_mean, dict):
            d.y = (d.y - y_mean["node_outputs"]) / y_std["node_outputs"]
            if getattr(d, "edge_y", None) is not None:
                d.edge_y = (d.edge_y - y_mean["edge_outputs"]) / y_std["edge_outputs"]
        else:
            d.y = (d.y - y_mean) / y_std
        if (
            sanitized_edge_mean is not None
            and sanitized_edge_std is not None
            and getattr(d, "edge_attr", None) is not None
        ):
            mean = sanitized_edge_mean
            std = sanitized_edge_std
            if not isinstance(mean, torch.Tensor):
                mean = torch.as_tensor(mean, dtype=d.edge_attr.dtype, device=d.edge_attr.device)
            else:
                mean = mean.to(device=d.edge_attr.device, dtype=d.edge_attr.dtype)
            if not isinstance(std, torch.Tensor):
                std = torch.as_tensor(std, dtype=d.edge_attr.dtype, device=d.edge_attr.device)
            else:
                std = std.to(device=d.edge_attr.device, dtype=d.edge_attr.dtype)
            d.edge_attr = (d.edge_attr - mean) / std


class SequenceDataset(torch.utils.data.Dataset):
    """Simple ``Dataset`` for sequence data supporting multi-task labels."""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: Optional[np.ndarray],
        node_type: Optional[np.ndarray] = None,
        edge_type: Optional[np.ndarray] = None,
        edge_attr_seq: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_attr = None
        if edge_attr is not None:
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        self.edge_attr_seq = None
        if edge_attr_seq is not None:
            self.edge_attr_seq = torch.tensor(edge_attr_seq, dtype=torch.float32)
        self.node_type = None
        if node_type is not None:
            self.node_type = torch.tensor(node_type, dtype=torch.long)
        self.edge_type = None
        if edge_type is not None:
            self.edge_type = torch.tensor(edge_type, dtype=torch.long)

        first = Y[0]
        if isinstance(first, dict) or (isinstance(first, np.ndarray) and Y.dtype == object):
            self.multi = True
            self.Y = {}
            if "node_outputs" in first:
                self.Y["node_outputs"] = torch.stack(
                    [torch.tensor(y["node_outputs"], dtype=torch.float32) for y in Y]
                )
            if "edge_outputs" in first:
                self.Y["edge_outputs"] = torch.stack(
                    [torch.tensor(y["edge_outputs"], dtype=torch.float32) for y in Y]
                )
            if "pump_energy" in first:
                self.Y["pump_energy"] = torch.stack(
                    [torch.tensor(y["pump_energy"], dtype=torch.float32) for y in Y]
                )
            if "demand" in first:
                self.Y["demand"] = torch.stack(
                    [torch.tensor(y["demand"], dtype=torch.float32) for y in Y]
                )
            if self.edge_attr_seq is None and isinstance(first, dict) and "edge_attr_seq" in first:
                self.edge_attr_seq = torch.stack(
                    [torch.tensor(y["edge_attr_seq"], dtype=torch.float32) for y in Y]
                )
        else:
            self.multi = False
            self.Y = torch.tensor(Y, dtype=torch.float32)

        # Truncate to the shortest target length to avoid out-of-bounds access
        self.length = self.X.shape[0]
        if self.multi:
            for v in self.Y.values():
                self.length = min(self.length, v.shape[0])
            self.X = self.X[: self.length]
            for k in list(self.Y.keys()):
                self.Y[k] = self.Y[k][: self.length]
            if self.edge_attr_seq is not None:
                self.edge_attr_seq = self.edge_attr_seq[: self.length]
        else:
            self.length = min(self.length, self.Y.shape[0])
            self.X = self.X[: self.length]
            self.Y = self.Y[: self.length]
            if self.edge_attr_seq is not None:
                self.edge_attr_seq = self.edge_attr_seq[: self.length]

    def __len__(self) -> int:  # type: ignore[override]
        return self.length

    def __getitem__(self, idx: int):  # type: ignore[override]
        if self.multi:
            target = {k: v[idx] for k, v in self.Y.items()}
        else:
            target = self.Y[idx]
        if self.edge_attr_seq is not None:
            return self.X[idx], self.edge_attr_seq[idx], target
        return self.X[idx], target


def compute_sequence_norm_stats(
    X: np.ndarray,
    Y: np.ndarray,
    per_node: bool = False,
    static_cols: Optional[Sequence[int]] = None,
    node_mask: Optional[Sequence[bool]] = None,
):
    """Return mean and std for sequence arrays including multi-task targets.

    ``node_mask`` may be provided to exclude certain node indices from the
    aggregation when computing global statistics.
    """
    mask = None
    if node_mask is not None:
        mask = np.asarray(node_mask, dtype=bool)
    if per_node:
        flat = X.reshape(-1, X.shape[-2], X.shape[-1])
        x_mean = torch.tensor(flat.mean(axis=0), dtype=torch.float32)
        x_std = torch.tensor(flat.std(axis=0) + EPS, dtype=torch.float32)
        if static_cols:
            if mask is not None:
                x_flat = flat[:, mask, :].reshape(-1, X.shape[-1])
            else:
                x_flat = flat.reshape(-1, X.shape[-1])
            global_mean = torch.tensor(x_flat.mean(axis=0), dtype=torch.float32)
            global_std = torch.tensor(x_flat.std(axis=0) + EPS, dtype=torch.float32)
            for col in static_cols:
                x_mean[:, col] = global_mean[col]
                x_std[:, col] = global_std[col]
    else:
        if mask is not None:
            x_flat = X[:, :, mask, :].reshape(-1, X.shape[-1])
        else:
            x_flat = X.reshape(-1, X.shape[-1])
        x_mean = torch.tensor(x_flat.mean(axis=0), dtype=torch.float32)
        x_std = torch.tensor(x_flat.std(axis=0) + EPS, dtype=torch.float32)

    first = Y[0]
    if isinstance(first, dict) or (isinstance(first, np.ndarray) and Y.dtype == object):
        node = np.stack([y["node_outputs"] for y in Y])
        if mask is not None and not per_node:
            node = node[:, :, mask, :]
        if per_node:
            node_flat = node.reshape(-1, node.shape[-2], node.shape[-1])
            node_mean = torch.tensor(node_flat.mean(axis=0), dtype=torch.float32)
            node_std = torch.tensor(node_flat.std(axis=0) + EPS, dtype=torch.float32)
        else:
            node_flat = node.reshape(-1, node.shape[-1])
            node_mean = torch.tensor(node_flat.mean(axis=0), dtype=torch.float32)
            node_std = torch.tensor(node_flat.std(axis=0) + EPS, dtype=torch.float32)

        edge = np.stack([y["edge_outputs"] for y in Y])
        edge_flat = edge.reshape(-1, edge.shape[-1])
        edge_mean = torch.tensor(edge_flat.mean(axis=0), dtype=torch.float32)
        edge_std = torch.tensor(edge_flat.std(axis=0) + EPS, dtype=torch.float32)
        y_mean = {"node_outputs": node_mean, "edge_outputs": edge_mean}
        y_std = {"node_outputs": node_std, "edge_outputs": edge_std}
    else:
        if mask is not None and not per_node:
            Y = Y[:, :, mask, ...]
        if per_node:
            y_flat = Y.reshape(-1, Y.shape[-2], Y.shape[-1])
            y_mean = torch.tensor(y_flat.mean(axis=0), dtype=torch.float32)
            y_std = torch.tensor(y_flat.std(axis=0) + EPS, dtype=torch.float32)
        else:
            y_flat = Y.reshape(-1, Y.shape[-1])
            y_mean = torch.tensor(y_flat.mean(axis=0), dtype=torch.float32)
            y_std = torch.tensor(y_flat.std(axis=0) + EPS, dtype=torch.float32)

    return x_mean, x_std, y_mean, y_std


def apply_sequence_normalization(
    dataset: SequenceDataset,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean,
    y_std,
    edge_attr_mean: Optional[torch.Tensor] = None,
    edge_attr_std: Optional[torch.Tensor] = None,
    per_node: bool = False,
    static_cols: Optional[Sequence[int]] = None,
    skip_edge_attr_cols: Optional[Sequence[int]] = None,
) -> None:
    del per_node, static_cols  # parameters kept for API symmetry
    sanitized_edge_mean = sanitized_edge_std = None
    if edge_attr_mean is not None and edge_attr_std is not None:
        sanitized_edge_mean, sanitized_edge_std = sanitize_edge_attr_stats(
            edge_attr_mean, edge_attr_std, skip_edge_attr_cols
        )
    else:
        sanitized_edge_mean = edge_attr_mean
        sanitized_edge_std = edge_attr_std
    dataset.X = (dataset.X - x_mean) / x_std
    if dataset.multi:
        for k in dataset.Y:
            if k in y_mean:
                dataset.Y[k] = (dataset.Y[k] - y_mean[k]) / y_std[k]
    else:
        dataset.Y = (dataset.Y - y_mean) / y_std
    if (
        sanitized_edge_mean is not None
        and sanitized_edge_std is not None
        and dataset.edge_attr is not None
    ):
        mean = sanitized_edge_mean.to(device=dataset.edge_attr.device, dtype=dataset.edge_attr.dtype)
        std = sanitized_edge_std.to(device=dataset.edge_attr.device, dtype=dataset.edge_attr.dtype)
        dataset.edge_attr = (dataset.edge_attr - mean) / std
    if (
        sanitized_edge_mean is not None
        and sanitized_edge_std is not None
        and dataset.edge_attr_seq is not None
    ):
        mean = sanitized_edge_mean.to(device=dataset.edge_attr_seq.device, dtype=dataset.edge_attr_seq.dtype)
        std = sanitized_edge_std.to(device=dataset.edge_attr_seq.device, dtype=dataset.edge_attr_seq.dtype)
        while mean.dim() < dataset.edge_attr_seq.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        dataset.edge_attr_seq = (dataset.edge_attr_seq - mean) / std


def build_edge_attr(
    wn: wntr.network.WaterNetworkModel,
    edge_index: np.ndarray,
    pump_speeds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Return edge attribute matrix ``[E,13]`` for given edge index."""

    def _safe_float(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return 0.0
        if np.isnan(value_f) or np.isinf(value_f):
            return 0.0
        return value_f

    def _status_flags(status: object) -> Tuple[float, float, float]:
        if status is None:
            return 0.0, 0.0, 0.0
        if isinstance(status, LinkStatus):
            status_enum = status
        elif isinstance(status, str):
            status_upper = status.strip().upper()
            if status_upper == "OPEN":
                status_enum = LinkStatus.Open
            elif status_upper == "CLOSED":
                status_enum = LinkStatus.Closed
            elif status_upper == "ACTIVE":
                status_enum = LinkStatus.Active
            else:
                return 0.0, 0.0, 0.0
        else:
            try:
                status_enum = LinkStatus(status)
            except Exception:
                return 0.0, 0.0, 0.0
        return (
            1.0 if status_enum == LinkStatus.Open else 0.0,
            1.0 if status_enum == LinkStatus.Closed else 0.0,
            1.0 if status_enum == LinkStatus.Active else 0.0,
        )

    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    attr_dict: Dict[Tuple[int, int], List[float]] = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_map[link.start_node.name]
        j = node_map[link.end_node.name]
        length = getattr(link, "length", 0.0) or 0.0
        diam = getattr(link, "diameter", 0.0) or 0.0
        rough = getattr(link, "roughness", 0.0) or 0.0
        is_pump = link_name in wn.pump_name_list
        is_valve = link_name in wn.valve_name_list
        speed = 1.0
        if is_pump and pump_speeds is not None:
            speed = float(pump_speeds.get(link_name, 1.0))
        pump_col = float(speed if is_pump else 0.0)
        valve_setting = 0.0
        valve_open = 0.0
        valve_closed = 0.0
        valve_active = 0.0
        valve_minor_loss = 0.0
        if is_valve:
            valve_setting = _safe_float(getattr(link, "initial_setting", 0.0))
            valve_open, valve_closed, valve_active = _status_flags(
                getattr(link, "status", None)
            )
            valve_minor_loss = _safe_float(getattr(link, "minor_loss", 0.0))
        coeff_a = coeff_b = coeff_c = 0.0
        if is_pump:
            try:
                coeff_a, coeff_b, coeff_c = map(float, link.get_head_curve_coefficients())
            except Exception:
                coeff_a = coeff_b = coeff_c = 0.0
        attr_fwd = [
            float(length),
            float(diam),
            float(rough),
            valve_setting,
            valve_open,
            valve_closed,
            valve_active,
            valve_minor_loss,
            coeff_a,
            coeff_b,
            coeff_c,
            1.0,
            pump_col,
        ]
        attr_rev = attr_fwd.copy()
        if is_pump:
            attr_rev[-2] = 0.0
        attr_dict[(i, j)] = attr_fwd
        attr_dict[(j, i)] = attr_rev
    return np.array([
        attr_dict[(int(s), int(t))] for s, t in edge_index.T
    ], dtype=np.float32)


def build_node_type(wn: wntr.network.WaterNetworkModel) -> np.ndarray:
    """Return integer node type array for the network nodes."""
    types = []
    for n in wn.node_name_list:
        if n in wn.junction_name_list:
            types.append(0)
        elif n in wn.tank_name_list:
            types.append(1)
        elif n in wn.reservoir_name_list:
            types.append(2)
        else:
            types.append(0)
    return np.array(types, dtype=np.int64)


def build_pump_node_matrix(
    wn: wntr.network.WaterNetworkModel,
    dtype: Union[np.dtype, type] = np.float32,
) -> np.ndarray:
    """Return a ``(num_nodes, num_pumps)`` matrix encoding pump incidence.

    The matrix stores ``-1`` at pump suction nodes, ``+1`` at discharge nodes
    and attenuated positive weights ``1 / (d + 1)`` for all junctions that are
    hydraulically downstream of the discharge node (where ``d`` is the number
    of pipe/valve hops away from the pump). This exposes pump speed information
    to remote nodes that would otherwise be multiple graph layers away from the
    pump and allows the surrogate to reason about islanded districts supplied
    through booster pumps.
    """

    node_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    layout = np.zeros((len(node_map), len(wn.pump_name_list)), dtype=np.float32)
    if not wn.pump_name_list:
        return layout.astype(dtype, copy=False)

    graph = wn.to_graph().to_undirected()
    for pump_idx, pump_name in enumerate(wn.pump_name_list):
        pump = wn.get_link(pump_name)
        start = pump.start_node.name if hasattr(pump.start_node, "name") else pump.start_node
        end = pump.end_node.name if hasattr(pump.end_node, "name") else pump.end_node
        if start in node_map:
            layout[node_map[start], pump_idx] -= 1.0
        if end in node_map:
            layout[node_map[end], pump_idx] += 1.0

        if end not in graph:
            continue

        visited = {start}
        queue: deque[Tuple[str, int]] = deque([(end, 0)])
        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if node != end and node in node_map:
                weight = 1.0 / (dist + 1)
                layout[node_map[node], pump_idx] += weight

            neighbors = graph.adj.get(node, {})
            for nbr, edges in neighbors.items():
                if nbr in visited:
                    continue
                include = False
                for link_name, attrs in edges.items():
                    if link_name == pump_name:
                        include = False
                        break
                    link_type = attrs.get("type") if isinstance(attrs, dict) else None
                    if link_type is None:
                        try:
                            link_type = wn.get_link(link_name).link_type
                        except KeyError:
                            link_type = None
                    if link_type and str(link_type).lower() == "pump":
                        include = False
                        break
                    include = True
                if include:
                    queue.append((nbr, dist + 1))
    return layout.astype(dtype, copy=False)


def build_static_node_features(
    wn: wntr.network.WaterNetworkModel,
    num_pumps: int,
    include_chlorine: bool = True,
) -> torch.Tensor:
    """Return per-node static features including pump incidence signs.

    When ``include_chlorine`` is ``True`` the feature layout is
    ``[demand, 0, 0, 0, elevation, pump_1, ..., pump_N]`` corresponding to
    demand, pressure, chlorine, hydraulic head and elevation respectively.
    Without chlorine the layout becomes
    ``[demand, 0, 0, elevation, pump_1, ..., pump_N]``. Pump columns now store
    extended incidence weights: ``-1`` at suction nodes, ``+1`` at discharge
    nodes and decayed positive weights for downstream junctions that convey the
    current pump speed information deeper into the network. Runtime feature
    assembly multiplies these values by the instantaneous pump speeds.
    """

    num_nodes = len(wn.node_name_list)
    base_dim = 5 if include_chlorine else 4
    head_idx = 3 if include_chlorine else 2
    elev_idx = head_idx + 1
    feats = torch.zeros(num_nodes, base_dim + num_pumps, dtype=torch.float32)
    for idx, name in enumerate(wn.node_name_list):
        node = wn.get_node(name)
        if name in wn.junction_name_list:
            demand = node.demand_timeseries_list[0].base_value
        else:
            demand = 0.0
        if name in wn.junction_name_list or name in wn.tank_name_list:
            elev = node.elevation
        elif name in wn.reservoir_name_list:
            elev = node.base_head
        else:
            elev = node.head
        feats[idx, 0] = float(demand)
        feats[idx, elev_idx] = float(elev or 0.0)

    if num_pumps > 0:
        pump_layout = torch.from_numpy(
            build_pump_node_matrix(wn, dtype=np.float32)
        )
        offset = base_dim
        feats[:, offset : offset + num_pumps] = pump_layout
    return feats


def prepare_node_features(
    template: torch.Tensor,
    pressures: torch.Tensor,
    chlorine: Optional[torch.Tensor],
    pump_speeds: torch.Tensor,
    model: torch.nn.Module,
    demands: Optional[torch.Tensor] = None,
    node_type: Optional[torch.Tensor] = None,
    skip_normalization: bool = False,
    include_chlorine: Optional[bool] = None,
) -> torch.Tensor:
    """Assemble node features using precomputed static attributes.

    ``include_chlorine`` defaults to detecting whether the model predicts a
    chlorine target and whether a chlorine tensor was supplied.
    """
    num_nodes = template.size(0)

    num_pumps = pump_speeds.size(-1)

    if include_chlorine is None:
        include_chlorine = chlorine is not None and (
            (getattr(model, "y_mean_node", None) is not None and model.y_mean_node.size(-1) > 1)
            or template.size(1) >= 4 + num_pumps
        )

    base_dim = 5 if include_chlorine else 4
    head_idx = base_dim - 2
    elev_idx = base_dim - 1
    pump_offset = base_dim
    if template.size(1) < pump_offset + num_pumps:
        raise ValueError(
            "Static feature template does not include pump columns consistent with pump speeds"
        )
    pump_layout = template[:, pump_offset : pump_offset + num_pumps]
    pump_layout = pump_layout.to(dtype=torch.float32, device=template.device)
    pump_speeds = pump_speeds.to(dtype=torch.float32, device=template.device)
    pressures = pressures.to(dtype=torch.float32, device=template.device)
    if chlorine is not None:
        chlorine = chlorine.to(dtype=torch.float32, device=template.device)
    if demands is not None:
        demands = demands.to(dtype=torch.float32, device=template.device)
    node_type_tensor = (
        node_type.to(dtype=torch.long, device=template.device)
        if node_type is not None
        else None
    )

    if pressures.dim() == 2:
        batch_size = pressures.size(0)
        feats = template.expand(batch_size, num_nodes, template.size(1)).clone()
        if demands is not None:
            feats[:, :, 0] = demands
        feats[:, :, 1] = pressures
        if include_chlorine:
            feats[:, :, 2] = torch.log1p(chlorine / 1000.0)
        elevations = feats[:, :, elev_idx]
        head = pressures + elevations
        if node_type_tensor is not None:
            reservoir_mask = node_type_tensor == 2
            if reservoir_mask.any():
                head = torch.where(reservoir_mask.unsqueeze(0), pressures, head)
        feats[:, :, head_idx] = head
        if num_pumps:
            speeds = pump_speeds
            if speeds.dim() == 1:
                speeds = speeds.view(1, -1).expand(batch_size, -1)
            elif speeds.dim() == 2:
                if speeds.size(0) == 1 and batch_size > 1:
                    speeds = speeds.expand(batch_size, -1)
                elif speeds.size(0) != batch_size:
                    raise ValueError(
                        f"Pump speed batch dimension mismatch: expected {batch_size}, got {speeds.size(0)}"
                    )
            else:
                raise ValueError("Pump speeds must be 1D or 2D tensor")
            pump_vals = pump_layout.unsqueeze(0) * speeds.unsqueeze(1)
            feats[:, :, pump_offset : pump_offset + num_pumps] = pump_vals
        in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", None)
        if in_dim is not None:
            feats = feats[:, :, :in_dim]
        if not skip_normalization and getattr(model, "x_mean", None) is not None:
            mean = model.x_mean.to(feats.device)
            std = model.x_std.to(feats.device)
            if mean.dim() == 2:
                mean = mean.view(1, num_nodes, -1)
                std = std.view(1, num_nodes, -1)
            elif mean.numel() == num_nodes * template.size(1):
                mean = mean.view(1, num_nodes, -1)
                std = std.view(1, num_nodes, -1)
            else:
                mean = mean.view(1, 1, -1)
                std = std.view(1, 1, -1)
            mean = mean[..., : feats.size(-1)]
            std = std[..., : feats.size(-1)]
            feats = (feats - mean) / (std + EPS)
        return feats.view(batch_size * num_nodes, -1)

    feats = template.clone()
    if demands is not None:
        feats[:, 0] = demands
    feats[:, 1] = pressures
    if include_chlorine:
        feats[:, 2] = torch.log1p(chlorine / 1000.0)
    elevations = feats[:, elev_idx]
    head = pressures + elevations
    if node_type_tensor is not None:
        reservoir_mask = node_type_tensor == 2
        if reservoir_mask.any():
            head = torch.where(reservoir_mask, pressures, head)
    feats[:, head_idx] = head
    if num_pumps:
        speeds = pump_speeds.reshape(-1)
        if speeds.numel() != num_pumps:
            raise ValueError(
                f"Pump speed dimension mismatch: expected {num_pumps}, got {speeds.numel()}"
            )
        feats[:, pump_offset : pump_offset + num_pumps] = pump_layout * speeds
    in_dim = getattr(getattr(model, "layers", [None])[0], "in_channels", None)
    if in_dim is not None:
        feats = feats[:, :in_dim]
    if not skip_normalization and getattr(model, "x_mean", None) is not None:
        mean = model.x_mean.to(feats.device)
        std = model.x_std.to(feats.device)
        if mean.dim() == 2:
            mean = mean[:, : feats.size(-1)]
            std = std[:, : feats.size(-1)]
        elif mean.numel() == num_nodes * template.size(1):
            mean = mean.view(num_nodes, -1)[:, : feats.size(-1)]
            std = std.view(num_nodes, -1)[:, : feats.size(-1)]
        else:
            mean = mean[: feats.size(-1)]
            std = std[: feats.size(-1)]
        feats = (feats - mean) / (std + EPS)
    return feats


def build_pump_coeffs(
    wn: wntr.network.WaterNetworkModel, edge_index: np.ndarray
) -> np.ndarray:
    """Return pump curve coefficients ``[A, B, C]`` for each edge."""
    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    coeff_dict: Dict[Tuple[int, int], List[float]] = {}
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        i = node_map[pump.start_node.name]
        j = node_map[pump.end_node.name]
        a, b, c = pump.get_head_curve_coefficients()
        coeff = [float(a), float(b), float(c)]
        coeff_dict[(i, j)] = coeff
        coeff_dict[(j, i)] = coeff
    zero = [0.0, 0.0, 0.0]
    return np.array(
        [coeff_dict.get((int(s), int(t)), zero) for s, t in edge_index.T],
        dtype=np.float32,
    )


def build_edge_type(
    wn: wntr.network.WaterNetworkModel, edge_index: np.ndarray
) -> np.ndarray:
    """Return integer edge type array matching ``edge_index``."""
    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    pump_type_id = 1
    pump_types = {name: pump_type_id for name in wn.pump_name_list}
    valve_offset = pump_type_id + 1
    valve_types = {name: valve_offset + idx for idx, name in enumerate(wn.valve_name_list)}
    type_dict: Dict[Tuple[int, int], int] = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_map[link.start_node.name]
        j = node_map[link.end_node.name]
        if link_name in pump_types:
            t = pump_types[link_name]
        elif link_name in valve_types:
            t = valve_types[link_name]
        else:
            t = 0
        type_dict[(i, j)] = t
        type_dict[(j, i)] = t
    return np.array([type_dict[(int(s), int(t))] for s, t in edge_index.T], dtype=np.int64)


def build_edge_pairs(
    edge_index: np.ndarray, edge_type: Optional[np.ndarray] = None
) -> List[Tuple[int, int]]:
    """Return list of ``(i, j)`` tuples pairing forward and reverse edges."""
    pair_map: Dict[Tuple[int, int], int] = {}
    pairs: List[Tuple[int, int]] = []
    for eid in range(edge_index.shape[1]):
        u = int(edge_index[0, eid])
        v = int(edge_index[1, eid])
        if (v, u) in pair_map:
            j = pair_map[(v, u)]
            if edge_type is None or (edge_type[eid] == 0 and edge_type[j] == 0):
                pairs.append((j, eid))
        else:
            pair_map[(u, v)] = eid
    return pairs
