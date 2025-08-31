import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Sequence
import wntr

EPS = 1e-8


def compute_norm_stats(
    data_list, per_node: bool = False, node_mask: Optional[torch.Tensor] = None
):
    """Compute mean and std per feature/target dimension from ``data_list``.

    When ``per_node`` is ``True`` statistics are computed for each node index
    separately, otherwise they are aggregated across all nodes. ``node_mask``
    can be provided to exclude specific nodes (e.g., reservoirs or tanks)
    when computing the statistics.
    """
    mask = None
    if node_mask is not None:
        mask = node_mask.to(dtype=torch.bool, device=data_list[0].x.device)
    if per_node:
        all_x = torch.stack([d.x.float() for d in data_list], dim=0)
        if mask is not None:
            all_x = all_x[:, mask]
        x_mean = all_x.mean(dim=0)
        x_std = all_x.std(dim=0) + EPS
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
            if mask is not None:
                all_y_node = all_y_node[:, mask]
            all_y_edge = torch.stack([d.edge_y.float() for d in data_list], dim=0)
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
            if mask is not None:
                all_y = all_y[:, mask]
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
    node_mask: Optional[torch.Tensor] = None,
):
    del per_node  # unused but kept for API symmetry
    mask = node_mask.cpu() if node_mask is not None else None
    for d in data_list:
        if mask is not None:
            d.x[mask] = (d.x[mask] - x_mean) / x_std
        else:
            d.x = (d.x - x_mean) / x_std
        if isinstance(y_mean, dict):
            if mask is not None:
                d.y[mask] = (d.y[mask] - y_mean["node_outputs"]) / y_std[
                    "node_outputs"
                ]
            else:
                d.y = (d.y - y_mean["node_outputs"]) / y_std["node_outputs"]
            if getattr(d, "edge_y", None) is not None:
                d.edge_y = (d.edge_y - y_mean["edge_outputs"]) / y_std[
                    "edge_outputs"
                ]
        else:
            if mask is not None:
                d.y[mask] = (d.y[mask] - y_mean) / y_std
            else:
                d.y = (d.y - y_mean) / y_std
        if edge_attr_mean is not None and getattr(d, "edge_attr", None) is not None:
            d.edge_attr = (d.edge_attr - edge_attr_mean) / edge_attr_std


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
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_attr = None
        if edge_attr is not None:
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        self.node_type = None
        if node_type is not None:
            self.node_type = torch.tensor(node_type, dtype=torch.long)
        self.edge_type = None
        if edge_type is not None:
            self.edge_type = torch.tensor(edge_type, dtype=torch.long)

        first = Y[0]
        if isinstance(first, dict) or (isinstance(first, np.ndarray) and Y.dtype == object):
            self.multi = True
            self.Y = {
                "node_outputs": torch.stack(
                    [torch.tensor(y["node_outputs"], dtype=torch.float32) for y in Y]
                ),
                "edge_outputs": torch.stack(
                    [torch.tensor(y["edge_outputs"], dtype=torch.float32) for y in Y]
                ),
            }
        else:
            self.multi = False
            self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        if self.multi:
            return self.X[idx], {k: v[idx] for k, v in self.Y.items()}
        return self.X[idx], self.Y[idx]


def compute_sequence_norm_stats(
    X: np.ndarray,
    Y: np.ndarray,
    per_node: bool = False,
    node_mask: Optional[Sequence[bool]] = None,
):
    """Return mean and std for sequence arrays including multi-task targets."""

    mask = None
    if node_mask is not None:
        if isinstance(node_mask, torch.Tensor):
            mask = node_mask.cpu().numpy().astype(bool)
        else:
            mask = np.asarray(node_mask, dtype=bool)

    if per_node:
        flat = X.reshape(-1, X.shape[-2], X.shape[-1])
        if mask is not None:
            flat = flat[:, mask]
        x_mean = torch.tensor(flat.mean(axis=0), dtype=torch.float32)
        x_std = torch.tensor(flat.std(axis=0) + EPS, dtype=torch.float32)
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
        if per_node:
            node_flat = node.reshape(-1, node.shape[-2], node.shape[-1])
            if mask is not None:
                node_flat = node_flat[:, mask]
            node_mean = torch.tensor(node_flat.mean(axis=0), dtype=torch.float32)
            node_std = torch.tensor(node_flat.std(axis=0) + EPS, dtype=torch.float32)
        else:
            if mask is not None:
                node_flat = node[:, :, mask, :].reshape(-1, node.shape[-1])
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
        if per_node:
            y_flat = Y.reshape(-1, Y.shape[-2], Y.shape[-1])
            if mask is not None:
                y_flat = y_flat[:, mask]
            y_mean = torch.tensor(y_flat.mean(axis=0), dtype=torch.float32)
            y_std = torch.tensor(y_flat.std(axis=0) + EPS, dtype=torch.float32)
        else:
            if mask is not None:
                y_flat = Y[:, :, mask, :].reshape(-1, Y.shape[-1])
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
    node_mask: Optional[torch.Tensor] = None,
) -> None:
    del per_node  # parameter kept for API symmetry
    mask = node_mask.cpu() if node_mask is not None else None
    if mask is not None:
        dataset.X[:, :, mask, :] = (dataset.X[:, :, mask, :] - x_mean) / x_std
    else:
        dataset.X = (dataset.X - x_mean) / x_std
    if dataset.multi:
        if mask is not None and "node_outputs" in dataset.Y and "node_outputs" in y_mean:
            dataset.Y["node_outputs"][:, :, mask, :] = (
                dataset.Y["node_outputs"][:, :, mask, :] - y_mean["node_outputs"]
            ) / y_std["node_outputs"]
        elif "node_outputs" in dataset.Y and "node_outputs" in y_mean:
            dataset.Y["node_outputs"] = (
                dataset.Y["node_outputs"] - y_mean["node_outputs"]
            ) / y_std["node_outputs"]
        if "edge_outputs" in dataset.Y and "edge_outputs" in y_mean:
            dataset.Y["edge_outputs"] = (
                dataset.Y["edge_outputs"] - y_mean["edge_outputs"]
            ) / y_std["edge_outputs"]
    else:
        if mask is not None:
            dataset.Y[:, :, mask, :] = (dataset.Y[:, :, mask, :] - y_mean) / y_std
        else:
            dataset.Y = (dataset.Y - y_mean) / y_std
    if edge_attr_mean is not None and dataset.edge_attr is not None:
        dataset.edge_attr = (dataset.edge_attr - edge_attr_mean) / edge_attr_std


def build_edge_attr(
    wn: wntr.network.WaterNetworkModel, edge_index: np.ndarray
) -> np.ndarray:
    """Return edge attribute matrix [E,3] for given edge index."""
    node_map = {n: i for i, n in enumerate(wn.node_name_list)}
    attr_dict: Dict[Tuple[int, int], List[float]] = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_map[link.start_node.name]
        j = node_map[link.end_node.name]
        length = getattr(link, "length", 0.0) or 0.0
        diam = getattr(link, "diameter", 0.0) or 0.0
        rough = getattr(link, "roughness", 0.0) or 0.0
        attr = [float(length), float(diam), float(rough)]
        attr_dict[(i, j)] = attr
        attr_dict[(j, i)] = attr
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


def build_static_node_features(
    wn: wntr.network.WaterNetworkModel, num_pumps: int
) -> torch.Tensor:
    """Return per-node static features [demand, 0, elevation]."""
    num_nodes = len(wn.node_name_list)
    feats = torch.zeros(num_nodes, 3 + num_pumps, dtype=torch.float32)
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
        feats[idx, 2] = float(elev or 0.0)
    return feats


def prepare_node_features(
    template: torch.Tensor,
    pressures: torch.Tensor,
    pump_speeds: torch.Tensor,
    model: torch.nn.Module,
    demands: Optional[torch.Tensor] = None,
    skip_normalization: bool = False,
) -> torch.Tensor:
    """Assemble node features using precomputed static attributes."""
    num_nodes = template.size(0)
    num_pumps = pump_speeds.size(-1)
    pump_speeds = pump_speeds.to(dtype=torch.float32, device=template.device)

    if pressures.dim() == 2:
        batch_size = pressures.size(0)
        feats = template.expand(batch_size, num_nodes, template.size(1)).clone()
        if demands is not None:
            feats[:, :, 0] = demands
        feats[:, :, 1] = pressures
        feats[:, :, 3 : 3 + num_pumps] = pump_speeds.view(batch_size, 1, -1).expand(
            batch_size, num_nodes, num_pumps
        )
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
    feats[:, 3 : 3 + num_pumps] = pump_speeds.view(1, -1).expand(num_nodes, num_pumps)
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
    type_dict: Dict[Tuple[int, int], int] = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i = node_map[link.start_node.name]
        j = node_map[link.end_node.name]
        if link_name in wn.pipe_name_list:
            t = 0
        elif link_name in wn.pump_name_list:
            t = 1
        elif link_name in wn.valve_name_list:
            t = 2
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
