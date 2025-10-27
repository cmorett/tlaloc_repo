import random
import pickle
import os
import argparse
import time
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Union, Any, Set
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from contextlib import contextmanager
import math
try:  # Optional progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover - handled gracefully if unavailable
    tqdm = None

try:
    from .reproducibility import configure_seeds, save_config
except ImportError:  # pragma: no cover
    from reproducibility import configure_seeds, save_config

try:
    from .feature_utils import build_edge_attr, build_pump_node_matrix
except ImportError:  # pragma: no cover
    from feature_utils import build_edge_attr, build_pump_node_matrix

logger = logging.getLogger(__name__)

# Minimum allowed pressure [m].  Values below this threshold are clipped
# in both data generation and validation to keep preprocessing consistent.
MIN_PRESSURE = 0.0

# Default pump speed bounds used when the optional manual sampler is enabled.
# By default data generation now relies on EPANET controls which typically
# operate pumps in the ``[0.0, 1.0]`` band.  Manual overrides can expand this
# envelope up to ``DEFAULT_PUMP_SPEED_MAX``.
DEFAULT_PUMP_SPEED_MAX = 1.5
DEFAULT_PUMP_SPEED_MIN = 0.0
DEFAULT_PUMP_STEP = 0.05  # maximum absolute change per hour when sampling

# Resolve repository paths so all files are created inside the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
PLOTS_DIR = REPO_ROOT / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# File extensions produced by EPANET that need to be cleaned up after each
# simulation run.
TEMP_EXTENSIONS = [
    ".inp",
    ".rpt",
    ".bin",
    ".hyd",
    ".msx",
    ".msx-rpt",
    ".msx-bin",
    ".check.msx",
]


@contextmanager
def temp_simulation_files(prefix: Union[Path, str]):
    """Yield ``prefix`` and remove EPANET temporary files afterwards."""
    try:
        yield prefix
    finally:
        for ext in TEMP_EXTENSIONS:
            f = f"{prefix}{ext}"
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
            except PermissionError:
                logger.warning(f"Could not remove file {f}")

import numpy as np
import wntr
from wntr.network.base import LinkStatus
from wntr.metrics.economic import pump_energy
from wntr.network.controls import Comparison
import networkx as nx
import json


def log_array_stats(name: str, arr: np.ndarray) -> None:
    """Log basic statistics for ``arr`` and ensure it has no invalid values.

    ``arr`` is typically a numeric ``numpy.ndarray`` but some callers pass an
    object array containing dictionaries of arrays (e.g. the training targets).
    ``numpy.isnan`` does not support such object dtypes and raises a
    ``TypeError``.  To keep the logging utility broadly applicable we convert
    ``arr`` to ``numpy.ndarray`` and manually inspect object arrays.

    Parameters
    ----------
    name: str
        Human readable name for the array.
    arr: np.ndarray
        Array to inspect.  May contain nested arrays or dictionaries when
        ``dtype`` is ``object``.

    Raises
    ------
    ValueError
        If ``arr`` or any nested array contains ``NaN`` or infinite values.
    """

    def _count_invalid(obj: object) -> Tuple[int, int]:
        """Recursively count NaNs/Infs in ``obj`` regardless of ``dtype``."""
        if isinstance(obj, dict):
            n_tot = i_tot = 0
            for v in obj.values():
                n, i = _count_invalid(v)
                n_tot += n
                i_tot += i
            return n_tot, i_tot
        arr = np.asarray(obj)
        if np.issubdtype(arr.dtype, np.number):
            return (
                int(np.count_nonzero(np.isnan(arr))),
                int(np.count_nonzero(np.isinf(arr))),
            )
        if arr.dtype == object:
            n_tot = i_tot = 0
            for elem in arr.ravel():
                n, i = _count_invalid(elem)
                n_tot += n
                i_tot += i
            return n_tot, i_tot
        return (0, 0)

    arr_np = np.asarray(arr, dtype=object)
    nan_count, inf_count = _count_invalid(arr_np)

    logger.info(
        "%s: shape=%s dtype=%s nan=%d inf=%d",
        name,
        getattr(arr_np, "shape", "n/a"),
        arr_np.dtype,
        nan_count,
        inf_count,
    )
    if nan_count or inf_count:
        logger.error(
            "%s contains invalid values: %d NaNs, %d Infs",
            name,
            nan_count,
            inf_count,
        )
        raise ValueError(f"{name} contains invalid values")


def _get_link_output(container: Optional[Any], key: str) -> Optional[Any]:
    """Safely retrieve a link-level output from the simulation results."""

    if container is None:
        return None
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            value = getter(key)
        except Exception:
            value = None
        else:
            if value is not None:
                return value
    try:
        return container[key]
    except Exception:
        return None


def _extract_series_array(frame: Optional[Any], column: str) -> Optional[np.ndarray]:
    """Return a NumPy array for ``column`` from a pandas-like frame."""

    if frame is None:
        return None
    series = None
    try:
        series = frame[column]
    except Exception:
        getter = getattr(frame, "get", None)
        if callable(getter):
            try:
                series = getter(column)
            except Exception:
                series = None
    if series is None:
        return None
    if hasattr(series, "to_numpy"):
        values = series.to_numpy()
    else:
        values = np.asarray(series)
    return np.asarray(values, dtype=object)


def _coerce_value(value: object) -> float:
    """Convert EPANET status/setting values to floats while handling enums."""

    if value is None:
        return 0.0
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return 0.0
        return float(value)
    if isinstance(value, (np.integer, int)):
        return float(value)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, LinkStatus):
        return 1.0 if value == LinkStatus.Open else 0.0
    maybe_value = getattr(value, "value", None)
    if isinstance(maybe_value, (np.floating, float)):
        if np.isnan(maybe_value):
            return 0.0
        return float(maybe_value)
    if isinstance(maybe_value, (np.integer, int)):
        return float(maybe_value)
    if isinstance(value, str):
        text = value.strip().upper()
        if text in {"OPEN", "OPENED", "ON"}:
            return 1.0
        if text in {"CLOSED", "CLOSE", "OFF"}:
            return 0.0
        try:
            numeric = float(value)
        except ValueError:
            return 0.0
        return 0.0 if math.isnan(numeric) else numeric
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(numeric) else numeric


def _to_numeric_array(values: np.ndarray) -> np.ndarray:
    """Vectorised conversion of status/setting arrays to float."""

    arr = np.asarray(values, dtype=object)
    out = np.empty(arr.shape, dtype=np.float64)
    for idx, val in np.ndenumerate(arr):
        out[idx] = _coerce_value(val)
    return out


def _extract_applied_pump_speeds(
    pump_names: Iterable[str],
    status_df: Optional[Any],
    setting_df: Optional[Any],
    fallback: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[float]]:
    """Return the pump speeds enforced by EPANET at each reported timestep.

    Parameters
    ----------
    pump_names:
        Names of the pumps in the network.
    status_df, setting_df:
        Link-level output frames containing the discrete pump status and
        continuous speed settings exported by EPANET.  ``setting_df`` is
        preferred when available as it reflects variable speed commands,
        otherwise the binary status is converted to ``{0.0, 1.0}``.
    fallback:
        Optional dictionary containing pre-sampled commands.  When EPANET does
        not report any setting/status for a pump this sequence is used instead
        to preserve alignment with the generated features.
    """

    resolved: Dict[str, List[float]] = {}
    fallback = fallback or {}
    for pump_name in pump_names:
        actual: Optional[np.ndarray] = None
        setting_values = _extract_series_array(setting_df, pump_name)
        if setting_values is not None:
            actual = _to_numeric_array(setting_values)
        else:
            status_values = _extract_series_array(status_df, pump_name)
            if status_values is not None:
                actual = _to_numeric_array(status_values)

        fallback_array = np.asarray(fallback.get(pump_name, []), dtype=np.float64)
        if actual is None:
            actual = fallback_array.copy()
        else:
            actual = np.nan_to_num(np.asarray(actual, dtype=np.float64), nan=0.0)
            actual[~np.isfinite(actual)] = 0.0
            if actual.size == 0:
                actual = fallback_array.copy()
            elif fallback_array.size and actual.shape[0] < fallback_array.shape[0]:
                pad = fallback_array[actual.shape[0] :]
                if pad.size:
                    actual = np.concatenate([actual, pad])
        resolved[pump_name] = actual.astype(np.float64).tolist()
    return resolved


def _iter_controls(wn: wntr.network.WaterNetworkModel) -> Iterable[Any]:
    """Yield control objects defined in the network."""

    names_attr = getattr(wn, "control_name_list", None)
    if callable(names_attr):
        try:
            names = list(names_attr())
        except TypeError:
            names = []
    else:
        names = list(names_attr or [])
    seen: set[int] = set()
    for name in names:
        try:
            control = wn.get_control(name)
        except Exception:  # pragma: no cover - defensive
            continue
        if control is None or id(control) in seen:
            continue
        seen.add(id(control))
        yield control

    controls_attr = getattr(wn, "controls", None)
    candidate = None
    if isinstance(controls_attr, dict):
        candidate = controls_attr.values()
    elif controls_attr is not None:
        try:
            candidate = list(controls_attr)
        except TypeError:
            candidate = None
    if candidate:
        for control in candidate:
            if control is None or id(control) in seen:
                continue
            seen.add(id(control))
            yield control


def _jitter_control_thresholds(
    wn: wntr.network.WaterNetworkModel,
    magnitude: float,
) -> Dict[str, Dict[str, List[Any]]]:
    """Randomly perturb simple control thresholds to diversify pump schedules."""

    per_tank_conditions: Dict[str, Dict[str, List[Any]]] = {}
    if magnitude <= 0.0:
        return per_tank_conditions

    for control in _iter_controls(wn):
        condition = getattr(control, "_condition", None)
        if condition is None:
            continue
        threshold = getattr(condition, "_threshold", None)
        if threshold is None:
            continue
        try:
            offset = random.uniform(-magnitude, magnitude)
        except Exception:
            offset = 0.0
        new_threshold = max(0.0, float(threshold) + offset)
        setattr(condition, "_threshold", new_threshold)

        tank_obj = getattr(condition, "_source_obj", None)
        if tank_obj is None or not hasattr(tank_obj, "name"):
            continue
        relation = getattr(condition, "_relation", None)
        bucket = None
        if relation == Comparison.lt:
            bucket = "lt"
        elif relation == Comparison.gt:
            bucket = "gt"
        if bucket is None:
            continue
        tank_conditions = per_tank_conditions.setdefault(
            getattr(tank_obj, "name"), {"lt": [], "gt": []}
        )
        tank_conditions[bucket].append(condition)

    for tank_name, groups in per_tank_conditions.items():
        try:
            tank = wn.get_node(tank_name)
        except Exception:
            tank = None
        min_level = getattr(tank, "min_level", 0.0) if tank is not None else 0.0
        max_level = getattr(tank, "max_level", min_level + 1.0) if tank is not None else min_level + 1.0

        for bucket in ("lt", "gt"):
            for cond in groups[bucket]:
                value = getattr(cond, "_threshold", min_level)
                value = max(min_level, min(max_level, value))
                setattr(cond, "_threshold", value)

        if groups["lt"] and groups["gt"]:
            max_open = max(getattr(cond, "_threshold", min_level) for cond in groups["lt"])
            min_close = min(getattr(cond, "_threshold", max_level) for cond in groups["gt"])
            if max_open >= min_close - 0.1:
                adjustment = max_open - (min_close - 0.1)
                for cond in groups["lt"]:
                    new_thr = max(min_level, getattr(cond, "_threshold", min_level) - adjustment - 1e-3)
                    setattr(cond, "_threshold", new_thr)
            for cond in groups["gt"]:
                new_thr = min(max_level, getattr(cond, "_threshold", max_level))
                setattr(cond, "_threshold", new_thr)
    return per_tank_conditions


def _collect_tank_thresholds(
    wn: wntr.network.WaterNetworkModel,
) -> Dict[str, List[float]]:
    """Return mapping from tank name to control thresholds after jitter."""

    tank_thresholds: Dict[str, List[float]] = {}
    for control in _iter_controls(wn):
        condition = getattr(control, "_condition", None)
        if condition is None:
            continue
        tank_obj = getattr(condition, "_source_obj", None)
        if tank_obj is None or not hasattr(tank_obj, "name"):
            continue
        threshold = getattr(condition, "_threshold", None)
        if threshold is None:
            continue
        tank_thresholds.setdefault(tank_obj.name, []).append(float(threshold))
    return tank_thresholds


def _initialize_tank_levels(
    wn: wntr.network.WaterNetworkModel,
    thresholds: Dict[str, List[float]],
    frac_range: Tuple[float, float],
    *,
    threshold_bias: float = 0.6,
    threshold_band: float = 0.75,
) -> None:
    """Assign initial tank levels biased towards active control thresholds."""

    lower, upper = frac_range
    lower = max(0.0, float(lower))
    upper = max(lower, float(upper))

    for tname in wn.tank_name_list:
        tank = wn.get_node(tname)
        min_level = getattr(tank, "min_level", 0.0)
        max_level = getattr(tank, "max_level", min_level + 1.0)
        span = max(max_level - min_level, 1e-6)
        frac = random.uniform(lower, upper)
        baseline = min_level + frac * span

        target_levels = thresholds.get(tname, [])
        if target_levels and random.random() < max(0.0, min(1.0, threshold_bias)):
            base = random.choice(target_levels)
            base += random.uniform(-threshold_band, threshold_band)
        else:
            base = baseline
        base = max(min_level, min(max_level, base))
        tank.init_level = base


def _pumps_controlled_by_network(
    wn: wntr.network.WaterNetworkModel,
) -> Set[str]:
    """Return names of pumps referenced by explicit controls."""

    controlled: Set[str] = set()
    for control in _iter_controls(wn):
        actions = getattr(control, "_then_actions", [])
        for action in actions:
            target = getattr(action, "_target_obj", None)
            if target is None:
                continue
            name = getattr(target, "name", None)
            link_type = getattr(target, "link_type", None)
            if name and isinstance(link_type, str) and link_type.lower() == "pump":
                controlled.add(name)
    return controlled


PAIRED_PUMP_GROUPS: Tuple[Tuple[str, str], ...] = (("PU8", "PU9"),)


def _apply_paired_pump_patterns(
    pump_controls: Dict[str, List[float]],
    hours: int,
    min_speed: float,
    max_speed: float,
    *,
    groups: Tuple[Tuple[str, str], ...] = PAIRED_PUMP_GROUPS,
) -> None:
    """Assign correlated pump patterns for predefined pump pairs."""

    rng_min = min(min_speed, max_speed)
    rng_max = max(min_speed, max_speed)
    for primary, secondary in groups:
        if primary not in pump_controls or secondary not in pump_controls:
            continue
        seq_primary: List[float] = []
        seq_secondary: List[float] = []
        prev_state = None
        for _ in range(hours):
            state = random.random()
            if state < 0.35:
                sp1 = random.uniform(rng_min, rng_max)
                sp2 = 0.0
                state_label = "primary"
            elif state < 0.7:
                sp1 = 0.0
                sp2 = random.uniform(rng_min, rng_max)
                state_label = "secondary"
            elif state < 0.85:
                base = random.uniform(rng_min, rng_max)
                sp1 = base
                sp2 = base * random.uniform(0.35, 0.75)
                state_label = "both"
            else:
                sp1 = 0.0
                sp2 = 0.0
                state_label = "off"
            # Avoid rapid oscillations by biasing towards previous state
            if prev_state == state_label and state_label in {"primary", "secondary"}:
                sp2 = seq_secondary[-1] if seq_secondary else sp2
                sp1 = seq_primary[-1] if seq_primary else sp1
            seq_primary.append(sp1)
            seq_secondary.append(sp2)
            prev_state = state_label
        pump_controls[primary] = seq_primary
        pump_controls[secondary] = seq_secondary


def _jitter_reservoir_heads(
    wn: wntr.network.WaterNetworkModel,
    magnitude: float,
) -> None:
    """Apply uniform head jitter to reservoir nodes."""

    if magnitude <= 0.0:
        return
    for name in wn.reservoir_name_list:
        reservoir = wn.get_node(name)
        base = getattr(reservoir, "base_head", getattr(reservoir, "head", 0.0))
        delta = random.uniform(-magnitude, magnitude)
        new_head = max(0.0, float(base) + delta)
        if hasattr(reservoir, "base_head"):
            reservoir.base_head = new_head


def _sample_manual_pump_patterns(
    wn: wntr.network.WaterNetworkModel,
    hours: int,
    *,
    pump_speed_min: float,
    pump_speed_max: float,
    pump_step: float,
    on_threshold: float = 0.1,
) -> Dict[str, List[float]]:
    """Generate manual pump speed trajectories via truncated random walks."""

    pump_controls: Dict[str, List[float]] = {pn: [] for pn in wn.pump_name_list}
    if not pump_controls:
        return pump_controls

    max_step = max(pump_step, 1e-6)
    min_dwell = 2
    current_speed: Dict[str, float] = {}
    dwell_time: Dict[str, int] = {}
    is_on: Dict[str, bool] = {}

    for pn in wn.pump_name_list:
        spd = random.uniform(pump_speed_min, pump_speed_max)
        current_speed[pn] = spd
        pump_controls[pn].append(spd)
        is_on[pn] = spd > on_threshold
        dwell_time[pn] = 1

    for _h in range(1, hours):
        for pn in wn.pump_name_list:
            prev = current_speed[pn]
            step = random.gauss(0.0, max_step / 2.0)
            step = float(np.clip(step, -max_step, max_step))
            cand = prev + step
            if is_on[pn] and cand <= on_threshold and dwell_time[pn] < min_dwell:
                cand = max(cand, on_threshold)
            if not is_on[pn] and cand > on_threshold and dwell_time[pn] < min_dwell:
                cand = min(cand, on_threshold)
            cand = float(np.clip(cand, pump_speed_min, pump_speed_max))
            pump_controls[pn].append(cand)
            current_speed[pn] = cand
            if (cand > on_threshold) == is_on[pn]:
                dwell_time[pn] += 1
            else:
                is_on[pn] = cand > on_threshold
                dwell_time[pn] = 1

        if wn.pump_name_list and all(
            pump_controls[pn][-1] <= on_threshold for pn in wn.pump_name_list
        ):
            keep_on = random.choice(wn.pump_name_list)
            forced = random.uniform(
                max(pump_speed_min, on_threshold), pump_speed_max
            )
            pump_controls[keep_on][-1] = forced
            current_speed[keep_on] = forced
            is_on[keep_on] = True
            dwell_time[keep_on] = 1
    return pump_controls


def simulate_extreme_event(
    wn: wntr.network.WaterNetworkModel,
    pump_controls: Dict[str, List[float]],
    idx: int,
    event_type: str,
) -> None:
    """Apply an extreme event to the water network model in-place."""

    if event_type == "fire_flow":
        node_id = random.choice(list(wn.junction_name_list))
        pattern_name = f"{node_id}_pat_{idx}"
        pattern = wn.get_pattern(pattern_name)
        pattern.multipliers = [m * 10.0 for m in pattern.multipliers]
    elif event_type == "pump_failure":
        pump_id = random.choice(list(wn.pump_name_list))
        for h in range(len(pump_controls[pump_id])):
            pump_controls[pump_id][h] = 0.0
        link = wn.get_link(pump_id)
        link.initial_status = LinkStatus.Closed
    elif event_type == "quality_variation":
        for source in wn.source_name_list:
            # Some INP files might register a "source" that isn't actually
            # present in the node registry (WNTR returns the file name with an
            # added index like ``INP1``).  Only modify nodes that truly exist.
            if source in wn.node_name_list:
                n = wn.get_node(source)
                if hasattr(n, "initial_quality"):
                    factor = random.uniform(0.5, 1.5)
                    n.initial_quality *= factor
                    src = wn.get_source(source, None)
                    if src is not None:
                        src.strength *= factor



def _build_randomized_network(
    inp_file: str,
    idx: int,
    *,
    pump_outage: bool = False,
    local_surge: bool = False,
    pipe_closure: bool = False,
    tank_level_range: Tuple[float, float] = (0.0, 1.0),
    stress_test: bool = False,
    manual_pump_sampler: bool = False,
    paired_pump_override: bool = False,
    control_threshold_jitter: float = 0.5,
    reservoir_head_jitter: float = 1.5,
    tank_threshold_bias: float = 0.6,
    tank_threshold_band: float = 0.75,
    pump_speed_min: float = DEFAULT_PUMP_SPEED_MIN,
    pump_speed_max: float = DEFAULT_PUMP_SPEED_MAX,
    pump_step: float = DEFAULT_PUMP_STEP,
    demand_scale_min: float = 0.8,
    demand_scale_max: float = 1.2,
) -> Tuple[wntr.network.WaterNetworkModel, Dict[str, np.ndarray], Dict[str, List[float]]]:
    """Create a network with randomized demand patterns and optional pump overrides.

    Demand multipliers, initial tank levels, reservoir heads and control
    thresholds are perturbed to expose diverse hydraulic responses. Pump speeds
    default to those enforced by EPANET's own controls.  When
    ``manual_pump_sampler`` is enabled the legacy random walk sampler is used to
    drive variable-speed pumps for scenario stress testing.
    """

    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.quality_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.quality.parameter = "CHEMICAL"

    hours = int(wn.options.time.duration // wn.options.time.hydraulic_timestep)

    _jitter_control_thresholds(wn, control_threshold_jitter)
    _jitter_reservoir_heads(wn, reservoir_head_jitter)
    tank_thresholds = _collect_tank_thresholds(wn)
    _initialize_tank_levels(
        wn,
        tank_thresholds,
        tank_level_range,
        threshold_bias=tank_threshold_bias,
        threshold_band=tank_threshold_band,
    )
    controlled_pumps = _pumps_controlled_by_network(wn)
    uncontrolled_pumps = set(wn.pump_name_list) - controlled_pumps

    scale_dict: Dict[str, np.ndarray] = {}
    pattern_dict: Dict[str, wntr.network.elements.Pattern] = {}
    for jname in wn.junction_name_list:
        node = wn.get_node(jname)
        ts = node.demand_timeseries_list[0]
        if ts.pattern is None:
            base_mult = np.ones(hours, dtype=float)
        else:
            base_mult = np.array(ts.pattern.multipliers, dtype=float)
        multipliers = base_mult.copy()
        scale_factors = np.random.uniform(
            demand_scale_min, demand_scale_max, size=len(multipliers)
        )
        multipliers = multipliers * scale_factors
        multipliers = np.clip(multipliers, a_min=0.0, a_max=None)
        pat_name = f"{jname}_pat_{idx}"
        pat = wntr.network.elements.Pattern(pat_name, multipliers)
        wn.add_pattern(pat_name, pat)
        ts.pattern_name = pat_name
        scale_dict[jname] = multipliers
        pattern_dict[jname] = pat

    # Apply localized demand surge on a subnetwork
    if local_surge and wn.junction_name_list:
        G = wn.get_graph()
        start_node = random.choice(list(wn.junction_name_list))
        radius = random.randint(1, 2)
        nodes = [n for n in nx.ego_graph(G, start_node, radius=radius).nodes() if n in scale_dict]
        start_h = random.randint(0, max(hours - 4, 0))
        duration = random.randint(2, 4)
        factor = 1.8 if random.random() < 0.5 else 0.2
        for n in nodes:
            arr = scale_dict[n]
            arr[start_h : start_h + duration] *= factor
            arr = np.clip(arr, 0.0, None)
            scale_dict[n] = arr
            pattern_dict[n].multipliers = arr
    pump_controls: Dict[str, List[float]]
    if manual_pump_sampler:
        rng_min = min(pump_speed_min, pump_speed_max)
        rng_max = max(pump_speed_min, pump_speed_max)
        pump_controls = _sample_manual_pump_patterns(
            wn,
            hours,
            pump_speed_min=rng_min,
            pump_speed_max=rng_max,
            pump_step=max(pump_step, 1e-6),
        )
    else:
        pump_controls = {pn: [0.0] * hours for pn in wn.pump_name_list}
        if uncontrolled_pumps:
            rng_min = min(pump_speed_min, pump_speed_max)
            rng_max = max(pump_speed_min, pump_speed_max)
            free_patterns = _sample_manual_pump_patterns(
                wn,
                hours,
                pump_speed_min=rng_min,
                pump_speed_max=rng_max,
                pump_step=max(pump_step, 1e-6),
            )
            for pn in uncontrolled_pumps:
                pump_controls[pn] = free_patterns.get(pn, pump_controls[pn])

    if paired_pump_override:
        _apply_paired_pump_patterns(
            pump_controls,
            hours,
            pump_speed_min,
            pump_speed_max,
        )

    # Inject pump outage by forcing a pump off for a short period
    if pump_outage and wn.pump_name_list:
        pump_id = random.choice(list(wn.pump_name_list))
        start_h = random.randint(0, max(hours - 4, 0))
        duration = random.randint(2, 4)
        speeds = pump_controls[pump_id]
        for h in range(start_h, min(start_h + duration, hours)):
            speeds[h] = 0.0

    # Inject pipe closure
    if pipe_closure and wn.pipe_name_list:
        pipe_id = random.choice(list(wn.pipe_name_list))
        wn.get_link(pipe_id).initial_status = LinkStatus.Closed

    # Stress tests drastically increase demands and shut off a pump and pipe
    if stress_test:
        for jname in scale_dict:
            arr = scale_dict[jname] * random.uniform(2.0, 3.0)
            arr = np.clip(arr, 0.0, None)
            scale_dict[jname] = arr
            pattern_dict[jname].multipliers = arr
        if wn.pump_name_list:
            pid = random.choice(list(wn.pump_name_list))
            pump_controls[pid] = [0.0] * hours
        if wn.pipe_name_list:
            pid = random.choice(list(wn.pipe_name_list))
            wn.get_link(pid).initial_status = LinkStatus.Closed

    # Ensure pump controls contain finite values and match the simulation horizon
    for pn, seq in pump_controls.items():
        cleaned: List[float] = []
        for spd in seq:
            spd_val = float(spd) if spd is not None else 0.0
            if not math.isfinite(spd_val):
                logger.warning(
                    "Non-finite pump speed encountered for %s; replacing with 0.0",
                    pn,
                )
                spd_val = 0.0
            cleaned.append(spd_val)
        if len(cleaned) < hours:
            pad_value = cleaned[-1] if cleaned else 0.0
            cleaned.extend([pad_value] * (hours - len(cleaned)))
        elif len(cleaned) > hours:
            cleaned = cleaned[:hours]
        pump_controls[pn] = cleaned

    for pn in wn.pump_name_list:
        link = wn.get_link(pn)
        link.initial_status = LinkStatus.Open
        link.base_speed = 1.0
        if manual_pump_sampler or pn in uncontrolled_pumps:
            pat_name = f"{pn}_pat_{idx}"
            wn.add_pattern(
                pat_name, wntr.network.elements.Pattern(pat_name, pump_controls[pn])
            )
            link.speed_pattern_name = pat_name
        else:
            link.speed_pattern_name = None

    return wn, scale_dict, pump_controls


def _run_single_scenario(
    args,
    extreme_rate: float = 0.0,
    pump_outage_rate: float = 0.0,
    local_surge_rate: float = 0.0,
    tank_level_range: Tuple[float, float] = (0.0, 1.0),
    extreme_event_prob: Optional[float] = None,
    manual_pump_sampler: bool = False,
    manual_pump_fraction: float = 0.0,
    paired_pump_fraction: float = 0.0,
    control_threshold_jitter: float = 0.5,
    reservoir_head_jitter: float = 1.5,
    tank_threshold_bias: float = 0.6,
    tank_threshold_band: float = 0.75,
    pump_speed_min: float = DEFAULT_PUMP_SPEED_MIN,
    pump_speed_max: float = DEFAULT_PUMP_SPEED_MAX,
    pump_step: float = DEFAULT_PUMP_STEP,
    demand_scale_min: float = 0.8,
    demand_scale_max: float = 1.2,
) -> Optional[
    Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]
]:
    """Run a single randomized scenario.

    EPANET occasionally fails to write results when the hydraulics become
    infeasible. To make the data generation robust we retry a few times,
    rebuilding the randomized scenario each time.  If all attempts fail
    ``None`` is returned so the caller can skip this scenario.
    """

    if extreme_event_prob is not None:  # backward compatibility
        extreme_rate = extreme_event_prob

    idx, inp_file, seed = args

    status_df = None
    setting_df = None

    for attempt in range(3):
        if seed is not None:
            random.seed(seed + idx + attempt)
            np.random.seed(seed + idx + attempt)

        stress = random.random() < extreme_rate
        pump_out = stress or (random.random() < pump_outage_rate)
        surge = stress or (random.random() < local_surge_rate)
        manual_override = manual_pump_sampler or (random.random() < manual_pump_fraction)
        paired_override = random.random() < paired_pump_fraction

        wn, scale_dict, pump_controls = _build_randomized_network(
            inp_file,
            idx,
            pump_outage=pump_out,
            local_surge=surge,
            pipe_closure=pump_out or stress,
            tank_level_range=tank_level_range,
            stress_test=stress,
            manual_pump_sampler=manual_override,
            paired_pump_override=paired_override,
            control_threshold_jitter=control_threshold_jitter,
            reservoir_head_jitter=reservoir_head_jitter,
            tank_threshold_bias=tank_threshold_bias,
            tank_threshold_band=tank_threshold_band,
            pump_speed_min=pump_speed_min,
            pump_speed_max=pump_speed_max,
            pump_step=pump_step,
            demand_scale_min=demand_scale_min,
            demand_scale_max=demand_scale_max,
        )

        events = []
        hours = int(
            wn.options.time.duration // wn.options.time.hydraulic_timestep
        )
        if surge:
            events.append("local_surge")
        if pump_out:
            events.append("pump_outage")
        if stress:
            events.append("stress")
        if pump_out or stress:
            events.append("pipe_closure")
        if manual_override:
            events.append("manual_sampler")
        scenario_label = "+".join(events) if events else "normal"

        prefix = TEMP_DIR / f"temp_{os.getpid()}_{idx}_{attempt}"
        try:
            with temp_simulation_files(prefix) as pf:
                sim = wntr.sim.EpanetSimulator(wn)
                sim_results = sim.run_sim(file_prefix=str(pf))
                sim_results.scenario_type = scenario_label
                link_outputs = getattr(sim_results, "link", None)
                status_df = _get_link_output(link_outputs, "status")
                setting_df = _get_link_output(link_outputs, "setting")

            pressures_df = sim_results.node["pressure"]
            times = pressures_df.index
            if len(times) <= hours:
                logger.warning(
                    "Scenario %d attempt %d produced only %d timesteps (need %d); retrying",
                    idx,
                    attempt,
                    len(times),
                    hours + 1,
                )
                continue
            min_pressure = float(np.min(pressures_df.values))
            if min_pressure < -20.0:
                logger.warning(
                    "Scenario %d attempt %d reached unrealistic pressure %.2f m; retrying",
                    idx,
                    attempt,
                    min_pressure,
                )
                continue
            break
        except wntr.epanet.exceptions.EpanetException:
            if attempt == 2:
                return None
            else:
                continue

    flows = sim_results.link["flowrate"]
    heads = sim_results.node["head"]
    energy = pump_energy(flows, heads, wn)
    if energy[wn.pump_name_list].isna().any().any():
        raise ValueError("pump energy contains NaN")

    for df in [flows, heads, sim_results.node["pressure"], sim_results.node["quality"]]:
        if df.isna().any().any() or np.isinf(df.values).any():
            raise ValueError("invalid values detected in simulation results")

    pump_controls = _extract_applied_pump_speeds(
        wn.pump_name_list, status_df, setting_df, fallback=pump_controls
    )

    return sim_results, scale_dict, pump_controls


def extract_additional_targets(
    sim_results: wntr.sim.results.SimulationResults,
    wn: wntr.network.WaterNetworkModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of pipe flow rates and pump energy consumption."""

    flows_df = sim_results.link["flowrate"]
    heads_df = sim_results.node["head"]
    num_links = len(wn.link_name_list)
    flows_dir = np.zeros((len(flows_df), num_links * 2), dtype=np.float64)
    for idx, link_name in enumerate(wn.link_name_list):
        col = flows_df[link_name].values.astype(np.float64)
        flows_dir[:, idx * 2] = col
        flows_dir[:, idx * 2 + 1] = -col
    flow_rates = flows_dir
    energy_df = pump_energy(flows_df[wn.pump_name_list], heads_df, wn)
    pump_energy_arr = energy_df[wn.pump_name_list].values.astype(np.float64)
    return flow_rates, pump_energy_arr


def run_scenarios(
    inp_file: str,
    num_scenarios: int,
    seed: Optional[int] = None,
    extreme_rate: float = 0.0,
    pump_outage_rate: float = 0.0,
    local_surge_rate: float = 0.0,
    tank_level_range: Tuple[float, float] = (0.0, 1.0),
    num_workers: Optional[int] = None,
    show_progress: bool = False,
    manual_pump_sampler: bool = False,
    manual_pump_fraction: float = 0.0,
    paired_pump_fraction: float = 0.0,
    control_threshold_jitter: float = 0.5,
    reservoir_head_jitter: float = 1.5,
    tank_threshold_bias: float = 0.6,
    tank_threshold_band: float = 0.75,
    pump_speed_min: float = DEFAULT_PUMP_SPEED_MIN,
    pump_speed_max: float = DEFAULT_PUMP_SPEED_MAX,
    pump_step: float = DEFAULT_PUMP_STEP,
    demand_scale_min: float = 0.8,
    demand_scale_max: float = 1.2,
) -> List[
    Tuple[wntr.sim.results.SimulationResults, Dict[str, np.ndarray], Dict[str, List[float]]]
]:
    """Run multiple randomized scenarios in parallel and return their results.

    Scenarios that remain infeasible after several retries are skipped, so the
    returned list may contain fewer elements than ``num_scenarios``.
    When ``show_progress`` is ``True`` and :mod:`tqdm` is available a progress
    bar indicates completion status.
    """

    args_list = [(i, inp_file, seed) for i in range(num_scenarios)]
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    with Pool(processes=num_workers) as pool:
        func = partial(
            _run_single_scenario,
            extreme_rate=extreme_rate,
            pump_outage_rate=pump_outage_rate,
            local_surge_rate=local_surge_rate,
            tank_level_range=tank_level_range,
            manual_pump_sampler=manual_pump_sampler,
            manual_pump_fraction=manual_pump_fraction,
            paired_pump_fraction=paired_pump_fraction,
            control_threshold_jitter=control_threshold_jitter,
            reservoir_head_jitter=reservoir_head_jitter,
            tank_threshold_bias=tank_threshold_bias,
            tank_threshold_band=tank_threshold_band,
            pump_speed_min=pump_speed_min,
            pump_speed_max=pump_speed_max,
            pump_step=pump_step,
            demand_scale_min=demand_scale_min,
            demand_scale_max=demand_scale_max,
        )
        if show_progress and tqdm is not None:
            raw_results = [
                res for res in tqdm(
                    pool.imap(func, args_list),
                    total=len(args_list),
                    desc="Scenarios",
                )
            ]
        else:
            if show_progress and tqdm is None:
                logger.warning("tqdm is not installed; progress bar disabled.")
            raw_results = pool.map(func, args_list)

    # Filter out failed scenarios returned as ``None``
    results = [res for res in raw_results if res is not None]

    return results


def plot_dataset_distributions(
    demand_mults: Iterable[float],
    pump_speeds: Iterable[float],
    out_prefix: str,
    plots_dir: Path = PLOTS_DIR,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot distributions of demand multipliers and pump speeds."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    d_arr = np.asarray(list(demand_mults), dtype=float)
    p_arr = np.asarray(list(pump_speeds), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(d_arr, bins=30, color="tab:blue", alpha=0.7)
    axes[0].set_xlabel("Demand Multiplier")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Demand Distribution")

    axes[1].hist(p_arr, bins=30, color="tab:orange", alpha=0.7)
    axes[1].set_xlabel("Pump Speed")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Pump Speed Distribution")

    fig.tight_layout()
    fig.savefig(plots_dir / f"dataset_distributions_{out_prefix}.png")
    if not return_fig:
        plt.close(fig)
    return fig if return_fig else None


def plot_pressure_histogram(
    pressures_all: Iterable[float],
    pressures_base: Iterable[float],
    out_prefix: str,
    plots_dir: Path = PLOTS_DIR,
) -> None:
    """Plot histogram of pressures before/after augmentation."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    all_arr = np.asarray(list(pressures_all), dtype=float)
    base_arr = np.asarray(list(pressures_base), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if len(base_arr) > 0:
        ax.hist(base_arr, bins=50, alpha=0.5, label="before")
    ax.hist(all_arr, bins=50, alpha=0.5, label="after")
    ax.set_xlabel("Pressure [m]")
    ax.set_ylabel("Frequency")
    ax.set_title("Pressure Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"pressure_hist_{out_prefix}.png")
    plt.close(fig)


def split_results(
    results: List[
        Tuple[
            wntr.sim.results.SimulationResults,
            Dict[str, np.ndarray],
            Dict[str, List[float]],
        ]
    ],
    seed: Optional[int] = None,
) -> Tuple[List, List, List, Dict[str, int]]:
    num_total = len(results)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_total)
    n_train = int(0.7 * num_total)
    n_val = int(0.15 * num_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_results = [results[i] for i in train_idx]
    val_results = [results[i] for i in val_idx]
    test_results = [results[i] for i in test_idx]
    counts = {
        "train": len(train_results),
        "val": len(val_results),
        "test": len(test_results),
    }
    logger.info(
        "Split %d results into %d train, %d val, %d test scenarios",
        num_total,
        counts["train"],
        counts["val"],
        counts["test"],
    )
    return train_results, val_results, test_results, counts


def build_sequence_dataset(
    results: Iterable[
        Tuple[
            wntr.sim.results.SimulationResults,
            Dict[str, np.ndarray],
            Dict[str, List[float]],
        ]
    ],
    wn_template: wntr.network.WaterNetworkModel,
    seq_len: int,
    *,
    include_chlorine: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a dataset of sequences from simulation results.

    When ``include_chlorine`` is ``False`` the node feature vectors exclude
    chlorine concentrations and the targets only contain next-step pressure.
    """

    X_list: List[np.ndarray] = []
    Y_list: List[dict] = []
    scenario_types: List[str] = []
    edge_attr_seq_list: List[np.ndarray] = []

    pumps = np.array(wn_template.pump_name_list)
    num_pumps = len(pumps)
    pump_index_map = {name: idx for idx, name in enumerate(pumps)}
    node_names = wn_template.node_name_list
    tank_names = np.array(wn_template.tank_name_list)
    num_tanks = len(tank_names)

    node_index_map = {name: idx for idx, name in enumerate(node_names)}
    pump_edge_indices: Dict[str, Tuple[int, int]] = {}
    directed_edges: List[Tuple[int, int]] = []
    for link_name in wn_template.link_name_list:
        link = wn_template.get_link(link_name)
        i = node_index_map[link.start_node.name]
        j = node_index_map[link.end_node.name]
        directed_edges.append((i, j))
        directed_edges.append((j, i))
        if link_name in wn_template.pump_name_list:
            pump_edge_indices[link_name] = (len(directed_edges) - 2, len(directed_edges) - 1)
    edge_index_template = np.array(directed_edges, dtype=np.int64).T
    base_edge_attr_arr = build_edge_attr(wn_template, edge_index_template).astype(np.float64)
    base_edge_attr_arr[:, 2] = np.log1p(base_edge_attr_arr[:, 2])
    for idx_fwd, idx_rev in pump_edge_indices.values():
        base_edge_attr_arr[idx_fwd, -1] = 0.0
        base_edge_attr_arr[idx_rev, -1] = 0.0

    pump_layout = build_pump_node_matrix(wn_template, dtype=np.float64)

    for sim_results, _scale_dict, pump_ctrl in results:
        scenario_types.append(getattr(sim_results, "scenario_type", "normal"))
        # Clamp to avoid unrealistically low pressures while keeping the full
        # range otherwise.  Enforce a 5 m lower bound to match downstream
        # validation logic.
        pressures = sim_results.node["pressure"].clip(lower=MIN_PRESSURE)
        if include_chlorine:
            quality_df = sim_results.node["quality"].clip(lower=0.0, upper=4.0)
            param = str(wn_template.options.quality.parameter).upper()
            if "CHEMICAL" in param or "CHLORINE" in param:
                # convert mg/L to g/L used by CHEMICAL or CHLORINE models before
                # taking the logarithm so the surrogate sees reasonable scales
                quality_df = quality_df / 1000.0
            quality = np.log1p(quality_df)
        else:
            quality = None
        demands = sim_results.node.get("demand")
        if demands is None:
            raise KeyError("Simulation results missing 'demand' node output")
        max_d = float(demands.max().max())
        demands = demands.clip(lower=0.0, upper=max_d * 1.5)
        times = pressures.index
        flows_arr, energy_arr = extract_additional_targets(sim_results, wn_template)

        if len(times) <= seq_len:
            logger.warning(
                "Skipping scenario with only "
                f"{len(times)} timesteps (need {seq_len + 1})"
            )
            continue

        # Precompute arrays and mappings for faster indexing
        node_idx = {n: pressures.columns.get_loc(n) for n in node_names}
        p_arr = pressures.to_numpy(dtype=np.float64)
        q_arr = quality.to_numpy(dtype=np.float64) if quality is not None else None
        d_arr = demands.to_numpy(dtype=np.float64)
        head_df = sim_results.node["head"]
        head_idx = {n: head_df.columns.get_loc(n) for n in head_df.columns}
        h_arr = head_df.to_numpy(dtype=np.float64)
        pump_head_arr = np.zeros((num_pumps, len(times)), dtype=np.float64) if num_pumps else np.zeros((0, len(times)), dtype=np.float64)
        tank_level_arr = np.zeros((num_tanks, len(times)), dtype=np.float64) if num_tanks else np.zeros((0, len(times)), dtype=np.float64)
        if num_pumps:
            pump_ctrl_arr = np.asarray([pump_ctrl[p] for p in pumps], dtype=np.float64)
            for pump_idx, pump_name in enumerate(pumps):
                pump = wn_template.get_link(pump_name)
                start = pump.start_node.name if hasattr(pump.start_node, "name") else pump.start_node
                end = pump.end_node.name if hasattr(pump.end_node, "name") else pump.end_node
                start_col = head_idx[start]
                end_col = head_idx[end]
                pump_head_arr[pump_idx] = h_arr[:, end_col] - h_arr[:, start_col]
        else:
            pump_ctrl_arr = np.zeros((0, len(times)), dtype=np.float64)
        if num_tanks:
            for tank_idx, tank_name in enumerate(tank_names):
                col_idx = node_idx[tank_name]
                tank = wn_template.get_node(tank_name)
                level_series = p_arr[:, col_idx]
                span = max(tank.max_level - tank.min_level, 1e-6)
                level_frac = (level_series - tank.min_level) / span
                tank_level_arr[tank_idx] = np.clip(level_frac, 0.0, 1.0)

        X_seq: List[np.ndarray] = []
        node_out_seq: List[np.ndarray] = []
        edge_out_seq: List[np.ndarray] = []
        energy_seq: List[np.ndarray] = []
        demand_seq: List[np.ndarray] = []
        edge_attr_seq: List[np.ndarray] = []
        reservoir_names = set(wn_template.reservoir_name_list)
        for t in range(seq_len):
            if num_pumps:
                # Align pump features with the command that produced pressures at t+1.
                ctrl_idx = min(t, pump_ctrl_arr.shape[1] - 1)
                pump_vector = pump_ctrl_arr[:, ctrl_idx]
            else:
                pump_vector = pump_ctrl_arr
            pump_vector = np.nan_to_num(pump_vector, nan=0.0, posinf=0.0, neginf=0.0)
            head_step_idx = (
                min(t, pump_head_arr.shape[1] - 1) if pump_head_arr.size else 0
            )
            head_vector = (
                pump_head_arr[:, head_step_idx]
                if pump_head_arr.size
                else pump_head_arr
            )
            if num_tanks:
                tank_step_idx = min(t, tank_level_arr.shape[1] - 1)
                tank_vector = tank_level_arr[:, tank_step_idx]
            else:
                tank_vector = np.empty((0,), dtype=np.float64)
            node_pump = pump_layout * pump_vector
            edge_attr_t = base_edge_attr_arr.copy()
            if pump_edge_indices:
                for pump_name, (idx_fwd, idx_rev) in pump_edge_indices.items():
                    pump_idx = pump_index_map.get(pump_name)
                    if pump_idx is None:
                        continue
                    speed = float(pump_vector[pump_idx]) if num_pumps else 0.0
                    edge_attr_t[idx_fwd, -1] = speed
                    edge_attr_t[idx_rev, -1] = speed
            edge_attr_t = np.nan_to_num(edge_attr_t, nan=0.0, posinf=0.0, neginf=0.0)
            edge_attr_seq.append(edge_attr_t.astype(np.float64))
            feat_nodes = []
            for node in node_names:
                idx = node_idx[node]
                layout_idx = node_index_map[node]
                is_reservoir = node in reservoir_names
                if node in wn_template.reservoir_name_list:
                    # Reservoir nodes report ~0 pressure from EPANET. Use their
                    # fixed hydraulic head instead so the surrogate is aware of
                    # the supply level.
                    p_t = float(wn_template.get_node(node).base_head)
                else:
                    p_t = float(p_arr[t, idx])
                if include_chlorine:
                    c_t = float(q_arr[t, idx])
                else:
                    c_t = None
                if node in wn_template.junction_name_list:
                    d_t = float(d_arr[t, idx])
                else:
                    d_t = 0.0
                if node in wn_template.junction_name_list or node in wn_template.tank_name_list:
                    elev = wn_template.get_node(node).elevation
                elif node in wn_template.reservoir_name_list:
                    elev = wn_template.get_node(node).base_head
                else:
                    n = wn_template.get_node(node)
                    if getattr(n, "elevation", None) is not None:
                        elev = n.elevation
                    else:
                        elev = getattr(n, "base_head", 0.0)
                if elev is None:
                    elev = 0.0
                feat = [d_t, p_t]
                if include_chlorine:
                    feat.append(c_t)
                head_t = p_t if is_reservoir else p_t + elev
                feat.append(head_t)
                feat.append(elev)
                if num_pumps:
                    feat.extend(node_pump[layout_idx].tolist())
                if num_tanks:
                    feat.extend(tank_vector.tolist())
                feat_nodes.append(feat)
            X_seq.append(np.array(feat_nodes, dtype=np.float64))

            out_nodes = []
            for node in node_names:
                idx = node_idx[node]
                if node in wn_template.reservoir_name_list:
                    p_next = float(wn_template.get_node(node).base_head)
                else:
                    p_next = float(p_arr[t + 1, idx])
                if include_chlorine:
                    c_next = float(q_arr[t + 1, idx])
                    out_nodes.append([max(p_next, MIN_PRESSURE), max(c_next, 0.0)])
                else:
                    out_nodes.append([max(p_next, MIN_PRESSURE)])
            node_out_seq.append(np.array(out_nodes, dtype=np.float64))

            edge_out_seq.append(flows_arr[t + 1].astype(np.float64))
            energy_seq.append(energy_arr[t + 1].astype(np.float64))
            demand_next = []
            for node in node_names:
                if node in wn_template.junction_name_list:
                    idx = node_idx[node]
                    # Demand aligned to the next timestep ``t+1``
                    demand_next.append(float(d_arr[t + 1, idx]))
                else:
                    demand_next.append(0.0)
            demand_seq.append(np.array(demand_next, dtype=np.float64))

        X_list.append(np.stack(X_seq))
        scenario_edge_attr = np.stack(edge_attr_seq).astype(np.float32)
        Y_list.append({
            "node_outputs": np.stack(node_out_seq).astype(np.float32),
            "edge_outputs": np.stack(edge_out_seq).astype(np.float32),
            "pump_energy": np.stack(energy_seq).astype(np.float32),
            "demand": np.stack(demand_seq).astype(np.float32),
            "edge_attr_seq": scenario_edge_attr,
        })
        edge_attr_seq_list.append(scenario_edge_attr)

    if not X_list:
        raise ValueError(
            f"No scenarios contained at least {seq_len + 1} timesteps"
        )

    X = np.stack(X_list).astype(np.float32)
    Y = np.array(Y_list, dtype=object)
    edge_attr_seq_arr = np.stack(edge_attr_seq_list).astype(np.float32)
    return X, Y, np.array(scenario_types), edge_attr_seq_arr

def build_dataset(
    results: Iterable[
        Tuple[
            wntr.sim.results.SimulationResults,
            Dict[str, np.ndarray],
            Dict[str, List[float]],
        ]
    ],
    wn_template: wntr.network.WaterNetworkModel,
    *,
    include_chlorine: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    Y_list: List[dict] = []

    pumps = np.array(wn_template.pump_name_list)
    num_pumps = len(pumps)
    node_names = wn_template.node_name_list
    node_index_map = {name: idx for idx, name in enumerate(node_names)}
    pump_layout = build_pump_node_matrix(wn_template, dtype=np.float64)
    tank_names = np.array(wn_template.tank_name_list)
    num_tanks = len(tank_names)

    reservoir_names = set(wn_template.reservoir_name_list)
    for sim_results, _scale_dict, pump_ctrl in results:
        # Enforce a 5 m lower bound while keeping the upper range unrestricted
        # to capture extreme pressure values.
        pressures = sim_results.node["pressure"].clip(lower=MIN_PRESSURE)
        if include_chlorine:
            quality_df = sim_results.node["quality"].clip(lower=0.0, upper=4.0)
            param = str(wn_template.options.quality.parameter).upper()
            if "CHEMICAL" in param or "CHLORINE" in param:
                # CHEMICAL or CHLORINE quality models return mg/L, scale to g/L
                # before applying the log transform
                quality_df = quality_df / 1000.0
            quality = np.log1p(quality_df)
        else:
            quality = None
        demands = sim_results.node.get("demand")
        if demands is None:
            raise KeyError("Simulation results missing 'demand' node output")
        times = pressures.index
        flows_arr, energy_arr = extract_additional_targets(sim_results, wn_template)

        max_d = float(demands.max().max())
        demands = demands.clip(lower=0.0, upper=max_d * 1.5)

        # Precompute arrays and index mappings for faster lookup
        node_idx = {n: pressures.columns.get_loc(n) for n in node_names}
        p_arr = pressures.to_numpy(dtype=np.float64)
        q_arr = quality.to_numpy(dtype=np.float64) if quality is not None else None
        d_arr = demands.to_numpy(dtype=np.float64)
        head_df = sim_results.node["head"]
        head_idx = {n: head_df.columns.get_loc(n) for n in head_df.columns}
        h_arr = head_df.to_numpy(dtype=np.float64)
        pump_head_arr = np.zeros((num_pumps, len(times)), dtype=np.float64) if num_pumps else np.zeros((0, len(times)), dtype=np.float64)
        tank_level_arr = np.zeros((num_tanks, len(times)), dtype=np.float64) if num_tanks else np.zeros((0, len(times)), dtype=np.float64)
        if num_pumps:
            pump_ctrl_arr = np.asarray([pump_ctrl[p] for p in pumps], dtype=np.float64)
            for pump_idx, pump_name in enumerate(pumps):
                pump = wn_template.get_link(pump_name)
                start = pump.start_node.name if hasattr(pump.start_node, "name") else pump.start_node
                end = pump.end_node.name if hasattr(pump.end_node, "name") else pump.end_node
                pump_head_arr[pump_idx] = h_arr[:, head_idx[end]] - h_arr[:, head_idx[start]]
        else:
            pump_ctrl_arr = np.zeros((0, len(times)), dtype=np.float64)
        if num_tanks:
            for tank_idx, tank_name in enumerate(tank_names):
                col_idx = node_idx[tank_name]
                tank = wn_template.get_node(tank_name)
                level_series = p_arr[:, col_idx]
                span = max(tank.max_level - tank.min_level, 1e-6)
                level_frac = (level_series - tank.min_level) / span
                tank_level_arr[tank_idx] = np.clip(level_frac, 0.0, 1.0)

        for i in range(len(times) - 1):
            if num_pumps:
                # Use the command active during [i, i+1) which yields pressures at i+1.
                ctrl_idx = min(i, pump_ctrl_arr.shape[1] - 1)
                pump_vector = pump_ctrl_arr[:, ctrl_idx]
            else:
                pump_vector = pump_ctrl_arr
            pump_vector = np.nan_to_num(pump_vector, nan=0.0, posinf=0.0, neginf=0.0)
            head_step_idx = min(i, pump_head_arr.shape[1] - 1) if pump_head_arr.size else 0
            head_vector = pump_head_arr[:, head_step_idx] if pump_head_arr.size else pump_head_arr
            if num_tanks:
                tank_vector = tank_level_arr[:, min(i, tank_level_arr.shape[1] - 1)]
            else:
                tank_vector = np.empty((0,), dtype=np.float64)
            node_pump = pump_layout * pump_vector

            feat_nodes = []
            for node in node_names:
                idx = node_idx[node]
                layout_idx = node_index_map[node]
                is_reservoir = node in reservoir_names
                if node in wn_template.reservoir_name_list:
                    # Use the reservoir's constant head as the pressure input
                    p_t = float(wn_template.get_node(node).base_head)
                else:
                    p_t = max(p_arr[i, idx], MIN_PRESSURE)
                if include_chlorine:
                    c_t = max(q_arr[i, idx], 0.0)
                if node in wn_template.junction_name_list:
                    d_t = d_arr[i, idx]
                else:
                    d_t = 0.0

                if node in wn_template.junction_name_list or node in wn_template.tank_name_list:
                    elev = wn_template.get_node(node).elevation
                elif node in wn_template.reservoir_name_list:
                    elev = wn_template.get_node(node).base_head
                else:
                    n = wn_template.get_node(node)
                    if getattr(n, "elevation", None) is not None:
                        elev = n.elevation
                    else:
                        elev = getattr(n, "base_head", 0.0)
                if elev is None:
                    elev = 0.0

                feat = [d_t, p_t]
                if include_chlorine:
                    feat.append(c_t)
                head_t = p_t if is_reservoir else p_t + elev
                feat.append(head_t)
                feat.append(elev)
                if num_pumps:
                    feat.extend(node_pump[layout_idx].tolist())
                if num_tanks:
                    feat.extend(tank_vector.tolist())
                feat_nodes.append(feat)
            X_list.append(np.array(feat_nodes, dtype=np.float64))

            out_nodes = []
            demand_next = []
            for node in node_names:
                idx = node_idx[node]
                if node in wn_template.reservoir_name_list:
                    p_next = float(wn_template.get_node(node).base_head)
                else:
                    p_next = max(p_arr[i + 1, idx], MIN_PRESSURE)
                if include_chlorine:
                    c_next = max(q_arr[i + 1, idx], 0.0)
                    out_nodes.append([p_next, c_next])
                else:
                    out_nodes.append([p_next])
                if node in wn_template.junction_name_list:
                    demand_next.append(float(d_arr[i + 1, idx]))
                else:
                    demand_next.append(0.0)
            Y_list.append({
                "node_outputs": np.array(out_nodes, dtype=np.float32),
                "edge_outputs": flows_arr[i + 1].astype(np.float32),
                "pump_energy": energy_arr[i + 1].astype(np.float32),
                "demand": np.array(demand_next, dtype=np.float32),
            })

    if not X_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    X = np.stack(X_list).astype(np.float32)
    Y = np.array(Y_list, dtype=object)
    return X, Y


def build_edge_index(
    wn: wntr.network.WaterNetworkModel,
    pump_speeds: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return directed edge index, attributes, types and pump curves.

    Parameters
    ----------
    wn:
        Water network model used to extract structural information.
    pump_speeds:
        Optional mapping from pump names to their contemporaneous speed. When
        supplied, the returned edge attributes will include this value in the
        final column for both directions of every pump edge.  Missing entries
        default to ``1.0`` which corresponds to the nominal speed used by
        EPANET.  Non-pump edges always store ``0.0`` in the pump-speed column.
    """

    node_index_map = {name: idx for idx, name in enumerate(wn.node_name_list)}
    edges: List[List[int]] = []
    coeffs: List[List[float]] = []
    types: List[int] = []
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        i1 = node_index_map[link.start_node.name]
        i2 = node_index_map[link.end_node.name]
        edges.append([i1, i2])
        edges.append([i2, i1])
        is_pump = link_name in wn.pump_name_list
        if link_name in wn.pipe_name_list:
            t = 0
            a = b = c = 0.0
        elif is_pump:
            t = 1
            a, b, c = link.get_head_curve_coefficients()
        elif link_name in wn.valve_name_list:
            t = 2
            a = b = c = 0.0
        else:
            t = 0
            a = b = c = 0.0
        coeffs.extend([[a, b, c], [a, b, c]])
        types.extend([t, t])

    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = build_edge_attr(wn, edge_index, pump_speeds).astype(np.float32)
    # log-normalise roughness to keep the scale consistent with previous
    # datasets.  Downstream normalisation handles the remaining features.
    edge_attr[:, 2] = np.log1p(edge_attr[:, 2])

    assert edge_index.shape[0] == 2
    edge_type = np.array(types, dtype=np.int64)
    pump_coeffs = np.array(coeffs, dtype=np.float32)
    return edge_index, edge_attr, edge_type, pump_coeffs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=2000,
        help="Number of random scenarios to simulate",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch ops",
    )
    parser.add_argument(
        "--extreme-rate",
        type=float,
        default=0.0,
        help="Fraction of scenarios acting as stress tests with near-zero pressures",
    )
    parser.add_argument(
        "--pump-outage-rate",
        type=float,
        default=0.0,
        help="Probability of randomly shutting off a pump in each scenario",
    )
    parser.add_argument(
        "--local-surge-rate",
        type=float,
        default=0.0,
        help="Probability of applying a localized demand surge",
    )
    parser.add_argument(
        "--manual-pump-sampler",
        action="store_true",
        help=(
            "Enable the legacy pump speed random walk sampler. "
            "When disabled pumps follow EPANET controls and recorded speeds "
            "reflect the hydraulic response."
        ),
    )
    parser.add_argument(
        "--manual-pump-fraction",
        type=float,
        default=0.0,
        help="Fraction of scenarios that should use the manual pump sampler in addition to any explicit flag",
    )
    parser.add_argument(
        "--paired-pump-fraction",
        type=float,
        default=0.5,
        help="Probability of sampling correlated patterns for predefined pump pairs (e.g., PU8/PU9)",
    )
    parser.add_argument(
        "--pump-speed-min",
        type=float,
        default=DEFAULT_PUMP_SPEED_MIN,
        help="Minimum relative pump speed when the manual sampler is enabled",
    )
    parser.add_argument(
        "--pump-speed-max",
        type=float,
        default=DEFAULT_PUMP_SPEED_MAX,
        help="Maximum relative pump speed when the manual sampler is enabled",
    )
    parser.add_argument(
        "--pump-step",
        type=float,
        default=DEFAULT_PUMP_STEP,
        help="Maximum absolute hourly change in pump speed",
    )
    parser.add_argument(
        "--control-threshold-jitter",
        type=float,
        default=0.5,
        help="Uniform perturbation magnitude [m] applied to tank/level controls",
    )
    parser.add_argument(
        "--reservoir-head-jitter",
        type=float,
        default=1.5,
        help="Uniform perturbation magnitude [m] applied to reservoir heads",
    )
    parser.add_argument(
        "--tank-threshold-bias",
        type=float,
        default=0.6,
        help="Probability of initializing tank levels near active control thresholds",
    )
    parser.add_argument(
        "--tank-threshold-band",
        type=float,
        default=0.75,
        help="Jitter band [m] around control thresholds used when sampling initial tank levels",
    )
    parser.add_argument(
        "--demand-scale-min",
        type=float,
        default=0.8,
        help="Minimum multiplicative demand scaling factor",
    )
    parser.add_argument(
        "--demand-scale-max",
        type=float,
        default=1.2,
        help="Maximum multiplicative demand scaling factor",
    )
    parser.add_argument(
        "--exclude-stress",
        action="store_true",
        help="Drop stress-test scenarios from the saved datasets",
    )
    parser.add_argument(
        "--tank-level-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Fractional range for initial tank levels (0=minimum, 1=maximum)",
    )
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        help="Directory to store generated datasets",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers for scenario generation (defaults to CPU count)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1,
        help="Length of sequences to store in the dataset (1 for single-step)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display a progress bar during scenario simulation",
    )
    parser.add_argument(
        "--include-chlorine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include chlorine concentration features and targets",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        default=str(REPO_ROOT / "logs" / "data_generation.log"),
        help="File to write logs to",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_file = Path(args.log_file)
    if not log_file.is_absolute():
        log_file = REPO_ROOT / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(log_level)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    if args.seed is not None:
        configure_seeds(args.seed, args.deterministic)
    save_config(REPO_ROOT / "logs" / "config_data_generation.yaml", vars(args))

    inp_file = REPO_ROOT / "CTown.inp"
    N = args.num_scenarios

    results = run_scenarios(
        str(inp_file),
        N,
        seed=args.seed,
        extreme_rate=args.extreme_rate,
        pump_outage_rate=args.pump_outage_rate,
        local_surge_rate=args.local_surge_rate,
        tank_level_range=tuple(args.tank_level_range),
        num_workers=args.num_workers,
        show_progress=args.show_progress,
        manual_pump_sampler=args.manual_pump_sampler,
        manual_pump_fraction=args.manual_pump_fraction,
        paired_pump_fraction=args.paired_pump_fraction,
        control_threshold_jitter=args.control_threshold_jitter,
        reservoir_head_jitter=args.reservoir_head_jitter,
        tank_threshold_bias=args.tank_threshold_bias,
        tank_threshold_band=args.tank_threshold_band,
        pump_speed_min=args.pump_speed_min,
        pump_speed_max=args.pump_speed_max,
        pump_step=args.pump_step,
        demand_scale_min=args.demand_scale_min,
        demand_scale_max=args.demand_scale_max,
    )

    if args.exclude_stress:
        results = [r for r in results if "stress" not in getattr(r[0], "scenario_type", "")]

    train_res, val_res, test_res, split_counts = split_results(results, seed=args.seed)
    logger.info(
        "Scenario counts - train: %d, val: %d, test: %d",
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    demand_mults: List[float] = []
    pump_speeds: List[float] = []
    all_pressures: List[float] = []
    base_pressures: List[float] = []
    manifest_records: List[Dict[str, float]] = []
    for i, (sim, scale_dict, pump_ctrl) in enumerate(results):
        p_vals = sim.node["pressure"].values.astype(float).ravel()
        all_pressures.extend(p_vals.tolist())
        if getattr(sim, "scenario_type", "normal") == "normal":
            base_pressures.extend(p_vals.tolist())
        manifest_records.append(
            {
                "scenario": i,
                "label": getattr(sim, "scenario_type", "normal"),
                "min_pressure": float(np.min(p_vals)),
                "median_pressure": float(np.median(p_vals)),
                "max_pressure": float(np.max(p_vals)),
            }
        )
        for arr in scale_dict.values():
            demand_mults.extend(arr.ravel().tolist())
        for speeds in pump_ctrl.values():
            pump_speeds.extend(list(speeds))

    plot_dataset_distributions(demand_mults, pump_speeds, run_ts)
    plot_pressure_histogram(all_pressures, base_pressures, run_ts)
    extreme_count = sum(m["min_pressure"] < 10.0 for m in manifest_records)

    def _stats(seq: List[float]) -> Dict[str, float]:
        if len(seq) == 0:
            return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
        arr = np.asarray(seq, dtype=float)
        return {
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
        }

    pressure_stats = _stats(all_pressures)
    demand_stats = _stats(demand_mults)
    pump_stats = _stats(pump_speeds)
    summary = {
        "pressures": pressure_stats,
        "demand_multipliers": demand_stats,
        "pump_speeds": pump_stats,
        "num_extreme": int(extreme_count),
        "total_scenarios": len(manifest_records),
        "split_counts": split_counts,
    }
    logger.info(
        "Pressure min/mean/max: %.2f/%.2f/%.2f; Demand multiplier min/mean/max: %.2f/%.2f/%.2f; "
        "Pump speed min/mean/max: %.2f/%.2f/%.2f; Extreme scenarios: %d/%d",
        pressure_stats["min"],
        pressure_stats["mean"],
        pressure_stats["max"],
        demand_stats["min"],
        demand_stats["mean"],
        demand_stats["max"],
        pump_stats["min"],
        pump_stats["mean"],
        pump_stats["max"],
        summary["num_extreme"],
        summary["total_scenarios"],
    )
    summary_path = REPO_ROOT / "logs" / "data_generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    wn_template = wntr.network.WaterNetworkModel(str(inp_file))
    edge_attr_train_seq = edge_attr_val_seq = edge_attr_test_seq = None
    num_nodes = len(wn_template.node_name_list)
    base_feature_dim = 5 if args.include_chlorine else 4
    base_feature_dim += len(wn_template.pump_name_list)
    base_feature_dim += len(wn_template.tank_name_list)

    def _empty_sequence_arrays():
        return (
            np.empty((0, args.sequence_length, num_nodes, base_feature_dim), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype="<U1"),
            np.empty((0, args.sequence_length, 0, 0), dtype=np.float32),
        )

    if args.sequence_length > 1:
        if train_res:
            (
                X_train,
                Y_train,
                train_labels,
                edge_attr_train_seq,
            ) = build_sequence_dataset(
                train_res,
                wn_template,
                args.sequence_length,
                include_chlorine=args.include_chlorine,
            )
        else:
            X_train, Y_train, train_labels, edge_attr_train_seq = _empty_sequence_arrays()

        if val_res:
            (
                X_val,
                Y_val,
                val_labels,
                edge_attr_val_seq,
            ) = build_sequence_dataset(
                val_res,
                wn_template,
                args.sequence_length,
                include_chlorine=args.include_chlorine,
            )
        else:
            X_val, Y_val, val_labels, edge_attr_val_seq = _empty_sequence_arrays()

        if test_res:
            (
                X_test,
                Y_test,
                test_labels,
                edge_attr_test_seq,
            ) = build_sequence_dataset(
                test_res,
                wn_template,
                args.sequence_length,
                include_chlorine=args.include_chlorine,
            )
        else:
            X_test, Y_test, test_labels, edge_attr_test_seq = _empty_sequence_arrays()
    else:
        X_train, Y_train = build_dataset(
            train_res, wn_template, include_chlorine=args.include_chlorine
        )
        X_val, Y_val = build_dataset(
            val_res, wn_template, include_chlorine=args.include_chlorine
        )
        X_test, Y_test = build_dataset(
            test_res, wn_template, include_chlorine=args.include_chlorine
        )

    edge_index, edge_attr, edge_type, pump_coeffs = build_edge_index(wn_template)

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    log_array_stats("X_train", X_train)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    log_array_stats("Y_train", Y_train)
    np.save(os.path.join(out_dir, "Y_train.npy"), Y_train)
    log_array_stats("X_val", X_val)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    log_array_stats("Y_val", Y_val)
    np.save(os.path.join(out_dir, "Y_val.npy"), Y_val)
    log_array_stats("X_test", X_test)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    log_array_stats("Y_test", Y_test)
    np.save(os.path.join(out_dir, "Y_test.npy"), Y_test)
    if args.sequence_length > 1:
        if edge_attr_train_seq is not None:
            log_array_stats("edge_attr_train_seq", edge_attr_train_seq)
            np.save(os.path.join(out_dir, "edge_attr_train_seq.npy"), edge_attr_train_seq)
        if edge_attr_val_seq is not None:
            log_array_stats("edge_attr_val_seq", edge_attr_val_seq)
            np.save(os.path.join(out_dir, "edge_attr_val_seq.npy"), edge_attr_val_seq)
        if edge_attr_test_seq is not None:
            log_array_stats("edge_attr_test_seq", edge_attr_test_seq)
            np.save(os.path.join(out_dir, "edge_attr_test_seq.npy"), edge_attr_test_seq)
        log_array_stats("scenario_train", train_labels)
        np.save(os.path.join(out_dir, "scenario_train.npy"), train_labels)
        log_array_stats("scenario_val", val_labels)
        np.save(os.path.join(out_dir, "scenario_val.npy"), val_labels)
        log_array_stats("scenario_test", test_labels)
        np.save(os.path.join(out_dir, "scenario_test.npy"), test_labels)
    log_array_stats("edge_index", edge_index)
    np.save(os.path.join(out_dir, "edge_index.npy"), edge_index)
    log_array_stats("edge_attr", edge_attr)
    np.save(os.path.join(out_dir, "edge_attr.npy"), edge_attr)
    log_array_stats("edge_type", edge_type)
    np.save(os.path.join(out_dir, "edge_type.npy"), edge_type)
    log_array_stats("pump_coeffs", pump_coeffs)
    np.save(os.path.join(out_dir, "pump_coeffs.npy"), pump_coeffs)
    # Persist node ordering so training can validate feature alignment
    node_names = np.array(wn_template.node_name_list)
    log_array_stats("node_names", node_names)
    np.save(os.path.join(out_dir, "node_names.npy"), node_names)

    base_layout = ["demand", "pressure"]
    if args.include_chlorine:
        base_layout.append("chlorine")
    base_layout.extend(["head", "elevation"])
    if wn_template.pump_name_list:
        base_layout.extend([f"pump_speed_cmd_{p}" for p in wn_template.pump_name_list])
    if wn_template.tank_name_list:
        base_layout.extend([f"tank_level_{t}" for t in wn_template.tank_name_list])
    manifest = {
        "num_extreme": int(extreme_count),
        "total_scenarios": len(manifest_records),
        "include_chlorine": bool(args.include_chlorine),
        "include_head": True,
        "node_target_dim": 2 if args.include_chlorine else 1,
        "node_feature_layout": base_layout,
        "scenarios": manifest_records,
        "pump_names": list(wn_template.pump_name_list),
        "pump_feature_repeats": 1 if wn_template.pump_name_list else 0,
        "tank_names": list(wn_template.tank_name_list),
        "tank_feature_repeats": 1 if wn_template.tank_name_list else 0,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(out_dir, "train_results_list.pkl"), "wb") as f:
        pickle.dump(train_res, f)
    with open(os.path.join(out_dir, "val_results_list.pkl"), "wb") as f:
        pickle.dump(val_res, f)
    with open(os.path.join(out_dir, "test_results_list.pkl"), "wb") as f:
        pickle.dump(test_res, f)


if __name__ == "__main__":
    main()
