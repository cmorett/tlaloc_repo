"""Compatibility helpers for WNTR simulators.

This module wraps :class:`wntr.epanet.io.BinFile` to prefer double-precision
hydraulics binaries when available. Some EPANET toolkits (notably the 2.2
release distributed with WNTR) emit 64-bit floating point values in the
``.bin`` hydraulics output. Older readers that assume 32-bit values silently
corrupt the results. The :func:`make_simulator` helper defined here creates an
:class:`wntr.sim.EpanetSimulator` that first attempts to parse binaries with
64-bit floats and falls back to the legacy float32 reader when necessary.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import wntr
import numpy as np

_LOGGER = logging.getLogger(__name__)


class DoublePrecisionBinFile(wntr.epanet.io.BinFile):
    """Binary reader that prefers double-precision hydraulics outputs."""

    def __init__(
        self,
        result_types: Optional[Iterable[wntr.epanet.io.ResultType]] = None,
        network: bool = False,
        energy: bool = False,
        statistics: bool = False,
        convert_status: bool = True,
    ) -> None:
        self._result_types_arg = None if result_types is None else list(result_types)
        super().__init__(
            result_types=result_types,
            network=network,
            energy=energy,
            statistics=statistics,
            convert_status=convert_status,
        )
        self.ftype = "=f8"
        self._last_good_read: Optional[bool] = None
        self._fallback_reader: Optional[wntr.epanet.io.BinFile] = None
        self._fallback_exception: Optional[BaseException] = None
        self._fallback_used = False

    def finalize_save(self, good_read, sim_warnings) -> None:  # type: ignore[override]
        if isinstance(good_read, np.ndarray):
            good = bool(good_read.all())
        else:
            good = bool(good_read)
        self._last_good_read = good
        super().finalize_save(good_read, sim_warnings)

    def read(self, filename, convergence_error=False, darcy_weisbach=False, convert=True):  # type: ignore[override]
        self._last_good_read = None
        self._fallback_exception = None
        self._fallback_used = False
        try:
            results = super().read(
                filename,
                convergence_error=convergence_error,
                darcy_weisbach=darcy_weisbach,
                convert=convert,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            self._fallback_exception = exc
            self._last_good_read = False
            _LOGGER.debug("Double precision read failed with exception", exc_info=exc)
            results = None
        if not self._last_good_read or results is None or self._values_look_corrupt():
            results = self._read_with_fallback(
                filename,
                convergence_error=convergence_error,
                darcy_weisbach=darcy_weisbach,
                convert=convert,
            )
        return results

    def _read_with_fallback(self, filename, convergence_error=False, darcy_weisbach=False, convert=True):
        if self._fallback_reader is None:
            result_types = None if self._result_types_arg is None else list(self._result_types_arg)
            self._fallback_reader = wntr.epanet.io.BinFile(
                result_types=result_types,
                network=self.create_network,
                energy=self.keep_energy,
                statistics=self.keep_statistics,
                convert_status=self.convert_status,
            )
        if self._fallback_exception is not None or self._last_good_read is False:
            _LOGGER.debug("Falling back to float32 EPANET binary reader")
        results = self._fallback_reader.read(
            filename,
            convergence_error=convergence_error,
            darcy_weisbach=darcy_weisbach,
            convert=convert,
        )
        self._copy_state_from(self._fallback_reader)
        self._fallback_used = True
        self._last_good_read = True
        return results

    def _copy_state_from(self, other: wntr.epanet.io.BinFile) -> None:
        for attr in (
            "flow_units",
            "pres_units",
            "quality_type",
            "mass_units",
            "num_nodes",
            "num_tanks",
            "num_links",
            "num_pumps",
            "num_valves",
            "report_start",
            "report_step",
            "duration",
            "chemical",
            "chem_units",
            "inp_file",
            "report_file",
            "node_names",
            "link_names",
            "report_times",
            "num_periods",
            "averages",
            "peak_energy",
        ):
            if hasattr(other, attr):
                setattr(self, attr, getattr(other, attr))
        self.results = other.results

    @property
    def used_fallback(self) -> bool:
        """Return ``True`` if the float32 reader handled the most recent run."""

        return self._fallback_used

    def _values_look_corrupt(self) -> bool:
        if self.results is None:
            return False

        def _max_abs(df) -> float:
            if df is None:
                return 0.0
            values = getattr(df, "values", None)
            if values is None or values.size == 0:
                return 0.0
            finite = np.isfinite(values)
            if not finite.any():
                return 0.0
            return float(np.nanmax(np.abs(values[finite])))

        limit = 1e9
        node_frames = [
            self.results.node.get("pressure"),
            self.results.node.get("head"),
        ]
        link_frames = [
            self.results.link.get("flowrate"),
            self.results.link.get("velocity"),
            self.results.link.get("headloss"),
            self.results.link.get("setting"),
            self.results.link.get("reaction_rate"),
            self.results.link.get("friction_factor"),
        ]
        for df in node_frames + link_frames:
            if _max_abs(df) > limit:
                _LOGGER.debug("Detected implausible values while using double precision reader; triggering fallback")
                return True
        return False


def make_simulator(
    wn: wntr.network.WaterNetworkModel,
    *,
    result_types: Optional[Iterable[wntr.epanet.io.ResultType]] = None,
    network: bool = False,
    energy: bool = False,
    statistics: bool = False,
    convert_status: bool = True,
) -> wntr.sim.EpanetSimulator:
    """Create an :class:`~wntr.sim.EpanetSimulator` with a double-precision reader."""

    reader = DoublePrecisionBinFile(
        result_types=result_types,
        network=network,
        energy=energy,
        statistics=statistics,
        convert_status=convert_status,
    )
    return wntr.sim.EpanetSimulator(wn, reader=reader, result_types=result_types)


__all__ = ["DoublePrecisionBinFile", "make_simulator"]
