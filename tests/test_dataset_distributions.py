from pathlib import Path
import struct
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.data_generation import plot_dataset_distributions
from scripts.wntr_compat import DoublePrecisionBinFile
import wntr
from wntr.epanet.util import FlowUnits, PressureUnits, QualType, StatisticsType


def test_plot_dataset_distributions(tmp_path: Path):
    plot_dataset_distributions([0.8, 1.0, 1.2], [0.0, 0.5, 1.0], "unit", plots_dir=tmp_path)
    assert (tmp_path / "dataset_distributions_unit.png").exists()


def _pad_bytes(text: str, length: int) -> bytes:
    data = text.encode("ascii")
    if len(data) > length:
        raise ValueError("text too long")
    return data + b"\0" * (length - len(data))


def test_double_precision_reader_parses_fake_binary(tmp_path: Path) -> None:
    """Ensure the compatibility reader correctly handles 64-bit hydraulics files."""

    bin_path = tmp_path / "fake_double.bin"
    magic = 516114521
    version = 20012
    nnodes = 1
    ntanks = 0
    nlinks = 1
    npumps = 0
    nvalve = 0
    prolog = struct.pack(
        "<15i",
        magic,
        version,
        nnodes,
        ntanks,
        nlinks,
        npumps,
        nvalve,
        QualType.none.value,
        0,
        int(FlowUnits.SI),
        PressureUnits.Meters.value,
        StatisticsType.none.value,
        0,
        3600,
        0,
    )

    with bin_path.open("wb") as fh:
        fh.write(prolog)
        fh.write(b"\0" * 240)  # title records
        fh.write(_pad_bytes("fake.inp", 260))
        fh.write(_pad_bytes("fake.rpt", 260))
        fh.write(_pad_bytes("", 32))  # chemical name
        fh.write(_pad_bytes("mg/L", 32))
        fh.write(_pad_bytes("NODE1", 32))
        fh.write(_pad_bytes("LINK1", 32))
        fh.write(struct.pack("<i", 1))  # link start node index
        fh.write(struct.pack("<i", 1))  # link end node index
        fh.write(struct.pack("<i", 0))  # link type (pipe)
        fh.write(struct.pack("<d", 10.0))  # node elevation
        fh.write(struct.pack("<d", 100.0))  # link length
        fh.write(struct.pack("<d", 0.5))  # link diameter
        fh.write(struct.pack("<d", 0.0))  # peak energy placeholder
        # Hydraulic results for one report period (4 node + 8 link values)
        values = (
            1.5,
            101.25,
            50.125,
            0.5,
            2.25,
            0.5,
            1.75,
            0.25,
            1.0,
            1.2,
            0.05,
            0.02,
        )
        fh.write(struct.pack("<12d", *values))
        fh.write(struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))
        fh.write(struct.pack("<i", 0))  # number of periods
        fh.write(struct.pack("<i", 0))  # warning flag
        fh.write(struct.pack("<i", magic))

    reader = DoublePrecisionBinFile()
    results = reader.read(str(bin_path))
    pressure_df = results.node["pressure"]
    flow_df = results.link["flowrate"]
    assert pressure_df.iloc[0, 0] == pytest.approx(50.125)
    assert flow_df.iloc[0, 0] == pytest.approx(2.25)
    assert not reader.used_fallback

    legacy_reader = wntr.epanet.io.BinFile()
    legacy_results = legacy_reader.read(str(bin_path))
    legacy_pressure = legacy_results.node["pressure"].iloc[0, 0]
    # Float32 interpretation truncates the pressure to zero once the file is misaligned
    assert legacy_pressure == pytest.approx(0.0)
