import numpy as np
import pytest

from scripts.mpc_control import PressureBiasCorrector


def test_bias_corrector_update_apply_reset():
    bc = PressureBiasCorrector(num_nodes=3, window=2)
    node_map = {"a": 0, "b": 1, "c": 2}
    pressures = {"a": 10.0, "b": 20.0, "c": 30.0}

    bc.update(np.array([1.0, -2.0, 0.5]))
    corrected = bc.apply(dict(pressures), node_map)
    assert corrected["a"] == pytest.approx(9.0)
    assert corrected["b"] == pytest.approx(22.0)

    bc.update(np.array([2.0, -4.0, 1.0]))
    assert bc.bias[0] == pytest.approx(1.5)  # rolling mean

    bc.reset()
    corrected_reset = bc.apply(dict(pressures), node_map)
    assert corrected_reset["a"] == pytest.approx(10.0)
    assert np.allclose(bc.bias, 0.0)
