def test_min_clip():
    pressures = {'A': -1e-6, 'B': 10.0}
    chlorine = {'A': -0.5, 'B': 0.3}
    min_p = max(min(pressures.values()), 0.0)
    min_c = max(min(chlorine.values()), 0.0)
    assert min_p >= 0
    assert min_c >= 0
