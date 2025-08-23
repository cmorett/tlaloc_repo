def test_min_clip():
    pressures = {'A': -1e-6, 'B': 10.0}
    min_p = max(min(pressures.values()), 0.0)
    assert min_p >= 0
