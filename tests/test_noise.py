import numpy as np

from cavbench.noise import NOISE_LEVELS, GaussianNoise


def test_levels_match_report():
    assert NOISE_LEVELS == {"none": (0.0, 0.0), "low": (0.2, 0.1),
                            "medium": (0.5, 0.3), "high": (1.5, 1.0)}


def test_zero_noise_is_identity():
    pos = np.array([[1.0, 2.0], [3.0, 4.0]])
    spd = np.array([5.0, 6.0])
    p, s = GaussianNoise.from_level("none").perturb(pos, spd, np.random.default_rng(0))
    assert np.allclose(p, pos) and np.allclose(s, spd)


def test_noise_statistics():
    rng = np.random.default_rng(1)
    n = GaussianNoise(1.5, 1.0)
    pos = np.zeros((20000, 2))
    spd = np.full(20000, 10.0)
    p, s = n.perturb(pos, spd, rng)
    assert abs(p.std() - 1.5) < 0.05
    assert abs((s - 10.0).std() - 1.0) < 0.05
    assert (s >= 0).all()
