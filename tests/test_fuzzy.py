import numpy as np

from cavbench.arbiters.fuzzy import A_HIGH, A_LOW, RULES, FuzzyAcceptability, acceptance_bounds, band


def test_rule_base_size():
    assert len(RULES) == 13


def test_symbolic_bounds_match_proposition_1():
    b = acceptance_bounds()
    assert b["accept_short_wait"] == 0.2
    assert b["accept_any_wait"] == 0.5
    assert b["proceed_short_wait"] == 0.5
    assert b["short_wait_limit"] == 1.0


def test_proposition_1_numerically():
    """Over a grid of inputs: Accept with W <= 1 s implies R < 0.2, etc."""
    fz = FuzzyAcceptability(resolution=401)
    R, U = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 2, 41), indexing="ij")
    for w in np.linspace(0, 6, 25):
        A = fz.infer(R, U, np.full_like(R, w))
        if w <= 1.0:
            assert R[A > A_HIGH].max(initial=0) < 0.2
            assert R[A > A_LOW].max(initial=0) < 0.5
        assert R[A > A_HIGH].max(initial=0) < 0.5


def test_obvious_cases():
    fz = FuzzyAcceptability()
    assert band(float(fz.infer(0.0, 0.0, 0.0))) == "accept"
    assert band(float(fz.infer(1.0, 0.1, 0.0))) == "reject"
    # a long wait makes a medium-risk, predictable maneuver acceptable
    assert fz.infer(0.3, 0.05, 5.0) > fz.infer(0.3, 0.05, 0.0)
