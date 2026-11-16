"""MCTS baseline (Lin et al., T-ITS 2025): tree search, safety check, risk functions.

All tests use a small number of iterations to stay fast.
"""
import math

import numpy as np
import pytest

from cavbench.arbiters import FASTER, SLOWER, make_arbiter
from cavbench.arbiters.mcts import (
    MCTS_CALIBRATED, MCTSArbiter, MCTSCalibratedArbiter, MCTSConfig, MCTSPlanner, SceneModel,
    comfort_scale, dynamic_safe_distance, hdv_covariance, instantaneous_risk, rho,
    slower_step_acceleration, temporal_risk,
)
from cavbench.conflicts import ConflictGroup
from cavbench.envs import make_env
from cavbench.geometry import Path
from cavbench.runner import run_episode
from cavbench.state import SceneObservation, SharedState

L = 60.0
PATH_X = Path(np.c_[np.linspace(-L, L, 241), np.zeros(241)], key=(("tx_a", "tx_b"),))
PATH_Y = Path(np.c_[np.zeros(241), np.linspace(-L, L, 241)], key=(("ty_a", "ty_b"),))
ZONE = (np.zeros(2), 11.0)  # the crossing is at the origin


def crossing(d_a, v_a, d_b, v_b):
    """Two CAVs on perpendicular routes, d_a / d_b metres before the crossing point."""
    a = SharedState(0, True, PATH_X.position(L - d_a), v_a, PATH_X, L - d_a,
                    target_speed=9.0 if v_a > 6 else 4.5)
    b = SharedState(1, True, PATH_Y.position(L - d_b), v_b, PATH_Y, L - d_b,
                    target_speed=9.0 if v_b > 6 else 4.5)
    return SceneObservation(0.0, {0: a, 1: b}, 0.0, 0.0)


def planner(obs, cfg=None, seed=0):
    cfg = cfg or MCTSConfig()
    model = SceneModel(obs, ConflictGroup([0, 1]), cfg, np.random.default_rng(seed), {}, ZONE)
    return MCTSPlanner(model, cfg, np.random.default_rng(seed))


def test_registered_and_full_episode():
    arb = make_arbiter("mcts", n_iterations=10)
    assert isinstance(arb, MCTSArbiter) and arb.cfg.n_iterations == 10
    assert MCTSConfig().n_iterations == 1000 and MCTSConfig().d_max == 8  # Table I defaults
    m = run_episode(make_env("intersection", {"noise_level": "high"}), arb, seed=0)
    for key in ["success", "collision", "avg_speed", "episode_time", "decision_ms_mean",
                "mcts_iterations", "mcts_nodes", "mcts_pruned_fraction", "mcts_no_safe_root",
                "mcts_max_group_size"]:
        assert key in m
    assert m["mcts_iterations"] > 0 and m["mcts_iterations"] % 10 == 0
    assert m["mcts_nodes"] > 0 and 0.0 <= m["mcts_pruned_fraction"] <= 1.0
    assert m["mcts_max_group_size"] >= 1


def test_config_overrides_and_unsupported_options():
    arb = make_arbiter("mcts", n_iterations=5, d_max=3, config={"c_exp": 2.0})
    assert (arb.cfg.n_iterations, arb.cfg.d_max, arb.cfg.c_exp) == (5, 3, 2.0)
    with pytest.raises(NotImplementedError):
        make_arbiter("mcts", reuse_tree=True)


def test_reproducible():
    env = make_env("intersection", {"noise_level": "high"})
    env.reset(seed=2)
    rng = np.random.default_rng([2, 2026])
    probe = make_arbiter("mcts")
    for _ in range(100):
        obs = env.observe(rng)
        groups = probe.detector.groups(obs)
        if groups:
            break
        env.step(tuple(FASTER for _ in env.cav_vids()))
    assert groups
    results = []
    for _ in range(2):
        arb = make_arbiter("mcts", n_iterations=30)
        arb.reset(seed=7)
        acts, pl, _ = arb.plan(obs, groups[0])
        results.append((acts, pl.root.cN.copy(), pl.root.cQ.copy()))
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])
    assert np.array_equal(results[0][2], results[1][2])


def test_expand_creates_all_children_and_marks_collision_unsafe():
    pl = planner(crossing(8.0, 9.0, 8.0, 9.0))
    pl.expand(pl.root)
    r = pl.root
    assert len(r.ch_safe) == 3 ** 2 and pl.n_nodes == 1 + 9
    both_faster = int(np.flatnonzero((pl.joint == FASTER).all(axis=1))[0])
    both_slower = int(np.flatnonzero((pl.joint == SLOWER).all(axis=1))[0])
    assert not r.ch_safe[both_faster]  # both go FASTER into the crossing at the same time
    assert r.ch_safe[both_slower]
    # one CAV alone: 3 children
    obs = crossing(8.0, 9.0, 30.0, 9.0)
    del obs.states[1]
    cfg = MCTSConfig()
    one = MCTSPlanner(SceneModel(obs, ConflictGroup([0]), cfg, np.random.default_rng(0), {}, ZONE),
                      cfg, np.random.default_rng(0))
    one.expand(one.root)
    assert len(one.root.ch_safe) == 3


def test_unsafe_children_are_never_selected():
    pl = planner(crossing(8.0, 9.0, 8.0, 9.0), MCTSConfig(n_iterations=200))
    pl.run(200)
    assert (~pl.root.ch_safe).any() and pl.root.ch_safe.any()
    stack = [pl.root]
    while stack:
        n = stack.pop()
        if n.expanded:
            assert (n.cN[~n.ch_safe] == 0).all()
            stack.extend(n.children.values())
    k, ok = pl.best()
    assert ok and pl.root.ch_safe[k]


def test_ucb_backpropagation_and_best_child():
    cfg = MCTSConfig()
    pl = planner(crossing(30.0, 9.0, 30.0, 9.0), cfg)  # far apart: all children safe
    pl.expand(pl.root)
    r = pl.root
    assert r.ch_safe.all()
    # Eq. 25: an unvisited child has UCB = +inf
    r.N = 10
    r.cN[:] = 2
    r.cN[4] = 0
    assert pl.select(r) == 4
    r.cN[:] = np.arange(1, 10)
    r.cQ[:] = np.linspace(-1.0, 1.0, 9)
    ucb = r.cQ + cfg.c_exp * np.sqrt(math.log(10) / r.cN)
    assert pl.select(r) == int(np.argmax(ucb))
    # Eq. 28 on a two-level path: N <- N + 1, Q <- Q + (Q_eval - Q) / N^beta_dec
    r.N, r.Q = 3, 1.0
    r.cN[2], r.cQ[2] = 1, 2.0
    child = pl.child(r, 2)
    pl.backpropagate(child, 5.0)
    assert r.cN[2] == 2 and r.cQ[2] == pytest.approx(2.0 + (5.0 - 2.0) / 2 ** 0.75)
    assert r.N == 4 and r.Q == pytest.approx(1.0 + (5.0 - 1.0) / 4 ** 0.75)
    # Eq. 41: argmax of N + eps * Q over the safe children
    r.cN[:] = [5, 5, 4, 1, 1, 1, 1, 1, 1]
    r.cQ[:] = [-3.0, -1.0, 4.0, 0, 0, 0, 0, 0, 0]
    assert pl.best() == (2, True)  # 4 + 0.5 * 4 = 6 > 5 - 0.5
    r.ch_safe[2] = False
    assert pl.best() == (1, True)


def test_no_safe_root_child_returns_least_unsafe():
    obs = crossing(4.5, 9.0, 4.5, 9.0)  # collision is unavoidable: every child is unsafe
    arb = make_arbiter("mcts", n_iterations=20, zone_center=(0.0, 0.0))
    arb.reset(seed=0)
    acts = arb.arbitrate(obs, [ConflictGroup([0, 1])])
    pl = planner(obs)
    pl.expand(pl.root)
    assert not pl.root.ch_safe.any()
    least = int(np.argmin(pl.root.ch_risk))
    assert acts == {0: int(pl.joint[least][0]), 1: int(pl.joint[least][1])}
    assert arb.extra_metrics()["mcts_no_safe_root"] == 1
    assert arb.log[-1]["mode"] == "mcts_no_safe_root"


def test_dynamic_safe_distance_and_risk_functions():
    cfg = MCTSConfig()
    d_int = dynamic_safe_distance(2.0, 0.5, True, True, cfg, 9.0)
    d_app = dynamic_safe_distance(2.0, 0.5, True, False, cfg, 9.0)
    d_out = dynamic_safe_distance(2.0, 0.5, False, False, cfg, 9.0)
    assert d_int > d_app > d_out
    assert d_int == pytest.approx(3 * d_out) and d_app == pytest.approx(2 * d_out)
    dv = np.array([0.0, 2.0, 6.0, 12.0])
    assert np.all(np.diff(dynamic_safe_distance(dv, 0.0, False, False, cfg, 9.0)) > 0)
    th = np.array([0.0, 0.5, 1.5, np.pi])
    assert np.all(np.diff(dynamic_safe_distance(1.0, th, False, False, cfg, 9.0)) > 0)
    # Eq. 7 at rest: d_base
    assert dynamic_safe_distance(0.0, 0.0, False, False, cfg, 9.0) == pytest.approx(cfg.d_base)
    # Eq. 11
    assert rho(np.array([5.0, 6.0]), 5.0).tolist() == [0.0, 0.0]
    assert rho(2.5, 5.0) == pytest.approx(0.25) and rho(0.0, 5.0) == pytest.approx(1.0)
    # Eq. 9: risk 1 at contact without relative speed, larger with it
    assert instantaneous_risk(0.0, 4.0, 0.0, cfg, 9.0) == pytest.approx(1.0)
    assert instantaneous_risk(0.0, 4.0, 9.0, cfg, 9.0) == pytest.approx(1.3)
    # Eq. 10: weights 1/(1+t), averaged over T_p
    d_pred = np.zeros((5, 1))  # contact at every predicted point
    expected = sum(1 / (1 + t) for t in range(1, 6)) / 5
    assert temporal_risk(d_pred, np.array([4.0]), 5)[0] == pytest.approx(expected)
    assert temporal_risk(np.full((5, 1), 10.0), np.array([4.0]), 5)[0] == 0.0
    # Eq. 15: covariance grows with the prediction time and is a valid covariance
    c1, c2 = hdv_covariance(0.5, cfg), hdv_covariance(2.0, cfg)
    assert c2[0, 0] > c1[0, 0] and np.all(np.linalg.eigvalsh(c2) > 0)


def test_comfort_scale_formula():
    """comfort_scale = (a_med / a_step)^2, a_step = mean |acc| of one SLOWER tree step."""
    cfg = MCTSConfig()
    assert cfg.a_med is None and comfort_scale([0, 4.5, 9], cfg) == 1.0  # strict: off
    # Intersection 9 -> 4.5 m/s with kp = 1/0.6, bounds [-5, 3], 5 x 0.1 s:
    # -5, -5, -5, -5 (clipped), then kp * (4.5 - 7.0) = -4.1667  ->  v = 6.5833 m/s.
    v = 9.0
    for _ in range(5):
        v += max(min((4.5 - v) / 0.6, 3.0), -5.0) * 0.1
    a_step = (9.0 - v) / 0.5
    assert slower_step_acceleration([0, 4.5, 9], cfg) == pytest.approx(a_step)
    assert a_step == pytest.approx(4.8333, abs=1e-3)
    cal = MCTSConfig(a_med=1.5)
    assert comfort_scale([0, 4.5, 9], cal) == pytest.approx((1.5 / a_step) ** 2)
    assert comfort_scale([0, 6, 12], cal) == pytest.approx((1.5 / 5.0) ** 2)  # merge levels
    # It multiplies Q_acc and Q_jerk only: same state, the per-step reward changes by
    # (1 - scale) * (w2 w_acc acc^2 + w3 w_jerk jerk^2) for a single vehicle.
    obs = crossing(80.0, 9.0, 80.0, 4.5)  # far apart: no safety term
    m0 = SceneModel(obs, ConflictGroup([0, 1]), cfg, np.random.default_rng(0), {}, ZONE)
    m1 = SceneModel(obs, ConflictGroup([0, 1]), cal, np.random.default_rng(0), {}, ZONE)
    j = np.array([[SLOWER, SLOWER]])
    s0, s1 = m0.step(m0.root_state, j, 0), m1.step(m1.root_state, j, 0)
    np.testing.assert_allclose(s0.acc, s1.acc)
    cost = (cfg.w_self_eff * cfg.w_acc * s0.acc ** 2 + cfg.w_self_comfort * cfg.w_jerk * s0.jerk ** 2)
    k = m0.lam  # r_i = (q_self + lam q_other) / (1 + lam (n-1)); q_other has acc but no jerk
    acc_o = cfg.w_acc * s0.acc[0, ::-1] ** 2 * m0.w_other
    expect = ((1 - m1.comfort_scale) * (cost[0] + k * acc_o) / (1 + k)).sum()
    diff = m1.evaluate(s1, 1).reward[0] - m0.evaluate(s0, 1).reward[0]
    assert diff == pytest.approx(expect)


# Actions of the strict MCTS (40 iterations) in the first 25 decisions of intersection seed
# 100000, noise none, recorded with the version committed before the calibration (b4dccbe).
# Snapshot of the strict MCTS's joint actions. Regenerated after the route fix in
# cavbench/geometry.py (remaining_route keeps the lane a car is still on), which
# changes the shared states every arbiter sees; MCTS itself is unchanged.
STRICT_ACTIONS = ['2221', '2122', '2222', '2120', '2222', '2122', '2222', '2122', '2121',
                  '2121', '2222', '2222', '2222', '2122', '2222', '2222', '2221', '2122',
                  '2221', '2221', '2222', '2221', '2120', '2220', '2121']


def test_strict_kind_unchanged():
    seed = 100000
    # recorded with CAVBench's own rules (benchmark mode changes the scene)
    env = make_env("intersection", {"noise_level": "none", "benchmark_mode": False})
    arb = make_arbiter("mcts", n_iterations=40)
    assert arb.cfg.a_med is None and arb.cfg.jerk_def == "rate"
    assert arb.cfg == MCTSConfig(n_iterations=40)  # strict defaults
    env.reset(seed=seed)
    arb.reset(seed=seed + 10_000_019)
    rng = np.random.default_rng([seed, 2026])
    out = []
    for _ in range(len(STRICT_ACTIONS)):
        obs = env.observe(rng)
        acts = arb.act(obs, env.active_cav_vids())
        joint = tuple(int(acts.get(v, FASTER)) for v in env.cav_vids())
        out.append("".join(map(str, joint)))
        env.step(joint)
    assert out == STRICT_ACTIONS


def test_calibrated_kind_preset_and_overrides():
    arb = make_arbiter("mcts_calibrated")
    assert isinstance(arb, MCTSCalibratedArbiter) and arb.name == "mcts_calibrated"
    for k, v in MCTS_CALIBRATED.items():
        assert getattr(arb.cfg, k) == v
    strict = MCTSConfig()
    # Only calibration knobs differ from the strict version.
    changed = {k for k in strict.__dict__ if getattr(strict, k) != getattr(arb.cfg, k)}
    assert changed == set(MCTS_CALIBRATED) - {k for k, v in MCTS_CALIBRATED.items()
                                               if getattr(strict, k) == v}
    assert changed <= {"a_med", "w_v2v", "w_v2h", "tp_dt", "jerk_def"}
    assert make_arbiter("mcts_calibrated", n_iterations=7).cfg.n_iterations == 7
    with pytest.raises(ValueError):
        MCTSArbiter(jerk_def="other")



def test_calibrated_v2h_weight():
    """V2H calibration (docs/mcts_baseline.md, "Calibration" / "V2H"): the calibrated
    kind uses w_v2h = 10, the strict kind keeps 1; without human drivers (Case 1) the
    V2H weight cannot change any decision."""
    assert make_arbiter("mcts").cfg.w_v2h == 1.0
    assert make_arbiter("mcts_calibrated").cfg.w_v2h == 10.0
    assert MCTS_CALIBRATED["w_v2h"] == 10.0

    def actions(config):
        seed = 100000
        env = make_env("intersection", {"initial_vehicle_count": 0, "spawn_during_episode": False})
        arb = make_arbiter("mcts_calibrated", n_iterations=30, config=config)
        env.reset(seed=seed)
        arb.reset(seed=seed + 10_000_019)
        rng = np.random.default_rng([seed, 2026])
        out = []
        for _ in range(10):
            obs = env.observe(rng)
            acts = arb.act(obs, env.active_cav_vids())
            joint = tuple(int(acts.get(v, FASTER)) for v in env.cav_vids())
            out.append(joint)
            env.step(joint)
        return out

    assert actions(None) == actions({"w_v2h": 1.0})


def test_fixed_safe_distance_option():
    """Trend-check option: fixed_d_safe replaces Eq. 7-8 by a constant; off by default."""
    assert MCTSConfig().fixed_d_safe is None
    assert make_arbiter("mcts_calibrated").cfg.fixed_d_safe is None
    dv = np.array([[0.0, 5.0], [9.0, 2.0]])
    dth = np.array([[0.0, 1.0], [np.pi, 0.5]])
    app = np.array([[True, False], [True, True]])
    inn = np.array([[False, False], [True, False]])
    dyn = dynamic_safe_distance(dv, dth, app, inn, MCTSConfig(), 9.0)
    assert dyn.min() >= 1.5 and dyn.max() > 2.0  # dynamic: depends on the pair
    fixed = dynamic_safe_distance(dv, dth, app, inn, MCTSConfig(fixed_d_safe=2.0), 9.0)
    assert fixed.shape == dyn.shape and np.all(fixed == 2.0)
    # the option reaches the scene model (risk terms use the constant)
    obs = crossing(6.0, 8.0, 6.0, 8.0)  # close to the crossing: dynamic d_safe > 2 m
    m_fix = planner(obs, MCTSConfig(fixed_d_safe=2.0)).model
    m_dyn = planner(obs).model
    assert not np.allclose(m_fix.evaluate(m_fix.root_state, 0).risk,
                           m_dyn.evaluate(m_dyn.root_state, 0).risk)
