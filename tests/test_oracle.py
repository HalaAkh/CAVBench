"""Tests of the oracle upper bound (perfect information, passing orders)."""
import dataclasses

import numpy as np

from cavbench.arbiters import ARBITERS, FASTER, IDLE, SLOWER, make_arbiter
from cavbench.arbiters.base import Arbiter
from cavbench.arbiters.oracle import (OracleArbiter, OracleConfig, _Plan, choose, copy_env,
                                      true_observation)
from cavbench.conflicts import ConflictGroup
from cavbench.envs import make_env
from cavbench.geometry import Path
from cavbench.runner import run_episode
from cavbench.state import SceneObservation, SharedState

SEED = 100000  # tuning seed


def _no_timing(m):
    return {k: v for k, v in m.items() if not k.startswith("decision_ms") and k != "log"}


def _same(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k  # (nan == nan)


def _env_with_group(min_size=2, scenario="intersection", need_hdv=False):
    """A real environment stepped until a conflict group of >= min_size CAVs
    (and, with need_hdv, a conflict with a human driver) exists."""
    env = make_env(scenario, {"noise_level": "none"})
    env.reset(seed=SEED)
    arb = OracleArbiter()
    arb.reset(seed=1)
    arb.attach(env)
    for _ in range(40):
        obs = true_observation(env)
        groups = [g for g in arb.detector.groups(obs)
                  if len(g.cavs) >= min_size and (g.others or not need_hdv)]
        if groups:
            return env, arb, obs, groups[0]
        env.step(tuple([FASTER] * len(env.controlled_vehicles)))
    raise AssertionError("no conflict group found")


def _snapshot(env):
    return {
        "time": env.time, "steps": env.steps, "sim_time": env.sim_time,
        "vehicles": [(id(v), tuple(v.position), v.speed, v.heading, v.crashed,
                      getattr(v, "target_speed", None)) for v in env.road.vehicles],
        "rng": str(env.np_random.bit_generator.state),
        "road_rng": str(env.road.np_random.bit_generator.state),
        "trajectory": list(env.trajectory),
        "vid": dict(env._vid),
        "paths": dict(env.path_cache._paths),
    }


def test_registered_and_full_episode():
    assert ARBITERS["oracle"] is OracleArbiter
    m = run_episode(make_env("intersection", {"noise_level": "none"}), make_arbiter("oracle"), SEED)
    for k in ("success", "collision", "severe_conflicts", "min_pet", "avg_speed",
              "decision_ms_mean", "n_decisions", "oracle_orders_evaluated",
              "oracle_options_evaluated", "oracle_enum_share", "oracle_replans",
              "oracle_cap_reached", "oracle_max_group_size"):
        assert k in m
    assert m["n_decisions"] > 0 and m["oracle_replans"] > 0


def test_reproducible_and_ignores_noise():
    """Same seed -> same decisions. The oracle reads the true states only, so
    the episode is also identical at every noise level and V2X setting."""
    runs = [run_episode(make_env("intersection", cfg), make_arbiter("oracle"), SEED, keep_log=True)
            for cfg in ({"noise_level": "none"}, {"noise_level": "none"},
                        {"noise_level": "high", "v2x_level": "degraded"})]
    acts = [[e["actions"] for e in m["log"]] for m in runs]
    assert acts[0] == acts[1] == acts[2]
    _same(_no_timing(runs[0]), _no_timing(runs[1]))
    _same(_no_timing(runs[0]), _no_timing(runs[2]))


def test_evaluating_options_does_not_change_the_real_env():
    env, arb, obs, g = _env_with_group()
    before = _snapshot(env)
    arb.plan_order(obs, g)
    arb.simulate_option(obs, g, {g.cavs[0]})
    sim = copy_env(env)
    sim.step(tuple([FASTER] * len(env.controlled_vehicles)))
    assert _snapshot(env) == before
    # ... and the copy is exact: same ids, positions and random state
    assert true_observation(copy_env(env)).states.keys() == obs.states.keys()
    c = copy_env(env)
    assert str(c.np_random.bit_generator.state) == before["rng"]
    for vid, st in true_observation(c).states.items():
        assert np.allclose(st.position, obs.states[vid].position)


def test_mode_a_lists_all_orders_and_large_groups_use_mode_b():
    assert len(OracleArbiter.enumerate_orders([1, 2])) == 2
    assert len(OracleArbiter.enumerate_orders([1, 2, 3])) == 6
    assert len(set(OracleArbiter.enumerate_orders([1, 2, 3, 4]))) == 24
    env, arb, obs, g = _env_with_group()
    assert len(arb.candidates(g)) == 2 * len(OracleArbiter.enumerate_orders(g.cavs))
    arb.plan_order(obs, g)
    assert arb.n_orders == len(arb.candidates(g))  # every candidate is started (pruned or not)
    small = OracleArbiter(oracle={"max_enum_cavs": len(g.cavs) - 1})
    small.reset(seed=1)
    small.attach(env)
    small.arbitrate(obs, [g])
    assert small.log[-1]["mode"] in ("option", "fallback")
    assert small.n_options == 2 * len(g.cavs) + 1 and small.n_orders == 0
    assert small.extra_metrics()["oracle_enum_share"] == 0.0


def _line(key, start, end, n=201):
    return Path(np.linspace(start, end, n), key=(key,))


def _state(vid, path, s, speed=5.0, target=4.5, is_cav=True):
    return SharedState(vid=vid, is_cav=is_cav, position=path.position(s), speed=speed,
                       path=path, s=s, target_speed=target if is_cav else None)


def test_order_following_rule_yields_until_previous_cleared():
    pa = _line(("test_a", "x"), (-50.0, 0.0), (50.0, 0.0))  # meet near s = 49 m
    pb = _line(("test_b", "y"), (0.0, -50.0), (0.0, 50.0))
    arb = OracleArbiter()

    def obs(sa, sb, speed_b=5.0, target_b=4.5):
        return SceneObservation(0.0, {0: _state(0, pa, sa), 1: _state(1, pb, sb, speed_b, target_b)},
                                0.0, 0.0)

    assert arb.order_actions(obs(30.0, 40.0), (0, 1)) == {0: FASTER, 1: SLOWER}
    assert arb.order_actions(obs(30.0, 40.0), (1, 0)) == {1: FASTER, 0: SLOWER}
    assert arb.order_actions(obs(30.0, 40.0, 0.2, 0.0), (0, 1))[1] == IDLE  # nearly stopped
    assert arb.order_actions(obs(51.0, 40.0), (0, 1))[1] == SLOWER  # 0 is inside the area
    assert arb.order_actions(obs(56.0, 40.0), (0, 1)) == {0: FASTER, 1: FASTER}  # cleared
    gone = SceneObservation(0.0, {1: _state(1, pb, 40.0)}, 0.0, 0.0)  # 0 has left
    assert arb.order_actions(gone, (0, 1)) == {1: FASTER}
    assert not arb.has_cleared(obs(30.0, 40.0), 0, {0, 1})
    assert arb.has_cleared(obs(56.0, 40.0), 0, {0, 1})


def test_plan_is_kept_and_replanned_on_group_change_or_period():
    env, arb, obs, g = _env_with_group()
    arb.arbitrate(obs, [g])
    assert arb.log[-1]["mode"] == "enum_plan" and arb.n_plans_a == 1
    arb.arbitrate(dataclasses.replace(obs, time=obs.time + 0.2), [g])
    assert arb.log[-1]["mode"] == "enum_follow" and arb.n_plans_a == 1
    assert arb.log[-1]["order"] == arb.log[-2]["order"]
    arb.arbitrate(dataclasses.replace(obs, time=obs.time + 2.0), [g])  # replan_period
    assert arb.log[-1]["mode"] == "enum_plan" and arb.n_plans_a == 2
    sub = ConflictGroup([g.cavs[0]], [c for c in g.conflicts if c.i == g.cavs[0]])
    arb.arbitrate(dataclasses.replace(obs, time=obs.time + 2.2), [sub])  # group changed
    assert arb.log[-1]["mode"] == "enum_plan" and arb.n_plans_a == 3


def test_choice_rule_is_lexicographic():
    r = [
        {"collision": True, "severe": 0, "cost": 1.0, "tiebreak": (0,)},
        {"collision": False, "severe": 2, "cost": 2.0, "tiebreak": (1,)},
        {"collision": False, "severe": 1, "cost": 9.0, "tiebreak": (2,)},
        {"collision": False, "severe": 1, "cost": 5.0, "tiebreak": (4,)},
        {"collision": False, "severe": 1, "cost": 5.0, "tiebreak": (3,)},
    ]
    assert choose(r) == 4  # no collision > fewest severe > lowest cost > arrival order
    assert choose(r[:2]) == 1
    assert choose(r, keys=("cost",)) == 0  # the choice rule is a parameter


def test_pruning_does_not_change_the_choice():
    env, arb, obs, g = _env_with_group()
    full = OracleArbiter(oracle=OracleConfig(prune=False))
    full.reset(seed=1)
    full.attach(env)
    assert arb.plan_order(obs, g)[:2] == full.plan_order(obs, g)[:2]


def test_oracle_does_not_consume_noise_samples():
    """The runner keeps calling env.observe(noise_rng); the oracle never gets
    that generator and does not change the env, so the noise samples stay matched."""
    env, arb, _, _ = _env_with_group()
    env.config["noise_level"] = "high"
    rng_a, rng_b = np.random.default_rng([SEED, 2026]), np.random.default_rng([SEED, 2026])
    obs_a = env.observe(rng_a)
    for _ in range(3):  # the oracle decides (and simulates copies) between observations
        arb.act(obs_a, env.active_cav_vids())
    assert arb.n_orders > 0
    obs_a2 = env.observe(rng_a)
    obs_b = env.observe(rng_b)
    obs_b2 = env.observe(rng_b)  # without the oracle in between
    for vid in obs_b.states:
        assert np.array_equal(obs_a.states[vid].position, obs_b.states[vid].position)
        assert np.array_equal(obs_a2.states[vid].position, obs_b2.states[vid].position)
    assert np.array_equal(rng_a.normal(size=10), rng_b.normal(size=10))


def test_other_arbiters_unchanged():
    """The oracle uses the runner's existing attach(env) hook; no new hook was
    added to the base class, and the other arbiters do not have it."""
    assert not hasattr(Arbiter, "attach")
    for kind in ("fcfs", "whatif_fuzzy", "no_coordination"):
        assert not hasattr(make_arbiter(kind), "attach")


def test_cavs_can_yield_to_human_drivers():
    """Item 1: with hdv_yield a CAV waits (yield at the conflict point, true states) for a human
    driver that reaches the shared conflict point earlier; without it, it goes."""
    pa = _line(("test_c", "x"), (-50.0, 0.0), (50.0, 0.0))  # CAV 0, point near s = 49 m
    ph = _line(("test_h", "y"), (0.0, -50.0), (0.0, 50.0))  # human driver 9
    arb = OracleArbiter()

    def obs(s_cav, s_hdv, v_hdv=8.0):
        return SceneObservation(0.0, {0: _state(0, pa, s_cav, 5.0),
                                      9: _state(9, ph, s_hdv, v_hdv, is_cav=False)}, 0.0, 0.0)

    earlier = obs(30.0, 40.0)  # TTCP: CAV 19 m / 5 m/s = 3.8 s, human 9 m / 8 m/s = 1.1 s
    assert arb.hdv_blocks(earlier, 0)
    assert arb.order_actions(earlier, (0,), hdv_yield=True) == {0: SLOWER}
    assert arb.order_actions(earlier, (0,), hdv_yield=False) == {0: FASTER}  # "go before it"
    assert not arb.hdv_blocks(obs(30.0, 5.0), 0)  # human arrives later (5.5 s): the CAV goes
    assert not arb.hdv_blocks(obs(30.0, 56.0), 0)  # human has cleared the point
    assert not arb.hdv_blocks(obs(50.0, 40.0), 0)  # CAV already at the point: it goes on


def test_candidates_include_wait_and_go_for_human_drivers():
    """Item 1: every order is tried with "wait for earlier human drivers" and
    "go before them"; a single CAV in conflict with a human driver now has a choice."""
    env, arb, obs, g = _env_with_group(min_size=1, need_hdv=True)
    cands = arb.candidates(g)
    assert {y for _, y in cands} == {True, False}
    assert len(cands) == 2 * len(OracleArbiter.enumerate_orders(g.cavs))
    order, hdv_yield, evals = arb.plan_order(obs, g)
    assert len(evals) == len(cands) and isinstance(hdv_yield, bool)


def test_other_groups_follow_their_plan_in_the_copy():
    """Item 2: in a copy, the CAVs of the other groups follow their own current
    plan (order-following rule), the CAVs in no group drive FASTER."""
    pa = _line(("test_d", "x"), (-50.0, 0.0), (50.0, 0.0))
    pb = _line(("test_e", "y"), (0.0, -50.0), (0.0, 50.0))
    pc = _line(("test_f", "z"), (-50.0, 30.0), (50.0, 30.0))  # meets nobody
    arb = OracleArbiter(safety=False)
    obs = SceneObservation(0.0, {0: _state(0, pa, 30.0), 1: _state(1, pb, 40.0),
                                 2: _state(2, pc, 10.0), 3: _state(3, pc, 60.0)}, 0.0, 0.0)
    other = frozenset({0, 1})
    arb.context = {other: _Plan((0, 1), True, 0.0)}
    planned = ConflictGroup([3])
    acts = arb.copy_actions(obs, planned, {3: SLOWER})
    assert acts == {0: FASTER, 1: SLOWER, 2: FASTER, 3: SLOWER}  # 1 waits for 0, as planned
    arb.context = {other: _Plan((1, 0), True, 0.0)}
    assert arb.copy_actions(obs, planned, {3: SLOWER})[0] == SLOWER
    # the context of a group never overrides that group's own candidate
    arb.context = {frozenset({3}): _Plan((3,), True, 0.0)}
    assert arb.copy_actions(obs, planned, {3: SLOWER})[3] == SLOWER


def test_context_holds_a_plan_for_every_group():
    env, arb, obs, g = _env_with_group()
    groups = arb.detector.groups(obs)
    arb.arbitrate(obs, groups)
    assert set(arb.context) == {frozenset(x.cavs) for x in groups}
    assert arb.context[frozenset(g.cavs)] is arb.plans[frozenset(g.cavs)]
