"""Tests of benchmark mode (the CoDrivingLLM benchmark's simulation rules)."""
import numpy as np

from cavbench.arbiters import make_arbiter
from cavbench.envs import make_env
from cavbench.runner import run_episode


def test_cav_exits_never_o0_in_benchmark_mode():
    env = make_env("intersection", {})
    for seed in range(30):
        env.reset(seed=seed)
        for v in env.controlled_vehicles:
            assert v.route[-1][1] != "o0"


def test_collision_rule_uses_shrunk_outlines():
    env = make_env("intersection", {})
    env.reset(seed=0)
    a, b = env.controlled_vehicles[:2]
    b.heading = a.heading
    b.position = a.position + np.array([np.cos(a.heading), np.sin(a.heading)]) * 4.7
    # full outlines (5 m) overlap by 0.3 m; 90 % outlines (4.5 m) do not
    assert not a._is_colliding(b, 0.0)[0]
    strict = make_env("intersection", {"benchmark_mode": False})
    strict.reset(seed=0)
    c, d = strict.controlled_vehicles[:2]
    d.heading = c.heading
    d.position = c.position + np.array([np.cos(c.heading), np.sin(c.heading)]) * 4.7
    assert c._is_colliding(d, 0.0)[0]


def test_arrived_cavs_leave_by_default_and_stay_on_request():
    env = make_env("intersection", {})
    m = run_episode(env, make_arbiter("whatif_fuzzy"), seed=1000)
    assert m["success"] and not any(v in env.road.vehicles for v in env.controlled_vehicles)
    env = make_env("intersection", {"bm_keep_arrived": True})
    env.reset(seed=1000)
    assert env.bm("keep_arrived") and env.bm("collision") and env.bm("exits")


def test_strict_mode_switches_every_rule_off():
    env = make_env("intersection", {"benchmark_mode": False})
    assert not any(env.bm(r) for r in ("collision", "exits", "keep_arrived",
                                       "keep_exited_hdvs", "stop_at_end"))
