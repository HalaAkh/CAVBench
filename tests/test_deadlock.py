"""Tests of UCRA's deadlock breaker and of the route fix it relies on."""
import numpy as np

from cavbench.arbiters import make_arbiter
from cavbench.envs import make_env
from cavbench.runner import run_episode


def test_previously_stuck_episodes_now_finish():
    # E1 episodes in which UCRA used to wait until the 50 s time-out
    for scenario, cfg, seed in [("intersection", {"noise_level": "none"}, 105),
                                ("intersection", {"noise_level": "low"}, 71),
                                ("merge", {"noise_level": "none"}, 18)]:
        m = run_episode(make_env(scenario, cfg), make_arbiter("whatif_fuzzy"), seed)
        assert m["success"], (scenario, seed)


def test_breaker_can_be_switched_off():
    env = make_env("intersection", {"noise_level": "none"})
    m = run_episode(env, make_arbiter("whatif_fuzzy", use_deadlock_breaker=False), 105)
    assert m["n_deadlock"] == 0


def test_route_keeps_the_lane_the_car_is_still_on():
    from cavbench.geometry import remaining_route
    env = make_env("intersection", {"noise_level": "none"})
    env.reset(seed=0)
    v = env.controlled_vehicles[0]
    for _ in range(200):  # drive until the followed lane switches before the lane end
        env.step(tuple(2 for _ in env.controlled_vehicles))
        if v.target_lane_index[:2] != v.lane_index[:2]:
            r = remaining_route(v)
            assert r[0][:2] == v.lane_index[:2] and r[1][:2] == v.target_lane_index[:2]
            return
