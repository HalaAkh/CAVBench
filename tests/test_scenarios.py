"""Scenario set-up: no vehicles start on top of each other; the optional
Case 2 set-up (hdv_per_approach) and the unchanged default scenes."""
import hashlib
import itertools

import numpy as np
import pytest

from cavbench.envs import make_env

CASES = [
    ("merge", {}),
    ("intersection", {}),
    ("intersection", {"controlled_vehicles": 8, "initial_vehicle_count": 14}),
    ("intersection", {"hdv_per_approach": 1}),
]


@pytest.mark.parametrize("scenario,cfg", CASES)
def test_no_overlapping_start(scenario, cfg):
    env = make_env(scenario, cfg)
    for seed in range(30):
        env.reset(seed=seed)
        cavs = env.controlled_vehicles
        assert not any(v.crashed for v in env.road.vehicles), seed
        for a, b in itertools.combinations(cavs, 2):
            assert np.linalg.norm(a.position - b.position) > 8.0, (scenario, seed)


def test_intersection_does_not_change_other_scenarios_drivers():
    """The intersection's human-driver settings must stay local to it. Before the
    fix they were written on HighwayEnv's shared IDMVehicle class, so a merge
    episode run after an intersection episode (same worker process) had
    different human drivers than one run before it."""
    from highway_env.vehicle.behavior import IDMVehicle

    from cavbench.arbiters import make_arbiter
    from cavbench.envs.intersection import IntersectionIDMVehicle
    from cavbench.runner import run_episode

    shared = (IDMVehicle.DISTANCE_WANTED, IDMVehicle.COMFORT_ACC_MAX, IDMVehicle.COMFORT_ACC_MIN)
    env = make_env("intersection", {})
    env.reset(seed=1)
    assert (IDMVehicle.DISTANCE_WANTED, IDMVehicle.COMFORT_ACC_MAX,
            IDMVehicle.COMFORT_ACC_MIN) == shared
    hdvs = [v for v in env.road.vehicles if v not in env.controlled_vehicles]
    assert hdvs and all(type(v) is IntersectionIDMVehicle for v in hdvs)
    assert (hdvs[0].DISTANCE_WANTED, hdvs[0].COMFORT_ACC_MAX, hdvs[0].COMFORT_ACC_MIN) == (7.0, 6.0, -3.0)

    def merge_result():
        m = run_episode(make_env("merge", {}), make_arbiter("fcfs"), 100009)
        return m["success"], m["episode_time"], m["min_pet"], m["avg_speed"]

    before = merge_result()
    run_episode(make_env("intersection", {}), make_arbiter("fcfs"), 100000)
    assert merge_result() == before


def test_hdv_per_approach_gives_four_plus_four():
    """Case 2 set-up: 4 CAVs + exactly 4 HDVs, one per approach, each 15-20 m
    ahead of (closer to the junction than) the CAV on its approach."""
    env = make_env("intersection", {"hdv_per_approach": 1, "spawn_during_episode": False})
    for seed in range(100000, 100020):
        env.reset(seed=seed)
        cavs = list(env.controlled_vehicles)
        hdvs = [v for v in env.road.vehicles if v not in cavs]
        assert len(cavs) == 4 and len(hdvs) == 4, seed
        assert sorted(v.lane_index[0] for v in hdvs) == ["o0", "o1", "o2", "o3"], seed
        for h in hdvs:
            cav = next(c for c in cavs if c.lane_index[0] == h.lane_index[0])
            ahead = h.lane.local_coordinates(h.position)[0] - cav.lane.local_coordinates(cav.position)[0]
            assert 15.0 - 1e-6 <= ahead <= 20.0 + 1e-6, (seed, ahead)
            assert h.route[-1][1] != "o" + h.lane_index[0][1:], seed  # does not turn back


def _scene_fingerprint(cfg):
    # recorded with CAVBench's own rules (benchmark mode draws CAV exits differently)
    env = make_env("intersection", {**cfg, "benchmark_mode": False})
    h = hashlib.sha256()
    for seed in range(10):
        env.reset(seed=seed)
        for v in env.road.vehicles:
            h.update(f"{v.position[0]:.6f},{v.position[1]:.6f},{v.speed:.6f},"
                     f"{v.lane_index},{getattr(v, 'route', None)}|".encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("cfg,expected", [
    ({}, "3c0a95ac3d89bb9b"),
    ({"controlled_vehicles": 8, "initial_vehicle_count": 14}, "1e8982c2900f4e4b"),
])
def test_default_intersection_scenes_unchanged(cfg, expected):
    """hdv_per_approach is off by default: the existing scenes (positions,
    speeds, routes of every vehicle, seeds 0-9) are exactly those recorded
    before the option was added."""
    assert _scene_fingerprint(cfg) == expected
