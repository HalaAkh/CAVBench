"""Scenario set-up: no vehicles start on top of each other."""
import itertools

import numpy as np
import pytest

from cavbench.envs import make_env

CASES = [
    ("merge", {}),
    ("intersection", {}),
    ("intersection", {"controlled_vehicles": 8, "initial_vehicle_count": 14}),
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
