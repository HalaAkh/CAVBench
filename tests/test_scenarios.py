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
