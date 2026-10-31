import numpy as np

from cavbench.conflicts import ConflictDetector
from cavbench.envs import make_env


def test_intersection_start_has_cav_conflicts():
    env = make_env("intersection", {"noise_level": "none"})
    found = 0
    for seed in range(5):
        env.reset(seed=seed)
        rng = np.random.default_rng(0)
        for _ in range(25):  # let vehicles approach the intersection
            obs = env.observe(rng)
            groups = ConflictDetector().groups(obs)
            if any(len(g.cavs) >= 2 for g in groups):
                found += 1
                break
            env.step(tuple([1] * len(env.controlled_vehicles)))
    assert found >= 3


def test_dttcp_threshold_respected():
    env = make_env("intersection", {"noise_level": "none"})
    env.reset(seed=0)
    obs = env.observe(np.random.default_rng(0))
    for c in ConflictDetector().detect(obs):
        assert c.dttcp < 8.0
