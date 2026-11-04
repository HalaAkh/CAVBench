"""The rollout model: uncertainty floor, PET danger and human drivers giving way."""
import numpy as np

from cavbench.arbiters import make_arbiter
from cavbench.arbiters.whatif_core import RolloutConfig, evaluate_options
from cavbench.envs import make_env


def _situations(n=40, noise="none"):
    env = make_env("intersection", {"noise_level": noise})
    arb = make_arbiter("whatif_fuzzy")
    env.reset(seed=1000)
    arb.reset(seed=1)
    rng = np.random.default_rng(0)
    out = []
    while len(out) < n:
        obs = env.observe(rng)
        out += [(obs, g) for g in arb.detector.groups(obs)]
        acts = arb.act(obs, env.active_cav_vids())
        _, _, te, tr, _ = env.step(tuple(int(acts.get(v, 2)) for v in env.cav_vids()))
        if te or tr:
            break
    return out


def test_uncertainty_floor_spreads_outcomes_without_noise():
    sits = _situations()
    assert sits
    flat = RolloutConfig(sigma_pos_min=0, sigma_vel_min=0, hdv_speed_sd=0, p_hdv_yield=0)
    for obs, g in sits[:10]:
        u0 = [r.uncertainty for r in evaluate_options(obs, g, flat, np.random.default_rng(1))]
        u1 = [r.uncertainty for r in evaluate_options(obs, g, RolloutConfig(), np.random.default_rng(1))]
        assert max(u0) < 1e-9  # no noise, no floor: one deterministic future
        assert max(u1) >= max(u0)


def test_pet_criterion_only_adds_danger():
    for obs, g in _situations():
        rng = lambda: np.random.default_rng(3)
        with_pet = evaluate_options(obs, g, RolloutConfig(), rng(), True)
        without = evaluate_options(obs, g, RolloutConfig(pet_safe=0), rng(), True)
        for a, b in zip(with_pet, without):
            assert a.risk >= b.risk - 1e-12


def test_pruning_does_not_change_risk():
    for obs, g in _situations(noise="high"):
        a = evaluate_options(obs, g, RolloutConfig(), np.random.default_rng(4), True)
        b = evaluate_options(obs, g, RolloutConfig(prune_pairs=False), np.random.default_rng(4), True)
        assert [r.risk for r in a] == [r.risk for r in b]
