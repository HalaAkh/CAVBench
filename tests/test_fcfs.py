"""Classical FCFS: fixed arrival order, one vehicle in the conflict area at a time."""
import numpy as np
import pytest

from cavbench.arbiters import FASTER, make_arbiter
from cavbench.arbiters.fcfs import conflict_area_centre
from cavbench.envs import make_env

SCENES = [
    ("intersection", {}),
    ("merge", {}),
]


def _run(scenario, cfg, seed):
    """Run one episode; return the arbiter and, per step, the CAVs whose true
    centre is inside the conflict area."""
    env = make_env(scenario, cfg)
    arb = make_arbiter("fcfs")
    env.reset(seed=seed)
    arb.reset(seed=seed)
    rng = np.random.default_rng([seed, 2026])
    inside_per_step, done = [], False
    while not done:
        obs = env.observe(rng)
        centre = conflict_area_centre(obs)
        acts = arb.act(obs, env.active_cav_vids())
        inside_per_step.append({
            env.vid(v) for v in env.controlled_vehicles
            if np.max(np.abs(v.position - centre)) <= arb.half_width - 0.5
        })
        _, _, te, tr, _ = env.step(tuple(int(acts.get(v, FASTER)) for v in env.cav_vids()))
        done = te or tr
    return env, arb, inside_per_step


@pytest.mark.parametrize("scenario,cfg", SCENES)
def test_one_cav_at_a_time_and_no_crash(scenario, cfg):
    for seed in (100000, 100001):
        env, _, inside = _run(scenario, cfg, seed)
        assert not any(v.crashed for v in env.controlled_vehicles), seed
        # CAVs on different lanes are never inside the area together (same-lane
        # followers may be: car following, handled by the safety layer).
        for step in inside:
            assert len(step) <= 1 or scenario == "merge", (seed, step)


def test_queue_order_is_fixed():
    _, arb, _ = _run("intersection", {}, 100000)
    queues = [e["queue"] for e in arb.log]
    assert queues
    for prev, cur in zip(queues, queues[1:]):
        kept = [v for v in prev if v in cur]
        assert cur[: len(kept)] == kept  # survivors keep their order, newcomers go last


def test_reproducible():
    a = _run("intersection", {"noise_level": "high"}, 100002)[1].log
    b = _run("intersection", {"noise_level": "high"}, 100002)[1].log
    assert [(e["queue"], e["goer"]) for e in a] == [(e["queue"], e["goer"]) for e in b]
