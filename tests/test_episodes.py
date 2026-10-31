import pytest

from cavbench.arbiters import make_arbiter
from cavbench.arbiters.codrivingllm_wrapper import parse_order
from cavbench.envs import make_env
from cavbench.runner import run_episode

FAST = {"rollout": {"n_rollouts": 20}}


@pytest.mark.parametrize("scenario", ["intersection", "merge"])
@pytest.mark.parametrize("kind,params", [
    ("whatif_fuzzy", FAST), ("fcfs", {}), ("no_coordination", {}),
    ("codrivingllm", {"llm_fn": lambda prompt: '{"order": []}'}),
])
def test_episode_runs(scenario, kind, params):
    env = make_env(scenario, {"noise_level": "medium", "duration": 15})
    m = run_episode(env, make_arbiter(kind, **params), seed=0)
    for key in ["success", "collision", "avg_speed", "episode_time", "severe_conflicts"]:
        assert key in m
    assert m["episode_time"] <= 15.5


def test_matched_seeds_are_reproducible():
    env = make_env("intersection", {"noise_level": "high", "duration": 15})
    a = run_episode(env, make_arbiter("fcfs"), seed=3)
    b = run_episode(env, make_arbiter("fcfs"), seed=3)
    assert a["avg_speed"] == b["avg_speed"] and a["episode_time"] == b["episode_time"]


def test_ablation_switches():
    env = make_env("intersection", {"noise_level": "high", "duration": 10})
    arb = make_arbiter("whatif_fuzzy", use_uncertainty=False, use_fallback=False, **FAST)
    run_episode(env, arb, seed=1)
    assert all(e["mode"] != "fallback" for e in arb.log)
    assert all(u == 0.0 for e in arb.log for u in e["uncertainty"])


def test_llm_reply_parsing():
    assert parse_order('ok {"order": [3, 1]}', [1, 2, 3]) == [3, 1, 2]
    assert parse_order("no json here", [1, 2]) is None
