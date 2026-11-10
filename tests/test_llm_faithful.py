"""Faithful CoDrivingLLM mode, memory and .env handling -- all offline (mock model)."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from cavbench.arbiters import FASTER, IDLE, SLOWER, make_arbiter
from cavbench.arbiters import codrivingllm_wrapper as w
from cavbench.arbiters.llm_faithful import (
    DECISION_SYSTEM, NEGOTIATION_SYSTEM, allowed_actions, decision_prompt, parse_action,
)
from cavbench.arbiters.llm_memory import FEEDBACK_BAD, FEEDBACK_GOOD, LLMMemory
from cavbench.envs import make_env
from cavbench.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]


class CountingMock:
    """Wraps the free mock model and counts calls per system prompt."""

    def __init__(self):
        self.client = w.LLMClient("mock")
        self.negotiation = self.decision = self.other = 0

    def __call__(self, prompt, system=None):
        if system == NEGOTIATION_SYSTEM:
            self.negotiation += 1
        elif system == DECISION_SYSTEM:
            self.decision += 1
        else:
            self.other += 1
        return self.client(prompt, system=system)


def _drive(env, arb, seed, n, decide):
    """Run up to n steps; ``decide(obs, active_cavs)`` returns the actions."""
    env.reset(seed=seed)
    arb.reset(seed=seed + 1)
    rng = np.random.default_rng([seed, 2026])
    for _ in range(n):
        obs = env.observe(rng)
        acts = decide(obs, env.active_cav_vids())
        _, _, te, tr, _ = env.step(tuple(int(acts.get(v, FASTER)) for v in env.cav_vids()))
        if te or tr:
            return


def test_faithful_calls_every_step():
    fn = CountingMock()
    arb = make_arbiter("codrivingllm_faithful", llm_fn=fn)
    env = make_env("intersection", {"noise_level": "high"})
    seen = {"conflict": 0, "free": 0}

    def decide(obs, cavs):
        n0, d0 = fn.negotiation, fn.decision
        acts = arb.act(obs, cavs)
        assert fn.negotiation - n0 == 1
        assert fn.decision - d0 == len([v for v in cavs if not obs.states[v].arrived])
        seen["conflict" if arb.detector.groups(obs) else "free"] += 1
        return acts

    _drive(env, arb, seed=0, n=60, decide=decide)
    with_conf, without_conf = seen["conflict"], seen["free"]
    assert with_conf > 0 and without_conf > 0  # both kinds of steps were checked
    assert fn.other == 0
    m = arb.extra_metrics()
    assert m["llm_calls_negotiation"] == fn.negotiation and m["llm_calls_decision"] == fn.decision


def test_simple_mode_calls_only_on_conflicts():
    calls = []
    mock = w.LLMClient("mock")

    def fn(prompt):
        calls.append(prompt)
        return mock(prompt)

    arb = make_arbiter("codrivingllm", llm_fn=fn)
    assert arb.mode == "simple"
    env = make_env("intersection", {"noise_level": "high"})
    def decide(obs, cavs):
        before = len(calls)
        acts = arb.act(obs, cavs)
        assert len(calls) - before == len(arb.detector.groups(obs))  # one per group, none without
        return acts

    _drive(env, arb, seed=0, n=60, decide=decide)
    assert calls  # some conflicts happened
    assert "llm_calls_negotiation" not in arb.extra_metrics()


def test_action_safety_check():
    # never empty
    for cap in (SLOWER, IDLE, FASTER):
        for sev in ([], ["slight"], ["general"], ["serious"], ["serious", "general"]):
            allowed = allowed_actions(cap, sev)
            assert allowed and SLOWER in allowed and allowed == sorted(allowed)
    assert allowed_actions(FASTER, []) == [SLOWER, IDLE, FASTER]
    # leader too close
    assert allowed_actions(IDLE, []) == [SLOWER, IDLE]
    assert allowed_actions(SLOWER, []) == [SLOWER]
    # must yield
    assert allowed_actions(FASTER, ["general"]) == [SLOWER, IDLE]
    assert allowed_actions(FASTER, ["serious"]) == [SLOWER]
    assert allowed_actions(FASTER, ["slight"]) == [SLOWER, IDLE, FASTER]


def test_invalid_answers_fall_back_to_most_cautious():
    assert parse_action('{"action": "FASTER"}', [SLOWER, IDLE]) is None  # not allowed
    assert parse_action("no json", [SLOWER, IDLE, FASTER]) is None
    assert parse_action('{"action": "idle", "reason": "x"}', [SLOWER, IDLE]) == IDLE

    arb = make_arbiter("codrivingllm_faithful", llm_fn=lambda prompt: "I am not sure.")
    env = make_env("intersection", {"noise_level": "none"})
    env.reset(seed=0)
    arb.reset(seed=1)
    obs = env.observe(np.random.default_rng(0))
    cavs = env.active_cav_vids()
    acts = arb.act(obs, cavs)
    assert all(acts[v] == SLOWER for v in cavs)
    assert arb.parse_failures == 1 + len(cavs)  # negotiation + every decision
    assert arb.extra_metrics()["llm_parse_failures"] == 1 + len(cavs)


def test_memory_retrieve_save_load(tmp_path):
    mem = LLMMemory("local", k=2)
    mem.add("vehicle turning left serious conflict yield to vehicle 5", "SLOWER", FEEDBACK_GOOD)
    mem.add("no conflicts at the moment, free road ahead", "FASTER", FEEDBACK_GOOD)
    mem.add("serious conflict, turning vehicle passes first", "FASTER", FEEDBACK_BAD)
    best = mem.retrieve("free road, no conflicts at the moment", k=1)
    assert best[0]["action"] == "FASTER" and best[0]["feedback"] == FEEDBACK_GOOD
    assert len(mem.retrieve("serious conflict", k=2)) == 2
    assert mem.hits == 3

    path = mem.save(tmp_path / "m.jsonl")
    again = LLMMemory.load(path)
    assert again.embedding == "local" and len(again) == 3
    assert [i["text"] for i in again.items] == [i["text"] for i in mem.items]
    assert again.retrieve("free road, no conflicts at the moment", k=1)[0]["text"] == best[0]["text"]
    with pytest.raises(ValueError):
        LLMMemory.load(path, embedding="openai")  # vectors would not be comparable


def test_frozen_memory_does_not_change(tmp_path):
    mem = LLMMemory("local")
    mem.add("You are vehicle 1, a CAV going straight. Your conflicts: serious.", "SLOWER", FEEDBACK_GOOD)
    mem.add("You have no conflicts at the moment.", "FASTER", FEEDBACK_GOOD)
    path = mem.save(tmp_path / "frozen.jsonl")
    before = path.read_bytes()
    arb = make_arbiter("codrivingllm_faithful", model="mock",
                       memory={"enabled": True, "k": 2, "path": str(path), "learn": False})
    m = run_episode(make_env("intersection", {"noise_level": "high", "duration": 15}), arb, seed=0)
    assert m["memory_items"] == 2 and len(arb.memory) == 2
    assert m["memory_hits"] == 2 * m["llm_calls_decision"]  # k examples per decision
    assert path.read_bytes() == before


def test_learning_memory_collects_feedback():
    arb = make_arbiter("codrivingllm_faithful", model="mock",
                       memory={"enabled": True, "learn": True, "name": "unused_test"})
    m = run_episode(make_env("intersection", {"noise_level": "high", "duration": 15}), arb, seed=100_000)
    assert m["memory_items"] > 0
    assert {i["feedback"] for i in arb.memory.items} <= {FEEDBACK_GOOD, FEEDBACK_BAD,
                                                         w.FEEDBACK_CRASH}
    assert not LLMMemory.path_for("unused_test").exists()  # the arbiter never writes files


def test_missing_memory_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        make_arbiter("codrivingllm_faithful", model="mock",
                     memory={"enabled": True, "name": "does_not_exist_xyz"})
    with pytest.raises(ValueError):
        make_arbiter("codrivingllm", model="mock", memory={"enabled": True, "learn": True})


def test_faithful_mock_episode_metrics():
    env = make_env("intersection", {"noise_level": "high"})
    m = run_episode(env, make_arbiter("codrivingllm_faithful", model="mock"), seed=0)
    for key in ["success", "collision", "avg_speed", "episode_time", "severe_conflicts",
                "llm_calls", "llm_cost_usd", "llm_calls_negotiation", "llm_calls_decision",
                "llm_calls_embedding", "memory_items", "memory_hits"]:
        assert key in m
    assert m["llm_calls_negotiation"] == m["steps"]
    assert m["llm_calls"] == m["llm_calls_negotiation"] + m["llm_calls_decision"]
    assert m["llm_cost_usd"] == 0 and m["llm_parse_failures"] == 0


def test_cache_key_of_simple_mode_unchanged():
    c = w.LLMClient("mock")
    old = hashlib.sha256(f"mock|0.0|{w.SYSTEM_PROMPT}|p".encode("utf-8")).hexdigest()
    assert c._key("p") == old
    assert c._key("p", NEGOTIATION_SYSTEM) != old


def test_memory_examples_change_prompt_and_cache_key():
    mem = LLMMemory("local", k=2)
    mem.add("You are vehicle 1, a CAV going straight. Serious conflict with vehicle 5.",
            "SLOWER", FEEDBACK_GOOD)
    mem.add("You are vehicle 2, a CAV turning left. General conflict with vehicle 3.",
            "FASTER", FEEDBACK_BAD)
    situation = "You are vehicle 1, a CAV going straight. Serious conflict with vehicle 4."
    allowed = [SLOWER, IDLE]
    memories = mem.retrieve(situation)
    assert len(memories) == 2

    zero_shot = decision_prompt(situation, allowed, [])
    two_shot = decision_prompt(situation, allowed, memories)
    assert "Example 1" not in zero_shot
    assert "Example 1" in two_shot and "Example 2" in two_shot
    for m in memories:
        assert m["text"] in two_shot and m["feedback"] in two_shot
    assert two_shot != zero_shot

    c = w.LLMClient("mock")
    assert c._key(two_shot, DECISION_SYSTEM) != c._key(zero_shot, DECISION_SYSTEM)
    # different examples give a different key as well
    one_shot = decision_prompt(situation, allowed, memories[:1])
    assert c._key(one_shot, DECISION_SYSTEM) != c._key(two_shot, DECISION_SYSTEM)


def test_env_file_parser(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# my secrets\n\nOTHER=1\nOPENAI_API_KEY = \"test-not-a-real-key\"\n", encoding="utf-8")
    assert w.read_env_file(f) == {"OTHER": "1", "OPENAI_API_KEY": "test-not-a-real-key"}
    assert w.read_env_file(tmp_path / "missing.env") == {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert w.api_key(f) == "test-not-a-real-key"
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    assert w.api_key(f) == "from-environment"  # the environment wins


def test_no_hard_coded_api_key():
    pattern = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")
    skip = {"venv", ".git", "third_party", "results", "__pycache__", ".pytest_cache"}
    found = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or skip & set(p.relative_to(ROOT).parts) or p.name == ".env":
            continue
        if p.suffix in {".py", ".yaml", ".yml", ".md", ".toml", ".txt", ".example", ".ipynb", ".cfg"}:
            if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
                found.append(str(p))
    assert not found, f"possible API key in {found}"
    assert (ROOT / ".env.example").read_text(encoding="utf-8").strip() == "OPENAI_API_KEY="
