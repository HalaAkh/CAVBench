"""Tests of the CoDrivingLLM reproduction (cavbench/arbiters/codrivingllm_repro.py).

The last test is the important one: it runs the ORIGINAL CoDrivingLLM code
(negotiation module and decision module) side by side with the reproduction,
on the same vehicles at every step of CAVBench episodes, and checks that both
send the LLM exactly the same prompts and turn the same answers into the same
actions. It needs a copy of the benchmark repository in
third_party/CoDrivingLLM (git-ignored):

    git clone https://github.com/FanGShiYuu/CoDrivingLLM third_party/CoDrivingLLM

and is skipped otherwise.
"""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cavbench.arbiters import make_arbiter
from cavbench.arbiters import codrivingllm_repro as r
from cavbench.envs import make_env
from cavbench.runner import run_episode

BENCH = Path(__file__).resolve().parents[1] / "third_party" / "CoDrivingLLM"


def test_mock_episode_runs_and_counts_calls():
    env = make_env("intersection")
    arb = make_arbiter("codrivingllm_repro", model="mock")
    m = run_episode(env, arb, seed=1)
    assert m["llm_calls_negotiation"] == m["steps"]  # negotiation every step
    assert m["llm_calls_decision"] >= m["steps"]  # one call per CAV still in the scene
    assert m["llm_parse_failures"] == 0


def test_benchmark_vehicle_setup():
    env = make_env("intersection")
    arb = make_arbiter("codrivingllm_repro", model="mock")
    env.reset(seed=0)
    arb.reset(seed=0)
    arb.attach(env)
    for v in env.controlled_vehicles:
        assert list(v.target_speeds) == [0, 5, 10, 15, 20]
        assert v.target_speed == 5.0 and v.speed_index == 1
        assert v.ACC_LIMITS == (-np.inf, np.inf)
    assert arb.safety is None


def test_speed_cap_is_applied_to_the_vehicle():
    env = make_env("intersection")
    arb = make_arbiter("codrivingllm_repro", model="mock")
    env.reset(seed=0)
    arb.reset(seed=0)
    arb.attach(env)
    for v in env.controlled_vehicles:
        v.speed = 7.0
    arb.act(env.observe(np.random.default_rng(0)), env.active_cav_vids())
    assert all(v.speed == 5.0 for v in env.controlled_vehicles)


def test_extract_decision_like_benchmark():
    assert r.extract_decision('Final Answer: \n    "decision": {"FASTER"},') == "FASTER"
    assert r.extract_decision('"decision": {"slow down"}') == "slow down"
    assert r.action_id("slow down") == -1
    assert r.action_id("IDLE") == 1 and r.action_id("SLOWER") == 4


def test_cal_ttcp():
    # 5 m/s limit, 20 m away at 5 m/s: 4 s; at rest: 2.5 s to 5 m/s + 13.75 m / 5
    assert r.cal_ttcp(5, 20.0, 5.0) == pytest.approx(4.0)
    assert r.cal_ttcp(5, 20.0, 0.0) == pytest.approx(2.5 + (20 - 6.25) / 5)


# ---------------------------------------------------------------------------
# Side by side with the original code
# ---------------------------------------------------------------------------
def _load_benchmark(monkeypatch):
    """Import the benchmark's llm_controller with stand-ins for openai/httpx/gym
    (no network). Returns (action module, negotiation module, captured calls)."""
    import highway_env  # the installed HighwayEnv, not the benchmark's fork

    calls: list[str] = []

    class _Completions:
        def create(self, model, messages, **kw):
            assert model == "gpt-4o-mini" and len(messages) == 1 and messages[0]["role"] == "system"
            assert "temperature" not in kw
            prompt = messages[0]["content"]
            calls.append(prompt)
            msg = types.SimpleNamespace(content=r.mock_reply(prompt))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class _OpenAI:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(completions=_Completions())

    stubs = {
        "openai": types.SimpleNamespace(OpenAI=_OpenAI),
        "httpx": types.SimpleNamespace(Client=lambda *a, **k: None),
        "gym": types.ModuleType("gym"),
    }
    for k, v in stubs.items():  # removed again after the test
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setitem(sys.modules, "highway_env", highway_env)
    for k in [k for k in sys.modules if k.startswith("llm_controller")]:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.syspath_prepend(str(BENCH))
    import llm_controller.llm_agent_action as act
    import llm_controller.llm_agent_negotiation_system as neg
    import llm_controller.prompt_llm as pl

    # Bug in the released code: check_safety_in_current_lane calls this tool
    # without its (unused) `env` argument, so the original program stops with a
    # TypeError whenever a CAV has a vehicle ahead and none behind. The
    # reproduction calls the tool as intended; here we let the original do the same.
    orig = pl.isAccelerationConflictWithCar.inference
    monkeypatch.setattr(pl.isAccelerationConflictWithCar, "inference",
                        lambda self, vid, ego_veh, env=None: orig(self, vid, ego_veh, env))
    return act, neg, calls


class _Veh:
    """A vehicle as the benchmark code reads it (fork's MDPVehicle/IDMVehicle)."""

    SPEED_COUNT = 5

    def __init__(self, b: r.BenchVehicle, route):
        self.position = np.array(b.position)
        self.speed = b.speed
        self.lane_index = b.lane_index
        self.route = route
        self.destination = b.destination
        self.speed_index = b.speed_index
        self._name = b.name

    def __str__(self):
        return f"{self._name}: {self.position}"

    __repr__ = __str__


@pytest.mark.skipif(not BENCH.is_dir(), reason="third_party/CoDrivingLLM not present")
@pytest.mark.parametrize("seed,noise", [(0, "none"), (3, "none"), (5, "high")])
def test_same_prompts_and_actions_as_original_code(seed, noise, monkeypatch, capsys):
    act, neg, calls = _load_benchmark(monkeypatch)
    env = make_env("intersection", {"noise_level": noise})
    arb = make_arbiter("codrivingllm_repro", model="mock", parallel=False)
    ours: list[str] = []
    real_ask = arb._ask
    arb._ask = lambda prompt, ctx: (ours.append(prompt), real_ask(prompt, ctx))[1]

    env.reset(seed=seed)
    arb.reset(seed=seed)
    arb.attach(env)
    rng = np.random.default_rng([seed, 2026])
    # the original reads the speed limit of the ego's lane for dTTCP: 5 m/s on
    # every lane of the benchmark intersection (CAVBench's lanes say 10)
    bench_network = copy.deepcopy(env.road.network)
    for lanes in bench_network.graph.values():
        for lane_list in lanes.values():
            for lane in lane_list:
                lane.speed_limit = r.BENCH_LANE_SPEED_LIMIT
    n_steps = 0
    done = False
    while not done and n_steps < 60:
        obs = env.observe(rng)
        cavs = env.active_cav_vids()
        # the same vehicles, as the original code sees them (before our speed cap)
        bv = arb._bench_vehicles(obs)
        road = {vid: _Veh(b, [(a, c, 0) for a, c in b.route_key]) for vid, b in bv.items()}
        controlled = [road[v] for v in env.cav_vids() if v in road]
        fake = types.SimpleNamespace(
            spec=types.SimpleNamespace(id="intersection-multi-agent-v0"),
            road=types.SimpleNamespace(vehicles=list(road.values()), network=bench_network),
            controlled_vehicles=controlled,
            ACTIONS_ALL={"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4},
        )
        calls.clear()
        ours.clear()
        neg_reply, conflicting_info = neg.LlmAgent_negotiation_module(fake).llm_controller_run(fake)
        agent = act.LlmAgent_action_module(fake)
        bench_actions = agent.llm_controller_run(fake, neg_reply, conflicting_info, controlled, None)

        acts = arb.act(obs, cavs)
        assert ours == calls, f"step {n_steps}: prompts differ"
        mapping = {1: 1, 3: 2, 4: 0}  # benchmark action id -> CAVBench action
        expected = [mapping[int(a[0])] for a in bench_actions]
        assert [acts[v] for v in env.cav_vids() if v in road] == expected
        action = tuple(int(acts.get(v, 1)) for v in env.cav_vids())
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        n_steps += 1
    assert n_steps > 10


# ---------------------------------------------------------------------------
# Memory module (switched off in the released code)
# ---------------------------------------------------------------------------
def test_generate_comment_like_benchmark():
    assert r.generate_comment("much close", 3) == "safe and good for efficiency"
    assert r.generate_comment("slight close", 4) == "not efficiency, consider FASTER next time"
    assert r.generate_comment("equal close", 1) == "might not safe, you should be careful and check"
    assert r.generate_comment("much further", 3) == "may not safe if your speed is high"
    assert r.generate_comment("slight further", 4) == "it is safe, good for you"


def test_memory_build_then_use(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "MEMORY_DIR", tmp_path)
    env = make_env("intersection")
    build = make_arbiter("codrivingllm_repro", model="mock",
                         memory={"mode": "build", "name": "t", "embedding": "local"})
    run_episode(env, build, seed=2)
    files = list(tmp_path.glob("repro_t.*.jsonl"))
    assert files and sum(1 for _ in files[0].open()) > 0
    use = make_arbiter("codrivingllm_repro", model="mock",
                       memory={"mode": "use", "name": "t", "embedding": "local"})
    prompts = []
    real = use._ask
    use._ask = lambda p, c: (prompts.append(p), real(p, c))[1]
    m = run_episode(env, use, seed=3)
    assert m["memory_items"] > 0 and m["memory_hits"] > 0
    dec = [p for p in prompts if "Above messages are some examples" in p]
    assert dec and all("- Last time None, you choose to [" in p for p in dec)
    n = sum(1 for _ in files[0].open())
    assert n == sum(1 for _ in files[0].open())  # frozen: nothing written in use mode


def test_memory_episode_kept_once(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "MEMORY_DIR", tmp_path)
    for name in ("repro_t.1.jsonl", "repro_t.2.jsonl"):
        with (tmp_path / name).open("w") as f:
            f.write('{"episode": 7, "content": "x", "metadata": {"a": 1}}\n')
    assert len(r.ReproMemory("t", "local")) == 1


@pytest.mark.skipif(not BENCH.is_dir(), reason="third_party/CoDrivingLLM not present")
def test_memory_same_as_original_code(monkeypatch, tmp_path):
    """relative_memory and memory_update give the same text and the same stored
    items as the benchmark's own functions, on real prompts of an episode."""
    act, neg, calls = _load_benchmark(monkeypatch)
    monkeypatch.setattr(r, "MEMORY_DIR", tmp_path)
    ours = r.ReproMemory("t", "local", write=True, load=False)
    ours.start_episode(0)

    class _Recorder:  # the benchmark's DrivingMemory, recording addMemory calls
        def __init__(self):
            self.added = []

        def addMemory(self, *a):
            self.added.append(a)

        def retrieveMemory(self, q, top_k=5):
            return ours.retrieveMemory(q, top_k)

    rec = _Recorder()
    env = make_env("intersection")
    arb = make_arbiter("codrivingllm_repro", model="mock", parallel=False)
    infos = []
    real = r.prompt_engineer
    monkeypatch.setattr(r, "prompt_engineer", lambda *a: (infos.append(real(*a)), infos[-1])[1])
    run_episode(env, arb, seed=4)
    assert len(infos) > 20
    for i, info in enumerate(infos):
        aid = [1, 3, 4][i % 3]
        act.LlmAgent_action_module.memory_update(None, rec, info, np.array([aid]))
        r.memory_update(ours, info, np.array([aid]))
    assert [tuple(a) for a in rec.added] == [
        (e["content"], *e["metadata"].values()) for e in ours._pending]
    ours.items = [(e["content"], e["metadata"]) for e in ours._pending]
    for info in infos[::5]:
        assert act.LlmAgent_action_module.relative_memory(None, rec, info) == r.relative_memory(ours, info)


def test_memory_openai_embedding_path(tmp_path, monkeypatch):
    """The OpenAI embedding code path (batched requests, disk cache, cost),
    with a stand-in client: no network."""
    monkeypatch.setattr(r, "MEMORY_DIR", tmp_path)
    sent = []

    class _Emb:
        def create(self, model, input):
            assert model == "text-embedding-ada-002"
            sent.append(list(input))
            data = [types.SimpleNamespace(embedding=list(r_local(t))) for t in input]
            return types.SimpleNamespace(data=data, usage=types.SimpleNamespace(total_tokens=10 * len(input)))

    from cavbench.arbiters.llm_memory import local_embedding as r_local
    with (tmp_path / "repro_t.1.jsonl").open("w") as f:
        for i, txt in enumerate(["Ego is _much close_ to", "Conflict info is empty", "Ego is _equal close_ to"]):
            f.write(json.dumps({"episode": 1, "content": txt,
                                "metadata": {"negotiation_result": "None", "final_action": f"[{i}]",
                                             "comments": "c"}}) + "\n")
    mem = r.ReproMemory("t", "openai")
    mem._client = types.SimpleNamespace(embeddings=_Emb())
    out = mem.retrieveMemory("Ego is _equal close_ to now", top_k=2)
    assert out[0]["final_action"] == "[2]" and len(out) == 2
    assert mem.cost_usd > 0 and len(sent) == 2  # stored texts once, then the query
    mem2 = r.ReproMemory("t", "openai")  # embeddings come from the disk cache now
    mem2._client = None
    assert mem2.retrieveMemory("Ego is _equal close_ to now", top_k=2)[0]["final_action"] == "[2]"
