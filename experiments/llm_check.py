"""Check the LLM baseline before the full E1 run: API key, cost and speed.

    python experiments/llm_check.py                    # simple mode: 1 test call + 5 short episodes
    python experiments/llm_check.py --mode faithful    # faithful mode (memory used if built)
    python experiments/llm_check.py --episodes 10
    python experiments/llm_check.py --model mock       # no API key needed (either mode)

It prints what the episodes cost and how long they took, and estimates the
cost and time of configs/e1_llm.yaml (simple) or configs/e1_llm_faithful.yaml
(faithful). The replies go into the same cache as the experiments, so these
episodes are not paid for twice.

The key is read from the OPENAI_API_KEY environment variable or from the
.env file in the repository root (see .env.example).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cavbench.arbiters import make_arbiter  # noqa: E402
from cavbench.arbiters.codrivingllm_wrapper import LLMClient, api_key  # noqa: E402
from cavbench.arbiters.llm_faithful import DECISION_SYSTEM  # noqa: E402
from cavbench.arbiters.llm_memory import LLMMemory  # noqa: E402
from cavbench.envs import make_env  # noqa: E402
from cavbench.runner import run_episode  # noqa: E402

E1_EPISODES = {
    "simple": 3 * 4 * 200,  # configs/e1_llm.yaml: scenarios x noise levels x seeds
    "faithful": 1 * 2 * 50,  # configs/e1_llm_faithful.yaml
}
E1_CONFIG = {"simple": "configs/e1_llm.yaml", "faithful": "configs/e1_llm_faithful.yaml"}

TEST_PROMPT = {
    "simple": ("Vehicles:\n- vehicle 1: automated (CAV), speed 8.0 m/s\n"
               "- vehicle 2: automated (CAV), speed 7.0 m/s\nConflicts:\n"
               "- vehicle 1 is 20.0 m from the conflict point (TTCP 2.5 s); vehicle 2 "
               "is 25.0 m from it (TTCP 3.6 s); dTTCP 1.1 s, serious.\n"
               "Give the passing order of all listed vehicles."),
    "faithful": ("You are vehicle 1, a CAV going straight. Speed 8.0 m/s, target speed 9.0 m/s "
                 "(target speed levels: 0, 4.5, 9 m/s).\nNearby vehicles:\n- vehicle 2: CAV, "
                 "turning, 15.0 m away, speed 7.0 m/s\nYour conflicts:\n- with vehicle 2 (CAV): "
                 "you are 20.0 m from the conflict point (TTCP 2.5 s), it is 25.0 m from it "
                 "(TTCP 3.6 s); dTTCP 1.1 s, serious.\nCoordinator's advice:\n- you pass first "
                 "before vehicle 2.\nAllowed actions (after the safety check): SLOWER, IDLE, "
                 "FASTER.\nChoose your action."),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--mode", default="simple", choices=["simple", "faithful"])
    ap.add_argument("--memory", default="auto",
                    help="faithful mode: memory name, 'none', or 'auto' (intersection_<model> if built)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--noise", default="high")
    ap.add_argument("--workers", type=int, default=8, help="for the time estimate only")
    a = ap.parse_args(argv)

    if a.model != "mock" and not api_key():
        print("OPENAI_API_KEY is not set, so the real API cannot be checked.\n"
              "  * copy .env.example to .env and write your key after OPENAI_API_KEY=\n"
              "    (the .env file is git-ignored; never put the key in code or configs), or\n"
              "  * set the environment variable OPENAI_API_KEY.\n"
              "To test the pipeline for free: python experiments/llm_check.py --model mock "
              f"--mode {a.mode}")
        sys.exit(1)

    params: dict = {"model": a.model, "mode": a.mode}
    if a.mode == "faithful" and a.memory != "none":
        name = f"intersection_{a.model}" if a.memory == "auto" else a.memory
        if LLMMemory.path_for(name).exists():
            params["memory"] = {"enabled": True, "k": 2, "name": name, "learn": False}
            print(f"Memory: {LLMMemory.path_for(name)} (2-shot)")
        elif a.memory == "auto":
            print(f"Memory: not built yet ({LLMMemory.path_for(name).name}); checking the 0-shot "
                  f"version. Build it with: python experiments/build_llm_memory.py --model {a.model}")
        else:
            sys.exit(f"Memory file {LLMMemory.path_for(name)} not found.")

    client = LLMClient(a.model)
    print(f"1) Test call to {a.model} ({a.mode} mode) ...")
    try:
        system = DECISION_SYSTEM if a.mode == "faithful" else None
        reply = client(TEST_PROMPT[a.mode], system=system)
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print(f"   OK. Reply: {reply.strip()[:200]}")

    print(f"2) {a.episodes} episodes (intersection, noise {a.noise}) ...")
    env = make_env("intersection", {"noise_level": a.noise})
    arb = make_arbiter("codrivingllm", llm_fn=client, **params)
    t0 = time.perf_counter()
    calls = new_calls = cost = 0.0
    for seed in range(a.episodes):
        m = run_episode(env, arb, seed)
        calls += m["llm_calls"]
        new_calls += m["llm_calls"] - m["llm_cache_hits"]
        cost += m["llm_cost_usd"]
        extra = ""
        if a.mode == "faithful":
            extra = (f" [negotiation {m['llm_calls_negotiation']}, decision {m['llm_calls_decision']}, "
                     f"embedding {m['llm_calls_embedding']}, memory hits {m['memory_hits']}]")
        print(f"   seed {seed}: success={m['success']} collision={m['collision']} "
              f"calls={m['llm_calls']} (new {m['llm_calls'] - m['llm_cache_hits']}){extra} "
              f"cost=${m['llm_cost_usd']:.4f} failures={m['llm_failures']}+{m['llm_parse_failures']}")
    wall = time.perf_counter() - t0

    n_e1 = E1_EPISODES[a.mode]
    per_call_s = client.api_seconds / max(new_calls, 1)
    per_ep_calls = calls / a.episodes
    per_ep_cost = cost / max(a.episodes, 1)
    if new_calls == 0:
        print("   (all replies came from the cache, so the cost and time below are unknown;"
              " use more --episodes)")
    print(f"3) Per episode: {per_ep_calls:.0f} LLM calls, ${per_ep_cost:.4f}; "
          f"{per_call_s:.2f} s per API call; {wall / a.episodes:.0f} s per episode")
    est_cost = per_ep_cost * n_e1 * (calls / max(new_calls, 1))
    est_hours = per_ep_calls * per_call_s * n_e1 / a.workers / 3600
    print(f"   Estimate for {E1_CONFIG[a.mode]} ({n_e1} episodes, {a.workers} workers): "
          f"about ${est_cost:.0f} and {est_hours:.1f} h.")
    if a.mode == "simple":
        print("   (The dense intersection has more conflicts, so it costs somewhat more per episode.)")
    else:
        print("   (Faithful mode calls the LLM at every step: about 1 + number of CAVs calls per 0.2 s.)")


if __name__ == "__main__":
    main()
