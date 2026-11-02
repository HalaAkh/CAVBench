"""Check the LLM baseline before the full E1 run: API key, cost and speed.

    python experiments/llm_check.py                 # 1 test call + 5 short episodes
    python experiments/llm_check.py --episodes 10
    python experiments/llm_check.py --model mock    # no API key needed

It prints what the episodes cost and how long they took, and estimates the
cost and time of configs/e1_llm.yaml. The replies go into the same cache as
the experiments, so these episodes are not paid for twice.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cavbench.arbiters import make_arbiter  # noqa: E402
from cavbench.arbiters.codrivingllm_wrapper import LLMClient  # noqa: E402
from cavbench.envs import make_env  # noqa: E402
from cavbench.runner import run_episode  # noqa: E402

E1_EPISODES = 3 * 4 * 200  # scenarios x noise levels x seeds in configs/e1_llm.yaml


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--noise", default="high")
    ap.add_argument("--workers", type=int, default=8, help="for the time estimate only")
    a = ap.parse_args(argv)

    client = LLMClient(a.model)
    print(f"1) Test call to {a.model} ...")
    try:
        reply = client("Vehicles:\n- vehicle 1: automated (CAV), speed 8.0 m/s\n"
                       "- vehicle 2: automated (CAV), speed 7.0 m/s\nConflicts:\n"
                       "- vehicle 1 is 20.0 m from the conflict point (TTCP 2.5 s); vehicle 2 "
                       "is 25.0 m from it (TTCP 3.6 s); dTTCP 1.1 s, serious.\n"
                       "Give the passing order of all listed vehicles.")
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print(f"   OK. Reply: {reply.strip()[:200]}")

    print(f"2) {a.episodes} episodes (intersection, noise {a.noise}) ...")
    env = make_env("intersection", {"noise_level": a.noise})
    arb = make_arbiter("codrivingllm", model=a.model, llm_fn=client)
    t0 = time.perf_counter()
    calls = new_calls = cost = 0.0
    for seed in range(a.episodes):
        m = run_episode(env, arb, seed)
        calls += m["llm_calls"]
        new_calls += m["llm_calls"] - m["llm_cache_hits"]
        cost += m["llm_cost_usd"]
        print(f"   seed {seed}: success={m['success']} collision={m['collision']} "
              f"calls={m['llm_calls']} (new {m['llm_calls'] - m['llm_cache_hits']}) "
              f"cost=${m['llm_cost_usd']:.4f} failures={m['llm_failures']}+{m['llm_parse_failures']}")
    wall = time.perf_counter() - t0

    per_call_s = client.api_seconds / max(new_calls, 1)
    per_ep_calls = calls / a.episodes
    per_ep_cost = cost / max(a.episodes, 1)
    if new_calls == 0:
        print("   (all replies came from the cache, so the cost and time below are unknown;"
              " use more --episodes)")
    print(f"3) Per episode: {per_ep_calls:.0f} LLM calls, ${per_ep_cost:.4f}; "
          f"{per_call_s:.2f} s per API call; {wall / a.episodes:.0f} s per episode")
    est_cost = per_ep_cost * E1_EPISODES * (calls / max(new_calls, 1))
    est_hours = per_ep_calls * per_call_s * E1_EPISODES / a.workers / 3600
    print(f"   Estimate for configs/e1_llm.yaml ({E1_EPISODES} episodes, {a.workers} workers): "
          f"about ${est_cost:.0f} and {est_hours:.1f} h.")
    print("   (The dense intersection has more conflicts, so it costs somewhat more per episode.)")


if __name__ == "__main__":
    main()
