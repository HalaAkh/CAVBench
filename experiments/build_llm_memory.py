"""Build the memory of the faithful CoDrivingLLM baseline on TRAINING seeds.

Fair, reproducible protocol: the memory is built here, on separate seeds
(100000 and up, never used by the evaluation configs), with learning switched
on. It is then FROZEN: evaluation runs (e.g. configs/e1_llm_faithful.yaml)
load the file with learn=False, so evaluation episodes stay independent of
each other and matched with the other methods.

    python experiments/build_llm_memory.py --model mock          # free, local embeddings
    python experiments/build_llm_memory.py                       # gpt-4o-mini (needs OPENAI_API_KEY)
    python experiments/build_llm_memory.py --episodes 40 --overwrite

The file goes to results/llm_memory/<name>.jsonl (default name:
intersection_<model>), which is git-ignored. The memory is saved after every
episode, so an interrupted build keeps what it learned (re-run with --resume).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cavbench.arbiters import make_arbiter  # noqa: E402
from cavbench.arbiters.codrivingllm_wrapper import api_key  # noqa: E402
from cavbench.arbiters.llm_memory import FEEDBACK_GOOD, LLMMemory  # noqa: E402
from cavbench.envs import make_env  # noqa: E402
from cavbench.runner import run_episode  # noqa: E402

TRAIN_SEED_START = 100_000  # evaluation configs use seeds 0 .. 199


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=TRAIN_SEED_START)
    ap.add_argument("--noise", default="none,high", help="noise levels, used in turn")
    ap.add_argument("--scenario", default="intersection")
    ap.add_argument("--name", default=None, help="memory name (default: <scenario>_<model>)")
    ap.add_argument("--embedding", default="auto", choices=["auto", "local", "openai"])
    ap.add_argument("--k", type=int, default=2, help="examples retrieved per decision while building")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", help="start again from an empty memory")
    group.add_argument("--resume", action="store_true", help="add to an existing memory file")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    if a.seed_start < 1000:
        raise SystemExit("Training seeds must not overlap the evaluation seeds (0-199): "
                         "use --seed-start >= 1000.")
    if a.model != "mock" and not api_key():
        raise SystemExit("OPENAI_API_KEY is not set: put OPENAI_API_KEY=sk-... in the .env file "
                         "(see .env.example), or use --model mock to test for free.")
    name = a.name or f"{a.scenario}_{a.model}"
    path = LLMMemory.path_for(name)
    if path.exists() and not (a.overwrite or a.resume):
        raise SystemExit(f"{path} already exists. Use --resume to add to it or --overwrite "
                         f"to start again.")
    if path.exists() and a.overwrite:
        path.unlink()

    arb = make_arbiter("codrivingllm_faithful", model=a.model, memory={
        "enabled": True, "learn": True, "name": name, "k": a.k, "embedding": a.embedding})
    mem = arb.memory
    noises = [n.strip() for n in a.noise.split(",") if n.strip()]
    envs = {n: make_env(a.scenario, {"noise_level": n}) for n in noises}
    print(f"Building memory '{name}' ({mem.embedding} embeddings, {len(mem)} items so far) "
          f"on seeds {a.seed_start}..{a.seed_start + a.episodes - 1}")
    t0, cost = time.time(), 0.0
    for i in range(a.episodes):
        seed = a.seed_start + i
        noise = noises[i % len(noises)]
        before = len(mem)
        m = run_episode(envs[noise], arb, seed)
        new = mem.items[before:]
        good = sum(it["feedback"] == FEEDBACK_GOOD for it in new)
        cost += m.get("llm_cost_usd", 0.0)
        print(f"  seed {seed} ({noise}): success={m['success']} collision={m['collision']} "
              f"+{len(new)} items ({good} positive, {len(new) - good} negative), "
              f"{m.get('llm_calls', 0)} calls, ${m.get('llm_cost_usd', 0.0):.4f}", flush=True)
        mem.save(path)
    print(f"Saved {len(mem)} items -> {path}  ({(time.time() - t0) / 60:.1f} min, ${cost:.2f})")


if __name__ == "__main__":
    main()
