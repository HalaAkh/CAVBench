"""Checks before (and between) the two CoDrivingLLM memory runs. Cost < $0.001.

    python experiments/check_codrivingllm_memory.py          # API: one chat call + one embedding call
    python experiments/check_codrivingllm_memory.py --memory # after the build: what the memory holds

Run the first form BEFORE the build: it stops with a clear message if the key,
the credit, gpt-4o-mini or text-embedding-ada-002 is not available, so that no
paid run fails half-way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cavbench.arbiters import codrivingllm_repro as r  # noqa: E402
from cavbench.arbiters.codrivingllm_wrapper import require_api_key  # noqa: E402


def check_api() -> None:
    require_api_key()
    from openai import OpenAI
    c = OpenAI(timeout=60.0)
    reply = c.chat.completions.create(model="gpt-4o-mini",
                                      messages=[{"role": "system", "content": "Answer with the word OK."}])
    print("gpt-4o-mini:", (reply.choices[0].message.content or "").strip()[:20], "(chat works)")
    e = c.embeddings.create(model=r.EMBED_MODEL, input=["Conflict info is empty"])
    print(f"{r.EMBED_MODEL}: vector of {len(e.data[0].embedding)} numbers (embeddings work)")
    print("All good: you can run configs/e1_codrivingllm_memory_build.yaml")


def check_memory(name: str) -> None:
    files = sorted(r.MEMORY_DIR.glob(f"repro_{name}.*.jsonl"))
    episodes, items = set(), 0
    for p in files:
        for line in p.open(encoding="utf-8"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            episodes.add(e.get("episode"))
            items += 1
    mem = r.ReproMemory(name, "local")
    print(f"memory '{name}': {len(files)} file(s), {len(episodes)} episode(s), {len(mem)} items")
    if len(episodes) < 20:
        print("Fewer than 20 episodes: run the build config again (finished episodes are skipped).")
    else:
        print("Complete: you can run configs/e1_codrivingllm_memory.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--name", default="intersection")
    a = ap.parse_args()
    check_memory(a.name) if a.memory else check_api()
