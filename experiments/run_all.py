"""Run every experiment of the paper, in order, with one command.

    python experiments/run_all.py              # E4, E5, E2, E3, E1 (without the LLM), E6
    python experiments/run_all.py --with-llm   # also the LLM baseline of E1 (API key)
    python experiments/run_all.py --episodes 20   # quick trial with fewer seeds

Each step resumes where it stopped, so the command can simply be run again
after an interruption. Results go to results/*.csv; then open
analysis/statistics.ipynb to get the tables, tests and figures.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-llm", action="store_true", help="also run configs/e1_llm.yaml")
    ap.add_argument("--episodes", type=int, help="override the number of seeds (for a trial)")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1))
    ap.add_argument("--llm-workers", type=int, default=8)
    a = ap.parse_args(argv)

    run = [sys.executable, str(ROOT / "experiments" / "run_experiment.py")]
    ep = ["--episodes", str(a.episodes)] if a.episodes else []
    steps = [
        ("E4 rollout sensitivity (Theorem 1)",
         [sys.executable, str(ROOT / "experiments" / "rollout_sensitivity.py"),
          str(ROOT / "configs" / "e4_rollout_sensitivity.yaml")]),
        ("E5 decision time (H4), one worker so timings are not disturbed",
         run + [str(ROOT / "configs" / "e5_realtime.yaml"), "--workers", "1"] + ep),
        ("E2 uncertainty ablation (H2)",
         run + [str(ROOT / "configs" / "e2_uncertainty_ablation.yaml"), "--workers", str(a.workers)] + ep),
        ("E3 fallback ablation (H3)",
         run + [str(ROOT / "configs" / "e3_fallback_ablation.yaml"), "--workers", str(a.workers)] + ep),
        ("E1 baselines (H1): UCRA, FCFS, no coordination",
         run + [str(ROOT / "configs" / "e1_baselines.yaml"), "--workers", str(a.workers)] + ep),
        ("E6 V2X channel (H5): message delay and packet loss",
         run + [str(ROOT / "configs" / "e6_v2x_channel.yaml"), "--workers", str(a.workers)] + ep),
    ]
    if a.with_llm:
        steps.append(("E1 LLM baseline (CoDrivingLLM-style, gpt-4o-mini)",
                      run + [str(ROOT / "configs" / "e1_llm.yaml"),
                             "--workers", str(a.llm_workers)] + ep))

    t_all = time.time()
    for i, (title, cmd) in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps)}] {title} ===", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"Step failed (exit code {r.returncode}). Fix the problem and run the same "
                  f"command again: finished episodes are kept.")
            sys.exit(r.returncode)
        print(f"--- done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nAll done in {(time.time() - t_all) / 3600:.1f} h. "
          f"Now open analysis/statistics.ipynb.")
    if not a.with_llm:
        print("The LLM baseline was not run: add your OpenAI key, check with "
              "'python experiments/llm_check.py', then run 'python experiments/run_all.py --with-llm' "
              "(the finished steps are skipped).")


if __name__ == "__main__":
    main()
