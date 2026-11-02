"""Run a CAVBench experiment from a YAML configuration.

Every combination of scenario x arbiter x noise level is run on the same
seeds (matched seeds: the same initial traffic and the same perception-noise
samples for every arbiter). One CSV row is written per episode, so an
interrupted run can be resumed: rows already in the output file are skipped.

Usage:
    python experiments/run_experiment.py configs/e1_baselines.yaml
    python experiments/run_experiment.py configs/e1_baselines.yaml --episodes 20 --workers 4
    python experiments/run_experiment.py configs/smoke.yaml --out results/smoke.csv

Config format (see configs/):
    name: e1_baselines
    episodes: 200            # seeds 0 .. episodes-1 (plus seed_offset)
    seed_offset: 0
    noise_levels: [none, low, medium, high]
    scenarios:
      - {label: intersection, scenario: intersection, env: {}}
    arbiters:
      - {label: UCRA, kind: whatif_fuzzy, params: {}}
      - {label: FCFS, kind: fcfs}
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cavbench.arbiters import make_arbiter  # noqa: E402
from cavbench.envs import make_env  # noqa: E402
from cavbench.runner import run_episode  # noqa: E402

KEY_COLUMNS = ["experiment", "scenario", "arbiter", "noise", "seed"]
METRIC_COLUMNS = [
    "success", "collision", "all_arrived", "n_pet_pairs", "severe_conflicts",
    "any_severe_conflict", "min_pet", "avg_speed", "episode_time", "steps",
    "decision_ms_mean", "decision_ms_p95", "decision_ms_max", "n_decisions",
    "n_fallback", "n_cautious",
    # LLM baseline only (0 for the other arbiters):
    "llm_calls", "llm_cache_hits", "llm_tokens_in", "llm_tokens_out", "llm_cost_usd",
    "llm_failures", "llm_parse_failures",
]
COLUMNS = KEY_COLUMNS + ["kind"] + METRIC_COLUMNS + ["wall_s"]

_CACHE: dict = {}  # per worker process: (scenario label, arbiter label, noise) -> (env, arbiter)


def _get(task):
    key = (task["scenario_label"], task["arbiter_label"], task["noise"])
    if key not in _CACHE:
        env_cfg = dict(task["env"])
        env_cfg["noise_level"] = task["noise"]
        env = make_env(task["scenario"], env_cfg)
        arb = make_arbiter(task["kind"], **task["params"])
        _CACHE[key] = (env, arb)
    return _CACHE[key]


def _run(task):
    warnings.filterwarnings("ignore")
    env, arb = _get(task)
    t0 = time.perf_counter()
    m = run_episode(env, arb, task["seed"])
    row = {
        "experiment": task["experiment"], "scenario": task["scenario_label"],
        "arbiter": task["arbiter_label"], "noise": task["noise"], "seed": task["seed"],
        "kind": task["kind"], "wall_s": round(time.perf_counter() - t0, 3),
    }
    row.update({k: m.get(k, 0) for k in METRIC_COLUMNS})
    return row


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if "scenarios" not in cfg:  # shorthand: a single scenario
        cfg["scenarios"] = [{"label": cfg.get("scenario", "intersection"),
                             "scenario": cfg.get("scenario", "intersection"),
                             "env": cfg.get("env", {})}]
    return cfg


def build_tasks(cfg: dict, episodes: int, done: set) -> list[dict]:
    seeds = [cfg.get("seed_offset", 0) + i for i in range(episodes)]
    tasks = []
    # Seed-major order: partial results are balanced across all conditions.
    for seed in seeds:
        for sc in cfg["scenarios"]:
            for noise in cfg["noise_levels"]:
                for arb in cfg["arbiters"]:
                    key = (cfg["name"], sc["label"], arb["label"], noise, str(seed))
                    if key in done:
                        continue
                    tasks.append({
                        "experiment": cfg["name"], "scenario_label": sc["label"],
                        "scenario": sc.get("scenario", sc["label"]), "env": sc.get("env") or {},
                        "arbiter_label": arb["label"], "kind": arb["kind"],
                        "params": arb.get("params") or {}, "noise": noise, "seed": seed,
                    })
    return tasks


def read_done(out: Path) -> set:
    if not out.exists():
        return set()
    with out.open() as f:
        return {tuple(r[c] for c in KEY_COLUMNS) for r in csv.DictReader(f)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--episodes", type=int, help="override the number of episodes (seeds)")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1),
                    help="parallel processes (use 1 when measuring decision time, E5)")
    ap.add_argument("--out", type=Path, help="CSV file (default: results/<name>.csv)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    episodes = args.episodes or cfg.get("episodes", 200)
    out = args.out or ROOT / "results" / f"{cfg['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = read_done(out)
    tasks = build_tasks(cfg, episodes, done)
    total = len(tasks) + len(done)
    print(f"{cfg['name']}: {len(tasks)} episodes to run ({len(done)} already in {out}), "
          f"{args.workers} worker(s)")
    if not tasks:
        return

    new_file = not out.exists()
    if not new_file:
        with out.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise SystemExit(f"{out} was written by an older version of CAVBench (different "
                             f"columns). Rename or delete it and run again.")
    t0 = time.time()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        if args.workers == 1:
            results = map(_run, tasks)
            pool = None
        else:
            pool = Pool(args.workers)
            results = pool.imap_unordered(_run, tasks, chunksize=1)
        try:
            for i, row in enumerate(results, 1):
                writer.writerow(row)
                f.flush()
                if i % 10 == 0 or i == len(tasks):
                    rate = i / (time.time() - t0)
                    left = (len(tasks) - i) / max(rate, 1e-9)
                    print(f"  {len(done) + i}/{total} episodes, ~{left / 60:.1f} min left", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
