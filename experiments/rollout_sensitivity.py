"""Experiment E4: empirical check of Theorem 1 (risk-estimation bound).

1. Decision situations (a noisy scene and a conflict group) are collected from
   episodes driven by the proposed arbiter.
2. For each situation, a reference risk R_k is estimated for every candidate
   maneuver with a large number of rollouts (default 10,000).
3. For N in {25, 50, 100, 200}, the risk is re-estimated `repeats` times with
   fresh random numbers, and the maximum error over the candidates,
   max_k |R_hat_k - R_k|, is recorded.

The output CSV lets the analysis notebook compare the empirical probability
P(max error >= gamma) with Theorem 1's bound 2 K exp(-2 N gamma^2).

Usage:
    python experiments/rollout_sensitivity.py configs/e4_rollout_sensitivity.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cavbench.arbiters import FASTER, make_arbiter  # noqa: E402
from cavbench.arbiters.whatif_core import RolloutConfig, evaluate_options  # noqa: E402
from cavbench.envs import make_env  # noqa: E402


def collect_situations(scenario, env_cfg, noise, n_situations, seed0, every):
    """Run the proposed arbiter and keep every `every`-th decision situation."""
    env = make_env(scenario, {**env_cfg, "noise_level": noise})
    arb = make_arbiter("whatif_fuzzy")
    out, seed, count = [], seed0, 0
    while len(out) < n_situations:
        env.reset(seed=seed)
        arb.reset(seed=seed)
        rng = np.random.default_rng([seed, 2026])
        done = False
        while not done and len(out) < n_situations:
            obs = env.observe(rng)
            for g in arb.detector.groups(obs):
                if len(g.cavs) >= 2:
                    count += 1
                    if count % every == 0:
                        out.append((copy.deepcopy(obs), copy.deepcopy(g), seed))
            cavs = env.cav_vids()
            acts = arb.act(obs, env.active_cav_vids())
            _, _, term, trunc, _ = env.step(tuple(int(acts.get(v, FASTER)) for v in cavs))
            done = term or trunc
        seed += 1
    return out


def risk_estimate(obs, group, n, rng, chunk=1000):
    """Risk of every candidate with n rollouts (in chunks to bound memory)."""
    hits, total = None, 0
    while total < n:
        m = min(chunk, n - total)
        res = evaluate_options(obs, group, RolloutConfig(n_rollouts=m), rng)
        h = np.array([r.risk for r in res]) * m
        hits = h if hits is None else hits + h
        total += m
    return hits / total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(args.config.read_text())
    out = args.out or ROOT / "results" / f"{cfg['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.random.default_rng(cfg.get("seed", 7))
    for noise in cfg["noise_levels"]:
        sits = collect_situations(cfg["scenario"], cfg.get("env", {}), noise,
                                  cfg["situations"], cfg.get("seed_offset", 0), cfg.get("every", 5))
        print(f"{noise}: {len(sits)} situations collected", flush=True)
        for si, (obs, group, seed) in enumerate(sits):
            ref = risk_estimate(obs, group, cfg["reference_rollouts"], rng)
            for n in cfg["n_values"]:
                for rep in range(cfg["repeats"]):
                    est = risk_estimate(obs, group, n, rng)
                    rows.append({
                        "noise": noise, "situation": si, "episode_seed": seed,
                        "K": len(group.cavs), "N": n, "repeat": rep,
                        "max_abs_error": float(np.max(np.abs(est - ref))),
                        "reference_risk_max": float(ref.max()),
                        "reference_risk_min": float(ref.min()),
                    })
            print(f"  situation {si + 1}/{len(sits)} done", flush=True)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
