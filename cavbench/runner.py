"""Run one episode of a scenario with an arbiter and collect its metrics."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .arbiters import FASTER, Arbiter
from .metrics import episode_metrics


def run_episode(env, arbiter: Arbiter, seed: int, keep_log: bool = False) -> dict:
    """Matched seeds: the same ``seed`` gives the same initial traffic and the
    same sequence of perception-noise samples for every arbiter."""
    env.reset(seed=seed)
    arbiter.reset(seed=seed + 10_000_019)
    noise_rng = np.random.default_rng([seed, 2026])
    done = False
    steps = 0
    while not done:
        obs = env.observe(noise_rng)
        cavs = env.cav_vids()
        acts = arbiter.act(obs, env.active_cav_vids())
        action = tuple(int(acts.get(v, FASTER)) for v in cavs)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    m = episode_metrics(env)
    dt = np.array(arbiter.decision_times) if arbiter.decision_times else np.array([np.nan])
    modes = Counter(entry.get("mode", "commit") for entry in arbiter.log)
    m.update({
        "seed": seed,
        "steps": steps,
        "decision_ms_mean": float(np.nanmean(dt)),
        "decision_ms_p95": float(np.nanpercentile(dt, 95)),
        "decision_ms_max": float(np.nanmax(dt)),
        "n_decisions": int(len(arbiter.decision_times)),
        "n_fallback": int(sum(n for m, n in modes.items() if m.startswith(("fallback", "reject")))),
        "n_cautious": int(modes.get("cautious", 0)),
    })
    if keep_log:
        m["log"] = list(arbiter.log)
    return m
