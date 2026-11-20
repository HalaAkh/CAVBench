"""All quantitative results of the paper in one report.

    python experiments/final_report.py            # reads results/*.csv -> results/final_report.md

Sections (each is skipped if its results file is missing):
  1. E1 main comparison: every method, per scenario, with 95 % confidence intervals
  2. E1 paired tests: UCRA against every other method on the same episodes
  3. CoDrivingLLM Table I format (success, PET, travel velocity)
  4. Oracle gap: how close UCRA comes to perfect information
  5. Robustness: noise level (E1), V2X channel (E6), traffic density
  6. Ablation: uncertainty U (E2), fallback (E3), age-aware sigma (E6)
  7. Sensitivity: number of rollouts (E4), parameters one at a time (E7),
     wrong noise assumption (E7b)
  8. Real-time: decision time (E5)

Statistics: proportions with Wilson 95 % intervals; means with t-based 95 %
intervals; paired comparisons on matched episodes (same scenario, noise and
seed): exact McNemar test for yes/no outcomes, Wilcoxon signed-rank test for
episode time (on episodes both methods completed).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(x: pd.Series) -> str:
    x = x.astype(bool)
    k, n = int(x.sum()), len(x)
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]" if n else "-"


def mean_ci(x: pd.Series, digits: int = 1) -> str:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return f"{x.mean():.{digits}f}" if len(x) else "-"
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f"{x.mean():.{digits}f} ± {h:.{digits}f}"


def mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    a, b = a.astype(bool), b.astype(bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def load(name: str) -> pd.DataFrame | None:
    p = RES / f"{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if "success" in d:
        d["timeout"] = ~d["success"].astype(bool) & ~d["collision"].astype(bool)
    return d


def summary(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row.update({
            "Episodes": len(g),
            "Success % (all arrived) [95% CI]": pct(g["success"]),
            "Collision-free % [95% CI]": pct(~g["collision"].astype(bool)),
            "Collisions": int(g["collision"].sum()),
            "Time-outs": int(g["timeout"].sum()),
            "Severe-conflict episodes (PET<1.5 s)": int(g["any_severe_conflict"].sum()),
            "Episode time [s]": mean_ci(g["episode_time"]),
            "CAV speed [m/s]": mean_ci(g["avg_speed"], 2),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame, ref: str, other: str, keys=("scenario", "noise", "seed")) -> dict | None:
    a = d[d["arbiter"] == ref].set_index(list(keys))
    b = d[d["arbiter"] == other].set_index(list(keys))
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return None
    a, b = a.loc[idx], b.loc[idx]
    out = {"First": ref, "Second": other, "Matched episodes": len(idx)}
    for col, name in [("success", "Success"), ("collision", "Collision"), ("any_severe_conflict", "Severe conflict")]:
        x, y, p = mcnemar(a[col], b[col])
        out[f"{name}: first only / second only (p)"] = f"{x} / {y} ({fmt_p(p)})"
    both = a["success"].astype(bool) & b["success"].astype(bool)
    if both.sum() > 5:
        ta, tb = a.loc[both, "episode_time"], b.loc[both, "episode_time"]
        p = stats.wilcoxon(ta, tb).pvalue if (ta != tb).any() else 1.0
        out["Episode time, both completed [s]"] = f"{ta.mean():.1f} vs {tb.mean():.1f} ({100 * (ta.mean() / tb.mean() - 1):+.0f} %, p {fmt_p(p)})"
    return out


# ---------------------------------------------------------------- sections
def s_e1(out: list[str], e1: pd.DataFrame) -> None:
    out.append("## 1. E1: UCRA against the baselines\n")
    out.append("All noise levels pooled. Same seeds for every method (the oracle and "
               "CoDrivingLLM were run on fewer noise levels, see the episode counts).\n")
    out.append(md(summary(e1, ["scenario", "arbiter"])) + "\n")
    out.append("### Noise level none only (the CoDrivingLLM paper's setting)\n")
    out.append(md(summary(e1[e1["noise"] == "none"], ["scenario", "arbiter"])) + "\n")
    out.append("## 2. E1 paired tests: UCRA against each method on the same episodes\n")
    rows = [r for o in e1["arbiter"].unique() if o != "UCRA" for r in [paired(e1, "UCRA", o)] if r]
    out.append(md(pd.DataFrame(rows)) + "\n")
    out.append("\"first only / second only\" = episodes where only the first (or only the second) method "
               "had that outcome. McNemar exact test on those episodes; "
               "Wilcoxon signed-rank test on episode time.\n")
    if {"CoDrivingLLM", "CoDrivingLLM-memory"} <= set(e1["arbiter"]):
        out.append("### CoDrivingLLM with its memory module (2-shot) against as released (0-shot)\n")
        r = paired(e1, "CoDrivingLLM-memory", "CoDrivingLLM")
        out.append(md(pd.DataFrame([r])) + "\n")
        out.append("Reported by Fang et al. (2025) for the intersection: 40 % with no interactions, 75 % after "
                   "20+ interactions (their Fig. 7), 90 % in their Table I (2-shot, 20 episodes).\n")


def s_table1(out: list[str], e1: pd.DataFrame) -> None:
    if "bpet_n" not in e1:
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("bt", ROOT / "experiments" / "benchmark_table.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    out.append("## 3. CoDrivingLLM Table I format (intersection, noise none)\n")
    t = bt.table(e1[(e1["scenario"] == "intersection") & (e1["noise"] == "none")])
    out.append(md(t.round(1)) + "\n")
    paper = pd.DataFrame(bt.PAPER, columns=["Method", bt.COLS[0], bt.COLS[2], bt.COLS[3], bt.COLS[4],
                                             bt.COLS[5], bt.COLS[6], bt.COLS[7]])
    out.append("Reported by Fang et al. (2025), Table I (their simulator, 20 episodes):\n")
    out.append(md(paper) + "\n")


def s_oracle(out: list[str], e1: pd.DataFrame) -> None:
    if "Oracle" not in set(e1["arbiter"]):
        return
    out.append("## 4. Oracle gap (noise none, episodes completed by both)\n")
    d = e1[e1["noise"] == "none"]
    rows = []
    for sc in d["scenario"].unique():
        for m in [x for x in d["arbiter"].unique() if x not in ("Oracle", "NoCoord")]:
            a = d[(d.scenario == sc) & (d.arbiter == m)].set_index("seed")
            o = d[(d.scenario == sc) & (d.arbiter == "Oracle")].set_index("seed")
            idx = a.index.intersection(o.index)
            if not len(idx):
                continue
            both = a.loc[idx, "success"].astype(bool) & o.loc[idx, "success"].astype(bool)
            if both.sum() < 5:
                continue
            ta, to = a.loc[idx][both]["episode_time"].mean(), o.loc[idx][both]["episode_time"].mean()
            ca, co, p = mcnemar(a.loc[idx, "collision"], o.loc[idx, "collision"])
            rows.append({"Scenario": sc, "Method": m, "Episode time [s]": f"{ta:.1f}",
                         "Oracle time [s]": f"{to:.1f}", "Efficiency vs oracle": f"{100 * to / ta:.0f} %",
                         "Collisions: method only / oracle only (p)": f"{ca} / {co} ({fmt_p(p)})"})
    out.append(md(pd.DataFrame(rows)) + "\n")
    out.append("Efficiency = oracle's episode time / method's episode time (100 % = as fast as "
               "perfect information).\n")


def s_robust(out: list[str], e1: pd.DataFrame, e6: pd.DataFrame | None) -> None:
    out.append("## 5. Robustness\n")
    out.append("### Perception noise (E1)\n")
    keep = [m for m in ["UCRA", "FCFS", "MCTS", "CoDrivingLLM", "CoDrivingLLM-memory", "Oracle"] if m in set(e1["arbiter"])]
    d = e1[e1["arbiter"].isin(keep)]
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    s = summary(d.sort_values("noise", key=lambda c: c.map(order)), ["arbiter", "noise"])
    out.append(md(s) + "\n")
    rows = []
    for m in keep:
        g = d[d.arbiter == m]
        if g["noise"].nunique() > 1:
            per = g.groupby("noise")["success"].mean()
            rows.append({"Method": m, "Success range across noise levels": f"{100 * per.min():.1f} - {100 * per.max():.1f} %",
                         "Drop none -> high [points]": f"{100 * (per.get('none', np.nan) - per.get('high', np.nan)):+.1f}"})
    if rows:
        out.append(md(pd.DataFrame(rows)) + "\n")
    if e6 is not None:
        out.append("### V2X channel: delay and packet loss (E6)\n")
        e6 = e6.copy()
        e6["v2x"] = e6["scenario"].str.extract(r"v2x_(\w+)")[0]
        e6["scene"] = e6["scenario"].str.replace(r"_v2x_\w+", "", regex=True)
        e6 = e6[e6["arbiter"].isin(["UCRA", "FCFS"])]
        out.append(md(summary(e6, ["arbiter", "v2x"])) + "\n")
    out.append("### Traffic density (E1, all noise levels)\n")
    out.append(md(summary(d[d.scenario.isin(["intersection", "intersection_dense"])], ["arbiter", "scenario"])) + "\n")


def ablation_block(out: list[str], d: pd.DataFrame, full: str, cut: str, title: str, keys=("scenario", "noise", "seed")) -> None:
    out.append(f"### {title}\n")
    out.append(md(summary(d, ["arbiter"])) + "\n")
    r = paired(d, full, cut, keys)
    if r:
        out.append(md(pd.DataFrame([r])) + "\n")


def s_ablation(out: list[str], e2, e3, e6) -> None:
    out.append("## 6. Ablation (one component removed, same episodes)\n")
    if e2 is not None:
        ablation_block(out, e2, "UCRA", "UCRA-noU", "Uncertainty term U (E2)")
    if e3 is not None:
        ablation_block(out, e3, "UCRA", "UCRA-noFallback", "Fallback option (E3)")
    if e6 is not None:
        d = e6[e6["arbiter"].isin(["UCRA", "UCRA-noAge"])].copy()
        d["v2x"] = d["scenario"].str.extract(r"v2x_(\w+)")[0]
        d = d[d["v2x"] != "ideal"]
        ablation_block(out, d, "UCRA", "UCRA-noAge", "Age-aware sigma under delayed V2X (E6, typical + degraded)")


def s_sensitivity(out: list[str], e4, e7, e7b) -> None:
    out.append("## 7. Sensitivity analysis\n")
    if e4 is not None:
        out.append("### Number of rollouts N: accuracy of the risk estimate (E4, Theorem 1)\n")
        rows = []
        for n, g in e4.groupby("N"):
            bound = min(1.0, 2 * 4 * math.exp(-2 * n * 0.15 ** 2))
            rows.append({"N": n, "Mean |error|": f"{g.max_abs_error.mean():.3f}",
                         "P(|error| >= 0.15)": f"{100 * (g.max_abs_error >= 0.15).mean():.2f} %",
                         "Theorem 1 bound (K = 4)": f"{100 * bound:.1f} %", "Max error": f"{g.max_abs_error.max():.3f}"})
        out.append(md(pd.DataFrame(rows)) + "\n")
    if e7 is not None:
        out.append("### Parameters one at a time (E7)\n")
        base = e7[e7["arbiter"] == "UCRA (default)"]
        b_s, b_c, b_t = base["success"].mean(), base["collision"].sum(), base["episode_time"].mean()
        rows = []
        for m, g in e7.groupby("arbiter", sort=False):
            r = paired(e7, "UCRA (default)", m) if m != "UCRA (default)" else None
            rows.append({"Variant": m, "Episodes": len(g), "Success %": f"{100 * g.success.mean():.1f}",
                         "Δ success [points]": "" if m == "UCRA (default)" else f"{100 * (g.success.mean() - b_s):+.1f}",
                         "Collisions": int(g.collision.sum()), "Δ collisions": "" if m == "UCRA (default)" else f"{int(g.collision.sum()) - b_c:+d}",
                         "Time-outs": int(g.timeout.sum()), "Episode time [s]": f"{g.episode_time.mean():.1f}",
                         "Δ time": "" if m == "UCRA (default)" else f"{100 * (g.episode_time.mean() / b_t - 1):+.0f} %",
                         "Success differs from default (p)": "" if not r else r["Success: first only / second only (p)"]})
        out.append(md(pd.DataFrame(rows)) + "\n")
    if e7b is not None:
        out.append("### Wrong noise assumption (E7b)\n")
        d = e7b.copy()
        d["scene"] = d["scenario"].str.split(" | ", regex=False).str[0]
        d["assumption"] = d["scenario"].str.split(" | ", regex=False).str[1]
        d["true noise"] = d["noise"]
        out.append(md(summary(d, ["scene", "true noise", "assumption"])) + "\n")
        out.append("Assumes low while the true noise is high = over-confident (more missed dangers, "
                   "i.e. false negatives); assumes high while there is no noise = over-cautious "
                   "(more waiting, i.e. false alarms).\n")


def s_realtime(out: list[str], e5) -> None:
    if e5 is None:
        return
    out.append("## 8. Real-time performance (E5, one worker)\n")
    rows = []
    for sc, g in e5.groupby("scenario"):
        rows.append({"Scenario": sc, "Episodes": len(g), "Mean decision [ms]": mean_ci(g["decision_ms_mean"]),
                     "p95 decision [ms] (mean over episodes)": f"{g['decision_ms_p95'].mean():.0f}",
                     "Episodes with p95 > 200 ms": int((g["decision_ms_p95"] > 200).sum()),
                     "Slowest decision [ms]": f"{g['decision_ms_max'].max():.0f}"})
    out.append(md(pd.DataFrame(rows)) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RES / "final_report.md")
    a = ap.parse_args(argv)
    e1, e2, e3, e4 = load("e1_baselines"), load("e2_uncertainty_ablation"), load("e3_fallback_ablation"), load("e4_rollout_sensitivity")
    e5, e6, e7, e7b = load("e5_realtime"), load("e6_v2x_channel"), load("e7_sensitivity"), load("e7_noise_mismatch")
    out = ["# CAVBench: final results\n"]
    if e1 is not None:
        s_e1(out, e1); s_table1(out, e1); s_oracle(out, e1); s_robust(out, e1, e6)
    s_ablation(out, e2, e3, e6)
    s_sensitivity(out, e4, e7, e7b)
    s_realtime(out, e5)
    text = "\n".join(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
