"""Figures for the paper, one figure per image, from the CSV files in results/.

    python experiments/make_figures.py

Writes PNG files to results/figures/, in four groups:
  1_main_*        all methods compared (lower bound, classic, latest, ours, upper bound)
  2_ablation_*    one figure per removed component of UCRA
  3_sensitivity_* one figure per parameter, plus the rollout accuracy (Theorem 1)
  4_robustness_*  one figure per change of conditions

The main comparison uses MATCHED episodes: the intersection episodes (noise level,
seed) that every compared method ran, so all bars rest on the same episodes.
Success = no collision and all CAVs arrived; error bars = 95 % Wilson intervals.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"

# role of each method, in display order (left to right / top to bottom)
METHODS = [
    ("NoCoord", "No coordination", "lower bound", "#8a939b"),
    ("FCFS", "FCFS", "classic", "#4f6fb0"),
    ("CoDrivingLLM", "CoDrivingLLM", "latest (benchmark)", "#c8372d"),
    ("CoDrivingLLM-memory", "CoDrivingLLM\n+ memory", "latest (benchmark)", "#e3867b"),
    ("MCTS", "MCTS", "latest", "#e0952a"),
    ("UCRA", "UCRA", "ours", "#0f8f7e"),
    ("Oracle", "Oracle", "upper bound", "#2e7d4f"),
]
LABEL = {m: l for m, l, _, _ in METHODS}
ROLE = {m: r for m, _, r, _ in METHODS}
COL = {m: c for m, _, _, c in METHODS}
UCRA_C, ABL_C = "#0f8f7e", "#c8372d"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "legend.frameon": False, "figure.dpi": 100})


# ----------------------------------------------------------------- helpers
def load(name):
    p = RES / f"{name}.csv"
    if not p.exists():
        print(f"  skipped: {p.name} not found")
        return None
    return pd.read_csv(p)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(s):
    """percentage with Wilson half-widths (lower, upper)"""
    s = s.astype(bool)
    k, n = int(s.sum()), len(s)
    lo, hi = wilson(k, n)
    p = k / n
    return 100 * p, 100 * (p - lo), 100 * (hi - p)


def mcnemar(a, b):
    a, b = a.astype(bool).values, b.astype(bool).values
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if x + y == 0 else binomtest(x, x + y).pvalue


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def save(fig, name, note=None):
    if note:
        fig.text(0.01, -0.07, note, ha="left", va="top", fontsize=9.5, color="#555", wrap=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.png")


def matched(e1):
    """intersection episodes (noise, seed) run by every compared method"""
    d = e1[e1.scenario == "intersection"]
    present = [m for m, *_ in METHODS if m != "Oracle" and m in set(d.arbiter)]
    keys = None
    for m in present:
        k = set(map(tuple, d[d.arbiter == m][["noise", "seed"]].values))
        keys = k if keys is None else keys & k
    keep = [(n, s) in keys for n, s in zip(d.noise, d.seed)]
    return d[keep]


def role_axis(ax, methods):
    ax.set_xticks(range(len(methods)), [f"{LABEL[m]}\n({ROLE[m]})" for m in methods], fontsize=10)


def bars_by_method(ax, methods, vals, err=None, fmt="{:.1f}", top=None, unit=""):
    x = np.arange(len(methods))
    ax.bar(x, vals, color=[COL[m] for m in methods], width=0.68,
           yerr=err, error_kw={"lw": 1, "capsize": 4, "ecolor": "#333"})
    hi = np.zeros(len(vals)) if err is None else np.asarray(err)[1]
    span = top or max(vals)
    for xi, v, e in zip(x, vals, hi):
        ax.text(xi, v + e + span * 0.02, fmt.format(v) + unit, ha="center", va="bottom", fontsize=11,
                fontweight="bold" if methods[xi] == "UCRA" else "normal")
    role_axis(ax, methods)
    if top:
        ax.set_ylim(0, top)


# ----------------------------------------------------------------- 1. main comparison
def main_figures(e1):
    d = matched(e1)
    methods = [m for m, *_ in METHODS if m in set(d.arbiter)]
    n = len(d[d.arbiter == "UCRA"])
    seeds = d[d.arbiter == "UCRA"].seed
    note = (f"Same {n} intersection episodes for every method (seeds {seeds.min()}-{seeds.max()}, noise none and high); "
            f"oracle: the {len(d[d.arbiter == 'Oracle'])} noise-free ones (it uses the true states).")
    get = {m: d[d.arbiter == m] for m in methods}

    # success
    s = [pct(get[m].success) for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_by_method(ax, methods, [v[0] for v in s], err=[[v[1] for v in s], [v[2] for v in s]], top=118, unit=" %")
    ax.set_ylabel("success (%)  ↑ better")
    ax.set_title("Main result: success (no collision and all CAVs arrived)")
    save(fig, "1_main_success", note + " Error bars: 95 % confidence interval.")

    # collisions
    c = [100 * get[m].collision.astype(bool).mean() for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_by_method(ax, methods, c, top=max(c) * 1.2, unit=" %")
    ax.set_ylabel("episodes with a collision (%)  ↓ better")
    ax.set_title("Main result: collisions")
    save(fig, "1_main_collisions", note)

    # near-misses
    v = [100 * get[m].any_severe_conflict.astype(bool).mean() for m in methods]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_by_method(ax, methods, v, top=max(v) * 1.2, unit=" %")
    ax.set_ylabel("episodes with a near-miss, PET < 1.5 s (%)  ↓ better")
    ax.set_title("Main result: near-misses")
    save(fig, "1_main_nearmisses", note)

    # episode time against UCRA on episodes completed by both
    others = [m for m in methods if m not in ("UCRA", "NoCoord")]
    U = get["UCRA"].set_index(["noise", "seed"])
    rows = []
    for m in others:
        B = get[m].set_index(["noise", "seed"])
        idx = U.index.intersection(B.index)
        both = U.loc[idx].success.astype(bool) & B.loc[idx].success.astype(bool)
        rows.append((m, U.loc[idx][both].episode_time.mean(), B.loc[idx][both].episode_time.mean(), int(both.sum())))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=UCRA_C, label="UCRA")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color=[COL[r[0]] for r in rows], label="compared method")
    for xi, r in zip(x, rows):
        ax.text(xi - 0.2, r[1] + 0.4, f"{r[1]:.1f}", ha="center", fontsize=10)
        ax.text(xi + 0.2, r[2] + 0.4, f"{r[2]:.1f}", ha="center", fontsize=10)
        ch = 100 * (r[1] / r[2] - 1)
        ax.text(xi, max(r[1], r[2]) + 2.6, f"UCRA {ch:+.0f} %", ha="center", fontsize=11, fontweight="bold",
                color=UCRA_C if ch < 0 else "#a0522d")
    ax.set_xticks(x, [f"vs {LABEL[r[0]]}\n({ROLE[r[0]]}, n={r[3]})" for r in rows], fontsize=10)
    ax.set_ylim(0, max(max(r[1], r[2]) for r in rows) * 1.25)
    ax.set_ylabel("episode time (s)  ↓ faster")
    ax.set_title("Main result: travel time, UCRA vs each method")
    ax.legend(loc="upper right")
    save(fig, "1_main_time", note + " Each pair uses only the episodes both methods completed (n). "
         "Minus = UCRA faster.")

    # overview: safety vs efficiency
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in methods:
        g = get[m]
        t = g[g.success.astype(bool)].episode_time.mean()
        sc = 100 * g.success.astype(bool).mean()
        ax.scatter(t, sc, s=420 if m == "UCRA" else 260, color=COL[m], edgecolor="black", lw=1, zorder=3)
        dx, dy, ha = 0.6, 0, "left"
        if m == "Oracle":
            dx, dy, ha = -0.8, 7, "right"
        if m == "MCTS":
            dx, dy, ha = -0.8, -6, "right"
        if m == "UCRA":
            dx, dy, ha = 0.9, 4, "left"
        if m == "FCFS":
            dx, ha = -0.6, "right"
        ax.annotate(f"{LABEL[m].replace(chr(10), ' ')}\n({ROLE[m]})", (t, sc), xytext=(t + dx, sc + dy),
                    ha=ha, va="center", fontsize=10.5, fontweight="bold" if m == "UCRA" else "normal")
    ax.set_xlabel("episode time of successful episodes (s)  ← faster")
    ax.set_ylabel("success (%)  ↑ safer")
    ax.set_ylim(0, 112)
    ax.set_xlim(8, 31)
    ax.axhspan(90, 112, xmin=0, xmax=(20 - 8) / 23, color=UCRA_C, alpha=0.07, zorder=0)
    ax.text(8.4, 108, "ideal: safe and fast", fontsize=10, color=UCRA_C, va="center")
    ax.set_title("Main result at a glance: safety vs efficiency")
    save(fig, "1_main_overview", note)

    # decision time
    d1 = e1[e1.scenario == "intersection"]
    dm = [m for m in ["FCFS", "UCRA", "Oracle", "CoDrivingLLM", "MCTS"] if m in set(d1.arbiter)]
    vals = [max(d1[d1.arbiter == m].decision_ms_mean.median(), 0.3) for m in dm]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dm))
    ax.bar(x, vals, color=[COL[m] for m in dm], width=0.65)
    ax.set_yscale("log")
    ax.axhline(200, color="#c8372d", ls="--", lw=1.5)
    ax.text(-0.4, 240, "real-time limit: 0.2 s (decisions at 5 Hz)", color="#c8372d", ha="left", fontsize=10.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.25, ("< 1 ms" if v < 1 else f"{v:.0f} ms" if v < 1000 else f"{v / 1000:.1f} s"),
                ha="center", fontsize=11, fontweight="bold" if dm[xi] == "UCRA" else "normal")
    role_axis(ax, dm)
    ax.set_ylim(0.15, 30000)
    ax.set_ylabel("decision time per step (ms, log scale)  ↓ better")
    ax.set_title("Main result: is each method fast enough for real time?")
    save(fig, "1_main_decision_time", "Median episode, intersection, same computer and load (CoDrivingLLM includes "
         "the API latency). Below the dashed line = fast enough.")

    # efficiency vs oracle (intersection)
    d0 = e1[(e1.scenario == "intersection") & (e1.noise == "none")]
    O = d0[d0.arbiter == "Oracle"].set_index("seed")
    em = [m for m in ["FCFS", "CoDrivingLLM", "CoDrivingLLM-memory", "MCTS", "UCRA"] if m in set(d0.arbiter)]
    eff = []
    for m in em:
        A = d0[d0.arbiter == m].set_index("seed")
        idx = A.index.intersection(O.index)
        both = A.loc[idx].success.astype(bool) & O.loc[idx].success.astype(bool)
        eff.append(100 * O.loc[idx][both].episode_time.mean() / A.loc[idx][both].episode_time.mean())
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_by_method(ax, em, eff, top=125, fmt="{:.0f}", unit=" %")
    for t in list(ax.texts):  # keep value labels clear of the 100 % line
        x0, y0 = t.get_position()
        if 92 <= y0 <= 104:
            t.set_position((x0, 104.5))
    ax.axhline(100, color=COL["Oracle"], ls="--", lw=1.5)
    ax.text(-0.4, 104.5, "oracle (perfect information) = 100 %", color=COL["Oracle"], ha="left", fontsize=10.5)
    ax.set_ylabel("efficiency vs oracle (%)  ↑ better")
    ax.set_title("Main result: how close to the upper bound?")
    save(fig, "1_main_vs_oracle", "Intersection, noise none: oracle's episode time / method's episode time, on episodes "
         "completed by both. MCTS can exceed 100 % because the oracle is an upper bound with perfect "
         "information, not a proven optimum.")

    # CoDrivingLLM on its own protocol
    d20 = d1[(d1.noise == "none") & (d1.seed < 1020)]
    f = lambda m: 100 * d20[d20.arbiter == m].success.astype(bool).mean()
    rows = [("their paper:\nno memory", 40, "#c9ced3"), ("ours:\nas released", f("CoDrivingLLM"), COL["CoDrivingLLM"]),
            ("ours:\n+ memory", f("CoDrivingLLM-memory"), COL["CoDrivingLLM-memory"]),
            ("their paper:\n20+ interactions", 75, "#c9ced3"), ("their paper:\nbest (Table I)", 90, "#c9ced3"),
            ("UCRA\n(ours)", f("UCRA"), UCRA_C)]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], color=[r[2] for r in rows], width=0.65,
           hatch=["//" if "paper" in r[0] else "" for r in rows], edgecolor="white")
    for xi, r in zip(x, rows):
        ax.text(xi, r[1] + 2, f"{r[1]:.0f} %", ha="center", fontsize=12, fontweight="bold" if "UCRA" in r[0] else "normal")
    ax.set_xticks(x, [r[0] for r in rows], fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("success (%)  ↑ better")
    ax.set_title("Benchmark: CoDrivingLLM on its own protocol (20 episodes, no noise)")
    save(fig, "1_main_codrivingllm_protocol", "Hatched grey bars: values reported by Fang et al. (2025) in their "
         "simulator. Coloured bars: our runs on the first 20 evaluation seeds. Our run of the released code "
         "reproduces their 40 %.")


# ----------------------------------------------------------------- 2. ablation
def ablation_fig(full, abl, name, title, abl_label, note):
    metrics = [("success", "success (%)\n↑ better"), ("collision", "episodes with\na collision (%) ↓"),
               ("any_severe_conflict", "episodes with a\nnear-miss (%) ↓")]
    F = full.set_index(["scenario", "noise", "seed"])
    A = abl.set_index(["scenario", "noise", "seed"])
    idx = F.index.intersection(A.index)
    F, A = F.loc[idx], A.loc[idx]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    for j, (D, lab, c) in enumerate([(F, "Full UCRA", UCRA_C), (A, abl_label, ABL_C)]):
        vals = [100 * D[mc].astype(bool).mean() for mc, _ in metrics]
        counts = [int(D[mc].astype(bool).sum()) for mc, _ in metrics]
        xx = x + (j - 0.5) * 0.38
        ax.bar(xx, vals, 0.38, color=c, label=lab)
        for xi, v, k in zip(xx, vals, counts):
            ax.text(xi, v + 1.5, f"{v:.1f} %\n({k})", ha="center", fontsize=10)
    for xi, (mc, _) in zip(x, metrics):
        p = mcnemar(F[mc], A[mc])
        ax.text(xi, 116, f"p = {p:.2g} ({stars(p)})" if p >= 1e-4 else f"p < 0.0001 ({stars(p)})",
                ha="center", fontsize=10, color="#333", style="italic")
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylim(0, 122)
    ax.set_ylabel("% of episodes")
    ax.set_title(title)
    ax.legend(loc="center right")
    save(fig, name, f"Same {len(idx)} episodes for both. Numbers in brackets: episode counts. p: exact McNemar test. "
         + note)


# ----------------------------------------------------------------- 3. sensitivity
SENS = [
    ("d_safe", "safety distance d_safe", "d_safe 1.5 m", "2 m (default)", "d_safe 2.5 m", "1.5 m", "2.5 m"),
    ("horizon", "prediction horizon T", "horizon 3 s", "4 s (default)", "horizon 5 s", "3 s", "5 s"),
    ("pet", "PET danger threshold", "PET 1.0 s", "1.5 s (default)", "PET 2.0 s", "1.0 s", "2.0 s"),
    ("thresholds", "fuzzy decision thresholds", "thresholds 0.35/0.65", "0.4/0.7 (default)", "thresholds 0.45/0.75", "0.35/0.65", "0.45/0.75"),
    ("risk_cap", "hard risk cap", "risk cap 0.4", "0.5 (default)", "risk cap 0.6", "0.4", "0.6"),
    ("hdv_giveway", "assumed human give-way probability", "HDV give-way 0.3", "0.5 (default)", "HDV give-way 0.7", "0.3", "0.7"),
    ("deadlock", "deadlock-breaker waiting time", "deadlock wait 1 s", "2 s (default)", "deadlock wait 3 s", "1 s", "3 s"),
    ("rollouts", "number of rollouts N", "N 50", "100 (default)", "N 200", "50", "200"),
]


def sensitivity_figs(e7):
    base = e7[e7.arbiter == "UCRA (default)"].set_index(["scenario", "noise", "seed"])
    for key, title, lo, dlab, hi, llab, hlab in SENS:
        groups = [(llab, e7[e7.arbiter == lo]), (dlab, None), (hlab, e7[e7.arbiter == hi])]
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (lab, g) in enumerate(groups):
            D = base if g is None else g.set_index(["scenario", "noise", "seed"]).loc[base.index]
            s, a, b = pct(D.success)
            col = UCRA_C if g is None else "#7fc7bb"
            ax.bar(i, s, 0.6, color=col, yerr=[[a], [b]], error_kw={"capsize": 4, "lw": 1})
            coll = int(D.collision.astype(bool).sum())
            t = D[D.success.astype(bool)].episode_time.mean()
            sig = "" if g is None else ("  " + stars(mcnemar(base.success, D.success)))
            ax.text(i, s + b + 1.5, f"{s:.1f} %{sig}", ha="center", fontsize=12, fontweight="bold")
            ax.text(i, 8, f"collisions: {coll}\ntime: {t:.1f} s", ha="center", fontsize=10, color="white",
                    fontweight="bold")
        ax.set_xticks(range(3), [groups[0][0], groups[1][0], groups[2][0]])
        ax.set_ylim(0, 112)
        ax.set_ylabel("success (%)  ↑ better")
        ax.set_title(f"Sensitivity: {title}")
        save(fig, f"3_sensitivity_{key}", "Intersection and dense intersection, noise none and high: the same 200 "
             "episodes for each value. Stars: success differs from the default (McNemar; n.s. = not significant).")


def rollout_fig(e4, gamma=0.15):
    rows = []
    for n, g in e4.groupby("N"):
        k = int(g.K.max())
        rows.append((n, 100 * (g.max_abs_error >= gamma).mean(), 100 * min(1.0, 2 * k * np.exp(-2 * n * gamma ** 2))))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[2] for r in rows], 0.4, color="#b9bfc5", label="Theorem 1: guaranteed upper limit")
    ax.bar(x + 0.2, [r[1] for r in rows], 0.4, color=UCRA_C, label="measured")
    for xi, r in zip(x, rows):
        ax.text(xi - 0.2, r[2] + 2, f"{r[2]:.1f} %", ha="center", fontsize=10)
        ax.text(xi + 0.2, r[1] + 2, f"{r[1]:.2f} %", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, [f"N = {r[0]}" + ("\n(used)" if r[0] == 100 else "") for r in rows])
    ax.set_ylim(0, 112)
    ax.set_ylabel(f"chance that a risk estimate is off by ≥ {gamma} (%)")
    ax.set_title("Sensitivity: number of rollouts N vs Theorem 1")
    ax.legend(loc="upper right")
    save(fig, "3_sensitivity_rollouts_theorem1", "60 decision situations, 50 repetitions each, compared with a "
         "10 000-rollout reference. The measured error is always far below the theorem's limit.")


# ----------------------------------------------------------------- 4. robustness
def robust_noise_all(e1):
    d = matched(e1)
    methods = [m for m in ["FCFS", "CoDrivingLLM", "CoDrivingLLM-memory", "MCTS", "UCRA"] if m in set(d.arbiter)]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    for j, (nz, alpha, lab) in enumerate([("none", 1.0, "no noise"), ("high", 0.55, "high noise (1.5 m, 1 m/s)")]):
        vals = [pct(d[(d.arbiter == m) & (d.noise == nz)].success) for m in methods]
        xx = x + (j - 0.5) * 0.38
        ax.bar(xx, [v[0] for v in vals], 0.38, color=[COL[m] for m in methods], alpha=alpha,
               edgecolor="black", lw=0.5, label=lab)
        for xi, v in zip(xx, vals):
            ax.text(xi, v[0] + 1.5, f"{v[0]:.0f}", ha="center", fontsize=10)
    role_axis(ax, methods)
    ax.set_ylim(0, 112)
    ax.set_ylabel("success (%)  ↑ better")
    ax.set_title("Robustness to perception noise: all methods")
    h = [plt.Rectangle((0, 0), 1, 1, color="#777", alpha=a) for a in (1.0, 0.55)]
    ax.set_ylim(0, 125)
    ax.legend(h, ["no noise (solid)", "high noise, 1.5 m / 1 m/s (light)"], loc="upper center", ncol=2)
    n = len(d[(d.arbiter == "UCRA") & (d.noise == "none")])
    save(fig, "4_robustness_noise_all_methods", f"Intersection, the same {n} episodes per method and noise level.")


def robust_noise_levels(e1):
    fig, ax = plt.subplots(figsize=(9, 5))
    levels = ["none", "low", "medium", "high"]
    for m in ["UCRA", "FCFS"]:
        g = e1[e1.arbiter == m]
        v = [pct(g[g.noise == nz].success) for nz in levels]
        ax.errorbar(range(4), [a[0] for a in v], yerr=[[a[1] for a in v], [a[2] for a in v]], marker="o", ms=8,
                    capsize=4, lw=2.5, color=COL[m], label=f"{LABEL[m]} ({ROLE[m]})")
        for i, a in enumerate(v):
            off = (1.8 if m == "UCRA" else -3.6) if i < 3 else (1.8 if m == "UCRA" else -4.2)
            ax.text(i + (0.12 if m == "UCRA" else -0.12), a[0] + off, f"{a[0]:.1f}", ha="center", fontsize=10,
                    color=COL[m], fontweight="bold")
    ax.set_xticks(range(4), ["none\n(0 m)", "low\n(0.2 m)", "medium\n(0.5 m)", "high\n(1.5 m)"])
    ax.set_ylim(75, 103)
    ax.set_ylabel("success (%)  ↑ better")
    ax.set_xlabel("perception noise level (position error)")
    ax.set_title("Robustness to perception noise: all four levels, all scenarios")
    ax.legend(loc="lower left")
    save(fig, "4_robustness_noise_levels", "Intersection, dense intersection and merge, 600 episodes per point. "
         "Only the methods run on the full design are shown (MCTS and CoDrivingLLM: cost).")


def robust_v2x(e6):
    e = e6.copy()
    e["v2x"] = e["scenario"].str.rsplit("_v2x_", n=1).str[1]
    ch = ["ideal", "typical", "degraded"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    for j, m in enumerate(["UCRA", "FCFS"]):
        v = [pct(e[(e.arbiter == m) & (e.v2x == c)].success) for c in ch]
        xx = x + (j - 0.5) * 0.38
        ax.bar(xx, [a[0] for a in v], 0.38, color=COL[m], yerr=[[a[1] for a in v], [a[2] for a in v]],
               error_kw={"capsize": 4}, label=f"{LABEL[m]} ({ROLE[m]})")
        for xi, a in zip(xx, v):
            ax.text(xi, a[0] + a[2] + 1.2, f"{a[0]:.1f}", ha="center", fontsize=10)
    ax.set_xticks(x, ["ideal\n(no delay, no loss)", "typical\n(0.1 s delay, 10 % lost)", "degraded\n(0.3 s delay, 30 % lost)"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("success (%)  ↑ better")
    ax.set_title("Robustness to V2X delay and packet loss")
    ax.set_ylim(0, 125)
    ax.legend(loc="upper center", ncol=2)
    save(fig, "4_robustness_v2x", "Intersection and dense intersection, noise none and high, 800 episodes per bar.")


def robust_density(e1):
    sc = [("intersection", "intersection\n(4 CAVs)"), ("intersection_dense", "dense intersection\n(8 CAVs)"),
          ("merge", "on-ramp merge\n(4 CAVs)")]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    for j, m in enumerate(["UCRA", "FCFS"]):
        v = [pct(e1[(e1.arbiter == m) & (e1.scenario == s)].success) for s, _ in sc]
        t = [e1[(e1.arbiter == m) & (e1.scenario == s) & e1.success.astype(bool)].episode_time.mean() for s, _ in sc]
        xx = x + (j - 0.5) * 0.38
        ax.bar(xx, [a[0] for a in v], 0.38, color=COL[m], yerr=[[a[1] for a in v], [a[2] for a in v]],
               error_kw={"capsize": 4}, label=f"{LABEL[m]} ({ROLE[m]})")
        for xi, a, tt in zip(xx, v, t):
            ax.text(xi, a[0] + a[2] + 1.2, f"{a[0]:.1f} %", ha="center", fontsize=10)
            ax.text(xi, 8, f"{tt:.0f} s", ha="center", fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(x, [l for _, l in sc])
    ax.set_ylim(0, 112)
    ax.set_ylabel("success (%)  ↑ better")
    ax.set_title("Robustness to traffic density and road layout")
    ax.set_ylim(0, 125)
    ax.legend(loc="upper center", ncol=2)
    save(fig, "4_robustness_density", "All noise levels, 800 episodes per bar. White numbers: mean episode time.")


def robust_mismatch(m):
    m = m.copy()
    m[["scene", "assumption"]] = m["scenario"].str.split(" \\| ", expand=True)
    cases = [("none", "high", "no real noise,\nUCRA assumes HIGH\n(over-cautious)"),
             ("none", "true", "no real noise,\nUCRA assumes it\ncorrectly"),
             ("high", "true", "high real noise,\nUCRA assumes it\ncorrectly"),
             ("high", "low", "high real noise,\nUCRA assumes LOW\n(over-confident)")]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cases))
    for j, (sc, lab) in enumerate([("intersection", "intersection"), ("dense", "dense intersection")]):
        vals, times = [], []
        for nz, a, _ in cases:
            g = m[(m.scene == sc) & (m.noise == nz) & (m.assumption == f"assumes {a}")]
            vals.append(pct(g.success)); times.append(g[g.success.astype(bool)].episode_time.mean())
        xx = x + (j - 0.5) * 0.38
        ax.bar(xx, [v[0] for v in vals], 0.38, color=UCRA_C if j == 0 else "#7fc7bb", label=lab,
               edgecolor=["black" if c[1] != "true" else "none" for c in cases], lw=1.5)
        for xi, v, t in zip(xx, vals, times):
            ax.text(xi, v[0] + 1.5, f"{v[0]:.0f} %", ha="center", fontsize=10)
            ax.text(xi, 8, f"{t:.1f} s", ha="center", fontsize=9.5, color="white", fontweight="bold")
    ax.set_xticks(x, [c[2] for c in cases], fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("success (%)  ↑ better")
    ax.set_title("Robustness to a wrong noise assumption")
    ax.set_ylim(0, 125)
    ax.legend(loc="upper center", ncol=2)
    save(fig, "4_robustness_wrong_noise_assumption", "100 episodes per bar. Black outline = wrong assumption. "
         "White numbers: mean episode time.")


# ----------------------------------------------------------------- run
def main():
    print("Figures ->", OUT)
    e1 = load("e1_baselines")
    if e1 is not None:
        main_figures(e1)
        robust_noise_all(e1); robust_noise_levels(e1); robust_density(e1)
    e3 = load("e3_fallback_ablation")
    if e3 is not None:
        ablation_fig(e3[e3.arbiter == "UCRA"], e3[e3.arbiter == "UCRA-noFallback"], "2_ablation_fallback",
                     "Ablation: removing the fallback (\"everyone yields\")", "UCRA without fallback",
                     "All 3 scenarios and 4 noise levels.")
    e2 = load("e2_uncertainty_ablation")
    if e2 is not None:
        ablation_fig(e2[e2.arbiter == "UCRA"], e2[e2.arbiter == "UCRA-noU"], "2_ablation_uncertainty",
                     "Ablation: removing the uncertainty term U", "UCRA without U",
                     "All 3 scenarios and 4 noise levels.")
    e6 = load("e6_v2x_channel")
    if e6 is not None:
        e = e6.copy()
        e["v2x"] = e["scenario"].str.rsplit("_v2x_", n=1).str[1]
        e = e[e.v2x.isin(["typical", "degraded"])]
        ablation_fig(e[e.arbiter == "UCRA"], e[e.arbiter == "UCRA-noAge"], "2_ablation_age_aware_spread",
                     "Ablation: removing the age-aware spread (delayed V2X)", "UCRA without age-aware spread",
                     "Typical and degraded V2X channels, where messages are old.")
        robust_v2x(e6)
    e7 = load("e7_sensitivity")
    if e7 is not None:
        sensitivity_figs(e7)
    e4 = load("e4_rollout_sensitivity")
    if e4 is not None:
        rollout_fig(e4)
    mm = load("e7_noise_mismatch")
    if mm is not None:
        robust_mismatch(mm)


if __name__ == "__main__":
    main()
