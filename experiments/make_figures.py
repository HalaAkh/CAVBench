"""Figures for the paper, made from the CSV files in results/.

    python experiments/make_figures.py

Writes PDF (for LaTeX) and PNG (for viewing) files to results/figures/. Uses the
same episodes and definitions as experiments/final_report.py (success = no
collision and all CAVs arrived; Wilson 95 % intervals; efficiency vs oracle on
noise-free episodes completed by both). Missing CSV files are skipped.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"
NOISE = ["none", "low", "medium", "high"]
COL = {"UCRA": "#0f8f7e", "FCFS": "#4f6fb0", "MCTS": "#d88a12", "CoDrivingLLM": "#c8372d",
       "CoDrivingLLM-memory": "#e07b6f", "Oracle": "#2e8b4f", "NoCoord": "#7a8590",
       "UCRA-noU": "#7fc7bb", "UCRA-noFallback": "#c8372d", "UCRA-noAge": "#7fc7bb", "paper": "#b9bfc5"}
NAME = {"NoCoord": "No coord.", "CoDrivingLLM-memory": "CoDrivingLLM\n+ memory", "CoDrivingLLM": "CoDrivingLLM"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.bbox": "tight", "figure.dpi": 150})


def load(name):
    p = RES / f"{name}.csv"
    if not p.exists():
        print(f"  skipped: {p.name} not found")
        return None
    return pd.read_csv(p)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_ci(s):
    k, n = int(s.sum()), len(s)
    lo, hi = wilson(k, n)
    p = k / n
    return 100 * p, 100 * (p - lo), 100 * (hi - p)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def hbar(ax, labels, vals, colors, err=None, fmt="{:.1f}", xmax=None, ref=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, xerr=err, error_kw={"lw": 0.8, "capsize": 2}, height=0.65)
    ax.set_yticks(y, labels)
    hi = np.zeros(len(vals)) if err is None else np.asarray(err)[1]
    for yi, v, e in zip(y, vals, hi):
        ax.text(v + e + (xmax or max(vals)) * 0.015, yi, fmt.format(v), va="center", fontsize=8)
    if xmax:
        ax.set_xlim(0, xmax)
    if ref is not None:
        ax.axvline(ref, color="#c8372d", ls="--", lw=1)


# ---------------------------------------------------------------- E1
def fig_main(e1):
    d = e1[e1.scenario == "intersection"]
    order = ["UCRA", "Oracle", "FCFS", "MCTS", "CoDrivingLLM-memory", "CoDrivingLLM", "NoCoord"]
    order = [m for m in order if m in set(d.arbiter)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), gridspec_kw={"wspace": 0.9})
    s = [rate_ci(d[d.arbiter == m].success.astype(bool)) for m in order]
    lab = [f"{NAME.get(m, m)} (n={len(d[d.arbiter == m])})" for m in order]
    hbar(axes[0], lab, [x[0] for x in s], [COL[m] for m in order],
         err=np.array([[x[1] for x in s], [x[2] for x in s]]), xmax=115)
    axes[0].set_title("(a) Success (%)")
    col = [100 * d[d.arbiter == m].collision.astype(bool).mean() for m in order]
    hbar(axes[1], [NAME.get(m, m) for m in order], col, [COL[m] for m in order], xmax=max(col) * 1.2)
    axes[1].set_title("(b) Episodes with a collision (%)")
    sev = [100 * d[d.arbiter == m].any_severe_conflict.astype(bool).mean() for m in order]
    hbar(axes[2], [NAME.get(m, m) for m in order], sev, [COL[m] for m in order], xmax=max(sev) * 1.2)
    axes[2].set_title("(c) Episodes with PET < 1.5 s (%)")
    fig.suptitle("Intersection, all noise levels (E1)", y=1.04, fontsize=10)
    save(fig, "fig_e1_main")


def paired_time(e1, a, b, scen=None, noise=None):
    d = e1
    if scen:
        d = d[d.scenario == scen]
    if noise:
        d = d[d.noise == noise]
    A = d[d.arbiter == a].set_index(["scenario", "noise", "seed"])
    B = d[d.arbiter == b].set_index(["scenario", "noise", "seed"])
    idx = A.index.intersection(B.index)
    both = A.loc[idx, "success"].astype(bool) & B.loc[idx, "success"].astype(bool)
    return A.loc[idx][both].episode_time.mean(), B.loc[idx][both].episode_time.mean(), int(both.sum())


def fig_time(e1):
    others = [m for m in ["FCFS", "CoDrivingLLM", "CoDrivingLLM-memory", "MCTS", "Oracle"] if m in set(e1.arbiter)]
    rows = [(m, *paired_time(e1, "UCRA", m)) for m in others]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=COL["UCRA"], label="UCRA")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color=[COL[r[0]] for r in rows], label="other method")
    for xi, r in zip(x, rows):
        ch = 100 * (r[1] / r[2] - 1)
        ax.text(xi, max(r[1], r[2]) + 0.8, f"{ch:+.0f} %", ha="center", fontsize=8)
    ax.set_xticks(x, [f"vs {NAME.get(r[0], r[0])}\n(n={r[3]})" for r in rows], fontsize=8)
    ax.set_ylabel("episode time (s)")
    ax.set_title("Episode time on matched episodes completed by both")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "fig_e1_paired_time")


def fig_oracle(e1):
    d = e1[e1.noise == "none"]
    rows = []
    for sc in ["intersection", "intersection_dense", "merge"]:
        for m in ["UCRA", "FCFS", "MCTS", "CoDrivingLLM", "CoDrivingLLM-memory"]:
            if m not in set(d[d.scenario == sc].arbiter) or "Oracle" not in set(d[d.scenario == sc].arbiter):
                continue
            ta, to, n = paired_time(d, m, "Oracle", scen=sc)
            if n >= 5:
                rows.append((sc, m, 100 * to / ta))
    scen = ["intersection", "intersection_dense", "merge"]
    methods = [m for m in ["UCRA", "MCTS", "CoDrivingLLM-memory", "CoDrivingLLM", "FCFS"] if any(r[1] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        for i, sc in enumerate(scen):
            v = [r[2] for r in rows if r[0] == sc and r[1] == m]
            if v:
                ax.bar(i + (j - (len(methods) - 1) / 2) * w, v[0], w, color=COL[m], label=NAME.get(m, m).replace("\n", " ") if i == 0 or not any(r[0] == "intersection" and r[1] == m for r in rows) else None)
                ax.text(i + (j - (len(methods) - 1) / 2) * w, v[0] + 1.5, f"{v[0]:.0f}", ha="center", fontsize=7)
    ax.axhline(100, color="#2e8b4f", ls="--", lw=1)
    ax.set_xticks(range(3), ["Intersection", "Dense intersection", "Merge"])
    ax.set_ylabel("efficiency vs oracle (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Efficiency relative to the perfect-information oracle (noise none)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig_oracle_efficiency")


def fig_noise(e1):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), gridspec_kw={"wspace": 0.3})
    for m in ["UCRA", "FCFS", "MCTS", "CoDrivingLLM-memory", "CoDrivingLLM"]:
        g = e1[e1.arbiter == m]
        if g.empty:
            continue
        xs, ys, lo, hi = [], [], [], []
        for i, nz in enumerate(NOISE):
            s = g[g.noise == nz].success.astype(bool)
            if len(s):
                p, a, b = rate_ci(s)
                xs.append(i); ys.append(p); lo.append(a); hi.append(b)
        axes[0].errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=4, capsize=2, color=COL[m],
                         ls="--" if m.startswith("Co") or m == "MCTS" else "-", label=NAME.get(m, m).replace("\n", " "))
    axes[0].set_xticks(range(4), ["none", "low", "medium", "high"])
    axes[0].set_xlabel("perception noise")
    axes[0].set_ylabel("success (%)")
    axes[0].set_title("(a) Success vs noise (all scenarios run)")
    axes[0].legend(frameon=False, fontsize=7)
    for m in ["UCRA", "FCFS"]:
        g = e1[(e1.arbiter == m) & e1.success.astype(bool)]
        t = [g[g.noise == nz].episode_time.mean() for nz in NOISE]
        e = [1.96 * g[g.noise == nz].episode_time.sem() for nz in NOISE]
        axes[1].errorbar(range(4), t, yerr=e, marker="o", ms=4, capsize=2, color=COL[m], label=m)
    axes[1].set_xticks(range(4), ["none", "low", "medium", "high"])
    axes[1].set_xlabel("perception noise")
    axes[1].set_ylabel("episode time (s)")
    axes[1].set_title("(b) Episode time vs noise (3 scenarios)")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "fig_robustness_noise")


def fig_density(e1):
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    sc = ["intersection", "intersection_dense", "merge"]
    for j, m in enumerate(["UCRA", "FCFS"]):
        vals = [rate_ci(e1[(e1.arbiter == m) & (e1.scenario == s)].success.astype(bool)) for s in sc]
        x = np.arange(3) + (j - 0.5) * 0.38
        ax.bar(x, [v[0] for v in vals], 0.38, color=COL[m], label=m,
               yerr=[[v[1] for v in vals], [v[2] for v in vals]], error_kw={"lw": 0.8, "capsize": 2})
        for xi, v in zip(x, vals):
            ax.text(xi, v[0] + 2.5, f"{v[0]:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(3), ["Intersection\n(4 CAVs)", "Dense\n(8 CAVs)", "Merge"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("success (%)")
    ax.set_title("Traffic density and topology (E1, all noise levels)")
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    save(fig, "fig_robustness_density")


# ---------------------------------------------------------------- E6
def fig_v2x(e6):
    e6 = e6.copy()
    e6[["base", "v2x"]] = e6["scenario"].str.rsplit("_v2x_", n=1, expand=True)
    ch = ["ideal", "typical", "degraded"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), gridspec_kw={"wspace": 0.3})
    for m in ["UCRA", "UCRA-noAge", "FCFS"]:
        g = e6[e6.arbiter == m]
        v = [rate_ci(g[g.v2x == c].success.astype(bool)) for c in ch]
        axes[0].errorbar(range(3), [x[0] for x in v], yerr=[[x[1] for x in v], [x[2] for x in v]], marker="o", ms=4,
                         capsize=2, color=COL[m], ls=":" if m == "UCRA-noAge" else "-", label=m)
        s = [100 * g[g.v2x == c].any_severe_conflict.astype(bool).mean() for c in ch]
        if m != "FCFS":
            axes[1].plot(range(3), s, marker="o", ms=4, color=COL[m], ls=":" if m == "UCRA-noAge" else "-", label=m)
    for ax in axes:
        ax.set_xticks(range(3), ["ideal", "typical\n(0.1 s, 10 %)", "degraded\n(0.3 s, 30 %)"])
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("success (%)")
    axes[0].set_title("(a) Success vs V2X channel (E6)")
    axes[1].set_ylabel("episodes with PET < 1.5 s (%)")
    axes[1].set_title("(b) Near-misses: age-aware spread on/off")
    save(fig, "fig_robustness_v2x")


# ---------------------------------------------------------------- ablation
def fig_ablation(e2, e3, e6):
    rows = []
    if e3 is not None:
        rows += [("Full UCRA", e3[e3.arbiter == "UCRA"], "UCRA"), ("without fallback (E3)", e3[e3.arbiter == "UCRA-noFallback"], "UCRA-noFallback")]
    if e2 is not None:
        rows.insert(1, ("without U (E2)", e2[e2.arbiter == "UCRA-noU"], "UCRA-noU"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), gridspec_kw={"wspace": 0.9})
    lab = [r[0] for r in rows]
    c = [COL[r[2]] for r in rows]
    hbar(axes[0], lab, [100 * r[1].success.astype(bool).mean() for r in rows], c, xmax=115)
    axes[0].set_title("(a) Success (%)")
    hbar(axes[1], lab, [int(r[1].collision.astype(bool).sum()) for r in rows], c, fmt="{:.0f}", xmax=500)
    axes[1].set_title("(b) Collisions (2400 episodes)")
    if e6 is not None:
        e = e6.copy()
        e["v2x"] = e["scenario"].str.rsplit("_v2x_", n=1).str[1]
        e = e[e.v2x.isin(["typical", "degraded"])]
        v = [int(e[e.arbiter == m].any_severe_conflict.astype(bool).sum()) for m in ["UCRA", "UCRA-noAge"]]
        hbar(axes[2], ["Full UCRA", "without age-aware\nspread (E6)"], v, [COL["UCRA"], COL["UCRA-noAge"]], fmt="{:.0f}", xmax=70)
        axes[2].set_title("(c) Near-miss episodes, delayed V2X")
    save(fig, "fig_ablation")


# ---------------------------------------------------------------- E4, E7
def fig_e4(e4, gamma=0.15):
    rows = []
    for n, g in e4.groupby("N"):
        k = int(g.K.max())
        rows.append((n, (g.max_abs_error >= gamma).mean(), min(1.0, 2 * k * np.exp(-2 * n * gamma ** 2)), g.max_abs_error.mean()))
    t = pd.DataFrame(rows, columns=["N", "emp", "bound", "mean"])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(t.N, t.bound * 100, "s--", color="#7a8590", label="Theorem 1 bound")
    ax.plot(t.N, np.maximum(t.emp * 100, 1e-3), "o-", color=COL["UCRA"], label="measured")
    for n, e in zip(t.N, t.emp):
        ax.annotate(f"{100 * e:.2f} %", (n, max(100 * e, 1e-3)), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.axvline(100, color="#c8372d", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(t.N, [str(n) for n in t.N])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("rollouts N"); ax.set_ylabel(f"P(max error ≥ {gamma}) (%)")
    ax.set_title("Risk-estimate accuracy vs Theorem 1 (E4)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig_e4_theorem1")


def mcnemar_p(a, b):
    from scipy.stats import binomtest
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if x + y == 0 else binomtest(x, x + y).pvalue


def fig_e7(e7):
    base = e7[e7.arbiter == "UCRA (default)"].set_index(["scenario", "noise", "seed"])
    b0 = 100 * base.success.astype(bool).mean()
    pairs = [("safety distance", "d_safe 1.5 m", "d_safe 2.5 m"), ("horizon T", "horizon 3 s", "horizon 5 s"),
             ("HDV give-way prob.", "HDV give-way 0.3", "HDV give-way 0.7"),
             ("fuzzy thresholds", "thresholds 0.35/0.65", "thresholds 0.45/0.75"),
             ("deadlock wait", "deadlock wait 1 s", "deadlock wait 3 s"), ("rollouts N", "N 50", "N 200"),
             ("PET threshold", "PET 1.0 s", "PET 2.0 s"), ("risk cap", "risk cap 0.4", "risk cap 0.6")]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (name, lo, hi) in enumerate(pairs[::-1]):
        for v, dy in ((lo, 0.18), (hi, -0.18)):
            g = e7[e7.arbiter == v].set_index(["scenario", "noise", "seed"])
            idx = base.index.intersection(g.index)
            delta = 100 * g.loc[idx].success.astype(bool).mean() - b0
            p = mcnemar_p(base.loc[idx].success.astype(bool), g.loc[idx].success.astype(bool))
            ax.barh(i + dy, delta, 0.32, color="#c8372d" if delta < 0 else "#2e8b4f", alpha=1 if dy > 0 else 0.6)
            label = v.split(" ", 1)[1] if " " in v else v
            ax.text(delta + (0.15 if delta >= 0 else -0.15), i + dy, f"{label}{' ★' if p < 0.05 else ''}",
                    va="center", ha="left" if delta >= 0 else "right", fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(pairs)), [p[0] for p in pairs[::-1]])
    ax.set_xlim(-9, 4)
    ax.set_xlabel(f"change in success (points) vs default {b0:.1f} %")
    ax.set_title("Sensitivity analysis (E7)  ★ p < 0.05 (McNemar)")
    save(fig, "fig_e7_tornado")


def fig_mismatch(m):
    m = m.copy()
    m[["scene", "assumption"]] = m["scenario"].str.split(" \\| ", expand=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), gridspec_kw={"wspace": 0.3})
    asm = ["assumes low", "assumes true", "assumes high"]
    cols = ["#d88a12", "#0f8f7e", "#4f6fb0"]
    for ax, sc in zip(axes, ["intersection", "dense"]):
        for j, a in enumerate(asm):
            for i, nz in enumerate(["none", "high"]):
                g = m[(m.scene == sc) & (m.assumption == a) & (m.noise == nz)]
                t = g[g.success.astype(bool)].episode_time.mean()
                ax.bar(i + (j - 1) * 0.26, t, 0.26, color=cols[j], label=a if i == 0 else None)
                ax.text(i + (j - 1) * 0.26, t + 0.4, f"{100 * g.success.astype(bool).mean():.0f}%", ha="center", fontsize=7)
        ax.set_xticks([0, 1], ["true noise: none", "true noise: high"])
        ax.set_ylabel("episode time (s)")
        ax.set_title(f"Wrong noise assumption, {sc} (E7b)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    fig.text(0.5, -0.06, "Labels above the bars: success rate.", ha="center", fontsize=8)
    save(fig, "fig_e7b_noise_mismatch")


# ---------------------------------------------------------------- CoDrivingLLM, decision time
def fig_codriving(e1):
    d = e1[(e1.scenario == "intersection") & (e1.noise == "none")]
    def first20(m):
        g = d[(d.arbiter == m) & (d.seed < 1020)]
        return 100 * g.success.astype(bool).mean() if len(g) else None
    rows = [("CoDrivingLLM, no memory\n(paper, Fig. 7)", 40, "paper"),
            ("CoDrivingLLM, released\n(ours, same 20 seeds)", first20("CoDrivingLLM"), "CoDrivingLLM"),
            ("CoDrivingLLM + memory\n(ours, same 20 seeds)", first20("CoDrivingLLM-memory"), "CoDrivingLLM-memory"),
            ("CoDrivingLLM, 20+ interactions\n(paper, Fig. 7)", 75, "paper"),
            ("CoDrivingLLM 2-shot\n(paper, Table I)", 90, "paper"),
            ("UCRA\n(ours, same 20 seeds)", first20("UCRA"), "UCRA")]
    rows = [r for r in rows if r[1] is not None]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    hbar(ax, [r[0] for r in rows], [r[1] for r in rows], [COL[r[2]] for r in rows], fmt="{:.0f} %", xmax=110)
    ax.set_xlabel("success (%), intersection, noise none, 20 episodes")
    ax.set_title("CoDrivingLLM on its own protocol (20 seeds)")
    ax.tick_params(axis="y", labelsize=7)
    save(fig, "fig_codrivingllm")


def fig_decision(e1, e5):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), gridspec_kw={"wspace": 0.45})
    d = e1[e1.scenario == "intersection"]
    order = [m for m in ["FCFS", "UCRA", "Oracle", "CoDrivingLLM", "MCTS"] if m in set(d.arbiter)]
    v = [max(d[d.arbiter == m].decision_ms_mean.median(), 0.3) for m in order]
    y = np.arange(len(order))[::-1]
    axes[0].barh(y, v, color=[COL[m] for m in order], height=0.6)
    axes[0].set_xscale("log"); axes[0].set_yticks(y, order)
    axes[0].axvline(200, color="#c8372d", ls="--", lw=1)
    axes[0].text(210, y[0] + 0.35, "0.2 s budget", color="#c8372d", fontsize=7)
    for yi, x in zip(y, v):
        axes[0].text(x * 1.15, yi, f"{x:.0f} ms" if x < 1000 else f"{x / 1000:.1f} s", va="center", fontsize=7)
    axes[0].set_xlabel("decision time per step (ms, log scale, median episode)")
    axes[0].set_title("(a) Decision time (E1, intersection)")
    if e5 is not None:
        sc = ["intersection", "intersection_dense", "merge"]
        data = [e5[(e5.scenario == s) & (e5.arbiter == "UCRA")].decision_ms_p95 for s in sc]
        axes[1].boxplot(data, tick_labels=["Intersection", "Dense", "Merge"], widths=0.5,
                        medianprops={"color": COL["UCRA"]})
        axes[1].axhline(200, color="#c8372d", ls="--", lw=1)
        axes[1].set_ylabel("95th-percentile decision time (ms)")
        axes[1].set_title("(b) UCRA alone, one worker (E5)")
    save(fig, "fig_decision_time")


def main():
    print("Figures ->", OUT)
    e1 = load("e1_baselines")
    if e1 is not None:
        fig_main(e1); fig_time(e1); fig_oracle(e1); fig_noise(e1); fig_density(e1); fig_codriving(e1)
        fig_decision(e1, load("e5_realtime"))
    e2, e3, e6 = load("e2_uncertainty_ablation"), load("e3_fallback_ablation"), load("e6_v2x_channel")
    if e6 is not None:
        fig_v2x(e6)
    if e2 is not None or e3 is not None:
        fig_ablation(e2, e3, e6)
    e4 = load("e4_rollout_sensitivity")
    if e4 is not None:
        fig_e4(e4)
    e7 = load("e7_sensitivity")
    if e7 is not None:
        fig_e7(e7)
    m = load("e7_noise_mismatch")
    if m is not None:
        fig_mismatch(m)


if __name__ == "__main__":
    main()
