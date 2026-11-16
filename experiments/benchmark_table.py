"""The results in the format of CoDrivingLLM's Table I (Fang et al., 2025).

    python experiments/benchmark_table.py                      # results/e1_baselines.csv
    python experiments/benchmark_table.py results/other.csv --scenario intersection --noise none

For each scenario, noise level and method: success as the paper counts it
(no collision), CAVBench's stricter success (no collision and all CAVs
arrived), PET average / maximum / minimum over all crossing pairs of all
episodes, and travel velocity average / maximum / minimum over all CAVs of
all episodes (docs/benchmark_metrics.md). The values reported in the paper
(intersection only, 20 episodes, their simulator) are printed underneath for
reference. The table is also written to results/benchmark_table.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Fang et al. (2025), Table I, unsignalized intersection: success, PET avg/max/min [s],
# travel velocity avg/max/min [m/s].
PAPER = [
    ("FCFS", 90, 18.1, 45.4, 1.4, 2.7, 7.9, 0.6),
    ("iDFST", 85, 15.1, 47.6, 1.2, 4.1, 7.9, 0.2),
    ("MCTS", 65, 11.8, 48.4, 1.2, 4.3, 7.9, 0.6),
    ("Cooperative game", 55, 5.7, 18.6, 0.8, 5.7, 7.9, 0.9),
    ("MADQN", 20, 3.7, 9.0, 1.8, 6.1, 6.5, 5.4),
    ("Dilu", 38, 2.4, 7.6, 0.2, 2.5, 5.8, 0.3),
    ("CoDrivingLLM (2-shot)", 90, 10.3, 44.0, 1.8, 5.7, 8.0, 1.0),
]
COLS = ["Success % (no collision)", "Success % (all arrived)", "PET avg", "PET max", "PET min",
        "Velocity avg", "Velocity max", "Velocity min", "Episodes"]


def table(df: pd.DataFrame) -> pd.DataFrame:
    need = {"collision_free", "bpet_sum", "bpet_n", "vel_sum", "vel_n"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"The file has no {sorted(missing)} columns: it was written before the "
                         f"Table I metrics were added. Re-run the experiment.")
    rows = []
    for (sc, noise, arb), g in df.groupby(["scenario", "noise", "arbiter"], sort=False):
        b = g[g["bpet_n"] > 0]
        rows.append({
            "Scenario": sc, "Noise": noise, "Method": arb,
            COLS[0]: 100 * g["collision_free"].astype(bool).mean(),
            COLS[1]: 100 * g["success"].astype(bool).mean(),
            COLS[2]: g["bpet_sum"].sum() / max(g["bpet_n"].sum(), 1),
            COLS[3]: b["bpet_max"].max(), COLS[4]: b["bpet_min"].min(),
            COLS[5]: g["vel_sum"].sum() / max(g["vel_n"].sum(), 1),
            COLS[6]: g["vel_max"].max(), COLS[7]: g["vel_min"].min(),
            COLS[8]: len(g),
        })
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".1f")  # needs the "tabulate" package
    except ImportError:
        return df.round(1).to_string(index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", type=Path, default=ROOT / "results" / "e1_baselines.csv")
    ap.add_argument("--scenario", help="only this scenario (e.g. intersection)")
    ap.add_argument("--noise", help="only this noise level (e.g. none)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark_table.md")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    if a.scenario:
        df = df[df["scenario"] == a.scenario]
    if a.noise:
        df = df[df["noise"] == a.noise]
    t = table(df)
    paper = pd.DataFrame(PAPER, columns=["Method", COLS[0], COLS[2], COLS[3], COLS[4],
                                         COLS[5], COLS[6], COLS[7]])
    text = ("## CAVBench (benchmark mode, same seeds for every method)\n\n"
            + _md(t)
            + "\n\n## Reported by Fang et al. (2025), Table I (intersection, their simulator, "
              "20 episodes)\n\n" + _md(paper) + "\n")
    print(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
