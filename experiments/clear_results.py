"""Delete rows from the results files so that they are run again.

The experiment runner skips every episode already in results/*.csv. After a
change to a method (or to the shared simulator), use this script to remove
the rows that must be recomputed, then run the experiments again: only the
removed rows are run.

    python experiments/clear_results.py --all            # every method, every file
    python experiments/clear_results.py --arbiter UCRA   # only UCRA and its variants
                                                          # (UCRA-noU, UCRA-noFallback, ...) and E4

CoDrivingLLM rows are never removed (they cost money).

A copy of each changed file is kept as results/old/<file>.csv.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="remove every row (all methods)")
    g.add_argument("--arbiter", help="remove rows whose method label starts with this text")
    a = ap.parse_args(argv)

    (RESULTS / "old").mkdir(parents=True, exist_ok=True)
    for path in sorted(RESULTS.glob("*.csv")):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
            header = rows[0].keys() if rows else []
        if "arbiter" not in header:  # E4 (rollout sensitivity): one file for UCRA's rollouts
            if a.all or (a.arbiter and "UCRA".startswith(a.arbiter)):
                shutil.copy2(path, RESULTS / "old" / path.name)
                path.unlink()
                print(f"{path.name}: removed (recomputed by run_all)")
            continue
        def drop(r):
            label = r["arbiter"]
            if label.startswith("CoDrivingLLM"):
                return False
            return a.all or label.startswith(a.arbiter)
        keep = [r for r in rows if not drop(r)]
        if len(keep) == len(rows):
            continue
        shutil.copy2(path, RESULTS / "old" / path.name)
        if not keep:  # nothing left: start a fresh file (with the current columns)
            path.unlink()
            print(f"{path.name}: removed all {len(rows)} rows")
            continue
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(header))
            w.writeheader()
            w.writerows(keep)
        print(f"{path.name}: removed {len(rows) - len(keep)} rows, kept {len(keep)}")


if __name__ == "__main__":
    main()
