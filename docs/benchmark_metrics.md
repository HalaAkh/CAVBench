# The metrics of CoDrivingLLM's Table I

CoDrivingLLM (Fang et al., IEEE T-VT 2025) compares methods by success rate,
PET (average / maximum / minimum) and travel velocity (average / maximum /
minimum). CAVBench records the same quantities for every episode, so that
`python experiments/benchmark_table.py` prints the results in the format of
that table, with the paper's own numbers underneath for reference.

| Metric | How the paper's data defines it | CAVBench column(s) |
|---|---|---|
| Success | no collision: in the authors' raw data (`videos&data`), episodes that reach the 50 s limit without a crash are counted as successes, and every failed episode ends early with a crash | `collision_free` (CAVBench also keeps its stricter `success`: no collision **and** all CAVs arrived) |
| PET avg / max / min | "Time Difference" in `results_summary.xlsx`: for each pair of vehicles whose paths cross, the time between the moments the two vehicles pass the crossing point; the table's values are the mean / max / min over all pairs of all episodes (checked: e.g. 2-shot 10.4 / 44.0 / 1.8 s from their files, 10.3 / 44 / 1.8 s in the table) | `bpet_sum`, `bpet_n`, `bpet_max`, `bpet_min` per episode; pairs with at least one CAV, crossing point as in CAVBench's PET, "passing" = closest approach (within 3 m) |
| Travel velocity avg / max / min | `average_v.xlsx`: one average speed per vehicle; mean / max / min over all vehicles of all episodes | `vel_sum`, `vel_n`, `vel_max`, `vel_min` per episode: the average speed of each **CAV** while it is in the scene |

Averages are pooled over all pairs (or CAVs) of all episodes, as in the paper:
average = Σ sum / Σ n; maximum and minimum over all episodes.

## Limits

* The authors' analysis script is not released. The PET definition above
  reproduces their table from their files; their per-vehicle speed does not
  equal the plain mean of the recorded speeds in every case, and their files
  include human-driven vehicles. CAVBench reports the CAVs, the vehicles the
  methods control.
* This PET (time between passing the crossing point) is not CAVBench's PET
  (gap between leaving and entering a 3 m conflict area). The latter, with its
  1.5 s threshold, remains the severe-conflict measure.
* The paper's numbers come from 20 episodes in its own simulator; CAVBench's
  come from the same seeds for every method in benchmark mode
  (docs/benchmark_mode.md).
