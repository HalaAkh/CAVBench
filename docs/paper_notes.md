# Notes for the paper

Decisions taken during development that the paper should state. Each item says
what to write and where the evidence is. All tuning used seeds 100000-100019;
the evaluation seeds (0-199) were not used before the final runs.

## Method (UCRA)

* **Number of rollouts N = 100**, from Theorem 1 (N ≥ 98 for K = 4,
  γ = 0.15, δ = 0.1). N = 50 would meet the decision-time target in every
  scene (p95 48 ms) but would weaken the bound to γ = 0.21; we keep the
  theorem's N. Experiment 4 shows N = 25 / 50 / 100 as a speed / precision
  trade-off and the empirical estimation error, which is smaller than the
  worst-case bound.
* **Hard risk cap:** an option with R̂ ≥ 0.5 never proceeds, for any waiting
  time; together with Theorem 1: a vehicle proceeds only if R < 0.5 + γ with
  probability ≥ 0.9, and an Accept with W ≤ 1 s implies R < 0.35.
  Docs: `docs/fuzzy_layer.md`.
* **Priority in the conflict area:** a CAV that has priority and can no longer
  stop keeps it until it has cleared the area.
* **Deadlock breaker** (H. El Akhrass): after 2 s in fallback, the option with
  the lowest near-contact risk (clearance < 0.5 m) is released at the cautious
  speed if that risk is ≤ 10 %. Together with the cap and the priority rule
  it removed all time-outs on the tuning seeds. Ablation on 240 tuning
  episodes (12 conditions × 20, same geometry):

  | Version | Success | Collisions | Time-outs | Severe-conflict eps |
  |---|---|---|---|---|
  | breaker only | 97.5 % | 3 | 3 | 3 |
  | cap + priority only | 96.7 % | 3 | 5 | 1 |
  | **both (final)** | **98.8 %** | 3 | **0** | 2 |

  Paired sign tests on the time-outs: both vs breaker only p ≈ 0.25, both vs
  cap + priority only p ≈ 0.06, so a trend, not proven at this sample size.
  The cap is not applied to the breaker's release (a liveness exception);
  state the guarantee with this exception.
* **Remaining weakness to watch:** 3 collisions in those 240 episodes, the same
  with and without the deadlock fixes, mostly at the intersection with degraded
  V2X (0.3 s delay, 30 % loss); the classical FCFS had none.

## Decision time (RQ4)

* Measure on a free machine: one process per physical core, one thread per
  process for the maths library. Earlier numbers of 130-260 ms came from an
  overloaded machine (more processes than cores).
* Report the **mean and the 95th percentile**, not the maximum. With N = 100:
  mean 27-66 ms in every scene; the 95th percentile is below 100 ms except in
  the dense scene (about 133 ms). The single slowest decision can exceed
  100 ms (≈ 220 ms in the dense scene); the first decision in a fresh process
  also pays a one-off route-geometry computation (≈ 0.9 s, cached afterwards).
* 98 % of the decision time is the rollout simulation; it scales with N.

## Baselines

* **FCFS** is the classical version (fixed arrival order, one vehicle in the
  conflict area at a time, 2σ noise margin). CoDrivingLLM uses FCFS without
  describing it; its reported FCFS (2.7 m/s, mean PET 18.1 s) matches this
  behaviour. Docs: `docs/fcfs_baseline.md`.
* **MCTS** (Lin et al., T-ITS 2025): re-implementation with documented
  adaptations and a calibration on tuning seeds; trends of the paper reproduced
  (depth, dynamic safety distance). Docs: `docs/mcts_baseline.md`. The MCTS
  paper does not compare against FCFS.
* **Oracle**: an approximate upper bound with perfect information
  (passing-order enumeration on copies of the simulation, after Xu et al.,
  2020); not a proven optimum. Docs: `docs/oracle_baseline.md` (branch oracle).

## Benchmark

* **Evaluation seeds:** an earlier E1 run on seeds 0-199 was inspected while
  developing the deadlock breaker (UCRA time-outs on seeds 105, 71 and 18 were
  replayed). To keep the evaluation unseen, the final runs should use a fresh
  seed range (proposal: 1000-1199) and the paper should say so.
* **Route fix:** `geometry.remaining_route` keeps the lane a car is still
  physically on while HighwayEnv already follows the next lane (2.5 m before
  its end). It changes the shared states of every arbiter; all final results
  use it.

* **Bug fixed before the final runs:** the intersection scenario modified
  HighwayEnv's shared human-driver class, so merge episodes run after an
  intersection episode in the same worker had different human drivers. Fixed
  (commit 3373d31); all final results use the fixed code.
