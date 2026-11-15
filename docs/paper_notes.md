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
* **Limitation:** rare deadlocks (≈ 2 % of tuning episodes, all without
  perception noise) when two CAVs stop close to each other's path; the action
  space has no maneuver (e.g. reversing) to resolve them.

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

* **Bug fixed before the final runs:** the intersection scenario modified
  HighwayEnv's shared human-driver class, so merge episodes run after an
  intersection episode in the same worker had different human drivers. Fixed
  (commit 3373d31); all final results use the fixed code.
