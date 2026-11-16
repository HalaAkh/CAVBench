# Benchmark mode: the CoDrivingLLM simulator's rules

CAVBench is our own simulator, built on HighwayEnv 1.12. CoDrivingLLM (Fang et
al., IEEE T-VT 2025) was evaluated in the authors' own copy of HighwayEnv from
2021 (in their repository, folder `highway_env`). To compare our results
directly with that paper, CAVBench runs by default with that simulator's rules
for **what counts as a crash** and **which traffic situations occur**
(`benchmark_mode: true`, the default for every experiment and every method).

## The rules

| # | Rule | CoDrivingLLM's simulator (their code) | Benchmark mode (default) | `benchmark_mode: false` (strict) |
|---|---|---|---|---|
| 1 | Collision | `Vehicle._is_colliding`: centres closer than one car length (5 m) **and** the cars' rectangles shrunk to 90 % (4.5 m × 1.8 m) intersect | the same | full 5 m × 2 m outlines touch (HighwayEnv 1.12) |
| 2 | CAV exits (intersection) | `"o" + randint(1, 4)`: o1, o2 or o3, never o0 | the same | any exit except the CAV's own approach |
| 5 | End of a decision step | the 0.2 s step stops at the simulation frame where the episode ends (`_simulate`) | the same | the step always runs its three frames |
| 3 | Arrived CAVs | stay on the road and keep driving | **off** (they leave the scene) | leave the scene |
| 4 | Human drivers past the end of their route | stay on the road | **off** (they leave) | leave |

Each rule can be switched on its own in a scenario's `env` settings:
`bm_collision`, `bm_exits`, `bm_stop_at_end`, `bm_keep_arrived`,
`bm_keep_exited_hdvs` (True / False; `None` follows `benchmark_mode`).

### Why rules 3 and 4 are off

When a vehicle reaches the end of its exit road, HighwayEnv has no lane to send
it to and turns it round into the opposite approach, so it drives back into the
intersection. CoDrivingLLM's simulator does the same. With rules 3 and 4 on, CAVs
that arrived early come back, "all CAVs arrived" is no longer true at one moment,
and returning cars collide with the others. On 20 seeds (1000-1019, noise none
and high) with all five rules on, classical FCFS failed every intersection
episode (all time-outs) and UCRA had 20 collisions in the dense scene; with rules
3 and 4 off, FCFS succeeded in 100 % (intersection) and UCRA in 95-98 %. The
artefact punishes methods for driving faster or slower than CoDrivingLLM (whose
CAVs are capped at 5 m/s and rarely reach the end of the road in time), not for
how they coordinate, so it is not used for the comparison.

### Not copied: the 5 m/s lane speed limit

Every lane of the benchmark intersection has `speed_limit = 5`. Their human
drivers ignore it (their IDM uses only the driver's own target speed, about
8 m/s). HighwayEnv 1.12's IDM caps the target speed at the lane limit, so setting
5 m/s would slow our human drivers down to 5 m/s and make the scene *less* like
theirs. CAVBench keeps 10 m/s (no effect on drivers aiming at about 8 m/s). The
CoDrivingLLM reproduction still uses 5 m/s where the method reads the lane limit
(its dTTCP calculation), as in their code.

## What stays the same

* Every method drives in the same world, with the same seeds; only the decision
  making differs. No method is changed: UCRA keeps its own 2 m safety clearance,
  and FCFS, MCTS, the oracle and CoDrivingLLM are unchanged.
* Severe conflicts are measured independently of the collision rule, from the
  trajectories (post-encroachment time < 1.5 s), so a milder collision rule
  cannot hide close passes.
* Everything else already followed the benchmark: road layout, car size, 15 Hz /
  5 Hz, 50 s episodes, CAV and human-driver set-up, no vehicles added during an
  episode.

## Remaining difference

The physics and driver models are HighwayEnv 1.12's, not the 2021 copy's (small
differences in car following and vehicle dynamics).

## For the paper

"For direct comparability with CoDrivingLLM, CAVBench uses the benchmark
simulator's collision rule (vehicle footprints reduced to 90 %) and CAV routing
(no exit towards o0). All methods are evaluated under these rules on unseen
seeds; severe conflicts are measured independently by PET."
