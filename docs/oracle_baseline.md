# Oracle: an upper bound with perfect information

`kind: oracle` (`cavbench/arbiters/oracle.py`, class `OracleArbiter`).

The oracle is **not a realistic method**. It cheats on purpose, to show the best
result that can be reached in each situation, so that the other methods can be
compared against it:

* **True states.** It reads the real vehicle states from the simulator: no
  perception noise, no V2X delay or packet loss. Conflict groups are detected on
  these true states, with the same `ConflictDetector` as every other arbiter
  (sigma = 0). The car-following safety layer (`FollowingSafety`) is applied as for
  every arbiter, also on the true states.
* **Known future.** It evaluates each option on an exact copy of the running
  simulation (`copy.deepcopy` of the environment, including its random
  generator): the same physics, the same human drivers with their own parameters,
  the same random spawns. So for the option being tested it knows what will
  really happen.

In the paper: "an upper bound with perfect information", obtained by
passing-order enumeration as in Xu et al. (2020). It is **not** a proven global
optimum (see the limitations below).

Reference for the enumeration: H. Xu, Y. Zhang, L. Li, W. Li, "Cooperative
driving at unsignalized intersections using tree search", IEEE Trans. Intell.
Transp. Syst., vol. 21, no. 11, pp. 4563–4571, 2020, doi:10.1109/TITS.2019.2940641.
They use the enumeration of all passing orders as the global-optimum reference,
evaluate the orders with an analytic model of all-CAV traffic, and re-plan every
2 s. We do the same, but evaluate the orders in the real simulator with perfect
information, human drivers included.

## How it gets the simulator

The runner already has a hook for arbiters that need the environment:
`run_episode` calls `arbiter.attach(env)` right after `env.reset()` (it was added
for the CoDrivingLLM reproduction). The oracle uses this hook to keep a reference
to the environment. It only reads and copies it; it never steps it. No new hook
was added, and nothing changes for the other arbiters, which do not have `attach`.

The runner still calls `env.observe(noise_rng)` at every step, exactly as before,
and the oracle never receives `noise_rng`. So every method keeps the same
perception-noise samples on the same seed (test
`test_oracle_does_not_consume_noise_samples`). Because the oracle ignores the
noisy observation, its results are the same at every noise level and V2X setting
(test `test_reproducible_and_ignores_noise`).

## Decision structure

As for every arbiter: at each 0.2 s step, if conflict groups exist, the oracle
decides for the CAVs in those groups; CAVs outside every group drive with the
base default (FASTER); then the safety layer is applied. Each group is planned
on its own. Which mode is used depends on the group size K:

| | Mode A: passing-order enumeration | Mode B: 4 s option search |
|---|---|---|
| Used when | K ≤ `max_enum_cavs` (5) | K > `max_enum_cavs` (e.g. dense scene, up to 8 CAVs) |
| Candidates | every passing order of the group's CAVs (K!), each with "wait for earlier human drivers" and "go before them": 2·K! (2, 4, 12, 48, 240) | the what-if method's candidates: one per CAV going first (with the CAVs queued ahead of it in its lane), each with and without waiting for earlier human drivers, plus "everyone yields": 2K + 1 |
| Evaluated on a copy | until every CAV of the group has cleared its conflict points (+ 1.5 s), cap 20 s | held for 4 s (20 decision steps) |
| Cost (3rd key) | total clearing time: sum over the group's CAVs (total delay, as Xu et al.) | minus the progress: metres driven by the group's CAVs in 4 s |
| Re-planned | every 2 s, and at once when the group changes | every step |
| Exhaustive? | yes, over passing orders | no |

A group with a single CAV (a conflict with human-driven vehicles only) has one
order, but still two candidates: wait for the human drivers that arrive earlier,
or go before them. Both are simulated, so the oracle chooses.

### Human drivers: wait for them, or go before them?

Human drivers cannot be ordered: the oracle does not control them, and in
HighwayEnv they do not give way to crossing traffic. What the oracle can choose is
whether the CAVs wait for them. Every candidate therefore comes in two versions:

* **wait (FCFS-like):** besides the order, every CAV also waits for a human-driven
  vehicle that would reach their first shared conflict point ahead **no later**
  than the CAV, and that has not cleared it yet. Arrival times use the true states
  and the same formula as the detector and FCFS: TTCP = d / max(speed, 1 m/s),
  both vehicles within the 50 m reaction range;
* **go:** the CAVs follow the order only, as in Xu et al. (all-CAV traffic).

The copies decide which is better, with the same choice rule. We did this rather
than always waiting (the simple option) for two reasons. First, the oracle can
then never be worse than "FCFS towards human drivers" just because it did not try
it: the FCFS-like version is always one of the candidates, and it wins all ties.
Second, waiting is not always the safer choice. A CAV that is already close to the
point and fast may be safer going through than braking in front of it, and only
the copy shows which is true. Choosing the version per human driver instead of
per candidate would multiply the candidates by 2 to the power of the number of
drivers, which is too slow (Case-2-like scenes have 4 drivers).

### The order-following rule (mode A)

Used both in the copies and in the real simulation, once an order is chosen:

* the first CAV of the order drives FASTER;
* every next CAV yields while an **earlier** CAV of the order has not cleared a
  conflict point that the two share, then drives FASTER. Yielding is SLOWER, or
  IDLE once the CAV is nearly stopped (speed below 1 m/s and target speed 0).
  Since the lowest target speed is 0, both actions end in the same stop;
* conflict points are the `ConflictDetector`'s meeting points of the two routes.
  A CAV "has cleared" a point once it is `passed_margin` (4.5 m, the detector's
  value) past it, or has left the scene. A CAV that has already reached the
  point does not wait (it can no longer stop before it);
* a CAV that has left the scene is dropped from the order;
* in the "wait" version of a candidate, every CAV also waits for earlier human
  drivers (previous section). In mode B the moving CAVs do the same; the others
  yield anyway.

The rule waits for **every** earlier CAV that shares a conflict point, not only
the one just before it. With only the previous one, in the order (A, C, B) where
A and C do not cross, B would wait for C but not for A. With this rule, the order
means what it says: nobody enters a conflict point before the CAVs placed before
it have left it.

The same rule decides when a CAV of the group "has cleared its conflict points"
for the cost: it is past (by 4.5 m) every meeting point with the group's CAVs and
with the human-driven vehicles in conflict with the group, or it has left.

### The choice rule

The candidates are compared lexicographically (`OracleConfig.choice`):

1. no collision of a CAV of the group;
2. fewest severe conflicts (PET < 1.5 s), counted with the **same** PET code as
   the episode metrics (`cavbench.metrics.pet_values`) on the copy's trajectory.
   That trajectory includes the episode so far, which is the same for every
   candidate. After all CAVs have cleared, the copy runs 1.5 s more, so that a
   vehicle entering a conflict area just behind is still counted;
3. smallest cost (mode A: total clearing time; mode B: minus the progress); a CAV
   still not cleared at the cap counts 20 s;
4. ties: the order closest to the arrival (TTCP) order, then the "wait for human
   drivers" version, i.e. first come, first served. In mode B: the CAV going first
   that arrives earliest, with "everyone yields" last. Deterministic.

**Pruning (speed-up, same result).** Orders are evaluated in arrival order first.
Once a candidate with no collision and no severe conflict is known, an order whose
total clearing time is already larger, counting each CAV not yet cleared at the
current time, can no longer win, so its evaluation stops early. The choice is
identical to the one without pruning (test `test_pruning_does_not_change_the_choice`).

### Plan once, then follow (mode A)

The chosen order, with its human-driver version, is stored for the group and
followed at every step with the order-following rule. It is re-planned every `replan_period` = 2 s (as Xu et al.),
and at once when the group changes (a CAV joins or leaves it, including a CAV that
has cleared and left the group).

## Parameters (`OracleConfig`, `params: {oracle: {...}}` in a config)

| Parameter | Default | Meaning |
|---|---|---|
| `max_enum_cavs` | 5 | mode A up to this group size, mode B above |
| `max_eval_time` | 20 s | cap of one order evaluation (counted in `oracle_cap_reached`) |
| `replan_period` | 2 s | mode A re-planning period |
| `option_horizon` | 4 s | mode B: how long each option is held |
| `post_clear_time` | 1.5 s | extra simulated time after everyone cleared (for PET) |
| `pet_threshold` | 1.5 s | severe conflict, as in the metrics |
| `stop_speed` | 1 m/s | "nearly stopped" for IDLE |
| `choice` | (collision, severe, cost) | lexicographic keys; ties → arrival order |
| `prune` | True | stop evaluating orders that can no longer win |

## Metrics and log

Extra CSV columns (0 for the other methods):

* `oracle_orders_evaluated`: mode A candidate evaluations (pruned ones included);
* `oracle_options_evaluated`: mode B candidate evaluations;
* `oracle_enum_share`: share of plans made in mode A (single-CAV groups included);
* `oracle_replans`: number of plans (mode A and mode B);
* `oracle_cap_reached`: order evaluations that reached `max_eval_time`;
* `oracle_max_group_size`: largest group seen.

`decision_ms` is measured as usual. It is large, which is expected: the oracle is
an upper bound, not a real-time method.

The arbiter log (`run_episode(..., keep_log=True)`) keeps one entry per group and
decision step, for a later "decision quality" metric:

* `time`, `group` (the group's CAV ids), `actions` (the action of each CAV);
* `mode`: `enum_plan` (new order), `enum_follow` (stored order followed),
  `option` or `fallback` (mode B);
* `order`: the passing order followed (mode A); `goer`: the CAV going first (mode B);
  `hdv_yield`: True if the CAVs wait for earlier human drivers;
* `evaluations` (when a plan is made): per candidate, collision, severe conflicts,
  cost or progress, and whether it was pruned or reached the cap.

`run_episode` also returns `oracle_other_crash_evals` (not a CSV column): the
number of evaluations in which a CAV outside the planned group crashed in the copy.

## Limitations

* **Not a proven global optimum.** The candidates follow a fixed order-following
  rule: drive FASTER, or yield by braking to a stop. A better trajectory for the
  same order may exist (for example slowing down early instead of stopping), and
  it is not searched.
* **One group at a time.** Each conflict group is planned on its own. In the copy,
  the CAVs of the **other groups follow their own current plan** with the same
  order-following rule. This is their stored order, or, for a group not planned
  yet, the arrival order with waiting for earlier human drivers; a mode B group's
  latest choice is written as an order (its moving CAVs first). CAVs in **no
  group** drive with the base default (FASTER), as they do in reality, but in
  reality they would be planned as soon as they enter a conflict, and in the copy
  they are not. Other plans are kept fixed during an evaluation and are not
  re-planned in the copy. So the "known future" is exact only for the group being
  planned; the rest of the scene in the copy is a good approximation, not the
  exact future.
* **Crashes of other vehicles in the copy.** For the reason above, a CAV outside
  the planned group can still crash in a copy. In the sanity check (tuning seeds,
  noise none) this happened in 207 evaluations in 20 intersection episodes,
  1840 in 8 dense episodes and 19 in 8 merge episodes. So the rule "a collision
  counts for a candidate only if a CAV of the group crashes" is still needed.
  Such a crash does not stop the copy: the crashed cars stay on the road as
  obstacles. It can still delay the group or block it in a way that would not
  happen in reality.
* **Human drivers: one choice per candidate.** The CAVs either wait for every
  earlier human driver or go before all of them. Mixed choices (wait for this
  driver, go before that one) are not searched.
* **Mode B is not exhaustive.** For groups larger than 5 CAVs only 2K + 1 options
  are compared, over 4 s; the oracle is weaker there.
* **Re-planning.** Between re-plans (2 s) the stored order is followed even if the
  scene changes, unless the group changes.
* **Same at every noise level.** By design: it ignores the noisy observation.

## Sanity check

Oracle vs FCFS vs UCRA, **tuning seeds only** (100000 and up), noise none, lab PC
with 15 episodes in parallel (so seconds per episode are slower than a single
run). "PET-viol. eps" = episodes with at least one severe conflict (PET < 1.5 s).

| Scene | Method | Episodes | Success | Collisions | Time-outs | PET-viol. eps | Mean min PET [s] | Mean speed [m/s] | Decision ms mean / p95 | s / episode |
|---|---|---|---|---|---|---|---|---|---|---|
| intersection | FCFS | 20 | 100 % | 0 | 0 | 0 | 2.92 | 5.68 | 3 / 12 | 1 |
| intersection | UCRA | 20 | 95 % | 0 | 1 | 1 | 2.06 | 5.43 | 50 / 102 | 4 |
| intersection | Oracle | 20 | 100 % | 0 | 0 | 0 | 2.98 | 5.92 | 758 / 2886 | 40 |
| intersection_dense | FCFS | 8 | 100 % | 0 | 0 | 0 | 2.67 | 4.57 | 4 / 15 | 2 |
| intersection_dense | UCRA | 8 | 100 % | 0 | 0 | 1 | 1.84 | 4.31 | 88 / 169 | 10 |
| intersection_dense | Oracle | 8 | 100 % | 0 | 0 | 0 | 2.68 | 4.68 | 7048 / 32199 | 580 |
| merge | FCFS | 8 | 100 % | 0 | 0 | 0 | 2.05 | 6.02 | 1 / 2 | 1 |
| merge | UCRA | 8 | 88 % | 0 | 1 | 0 | 2.22 | 6.39 | 60 / 103 | 7 |
| merge | Oracle | 8 | 100 % | 0 | 0 | 0 | 1.80 | 8.00 | 854 / 3712 | 41 |

The oracle is at least as good as FCFS on safety in every scene (no collision, no
PET violation) and faster (higher mean speed). Before the human-driver candidates
were added, it had a PET violation in 3 of 8 intersection episodes. In the merge
scene its mean minimum PET is lower than FCFS's (1.80 s vs 2.05 s), still above
1.5 s: it lets vehicles pass closer, which gives the higher speed (8.0 vs 6.0 m/s).
At noise none FCFS is already very strong in these scenes, so the oracle's margin
over it is small here; the difference is expected to grow with noise, which the
oracle ignores.

## Running it

```bash
python experiments/run_experiment.py configs/oracle_smoke.yaml --workers 6   # tuning seeds 100000-100002
python experiments/watch.py --arbiter oracle --seed 3                        # watch one episode
python experiments/run_experiment.py configs/e1_oracle.yaml --workers 8      # full E1 (long)
```

`configs/e1_oracle.yaml` uses the same name, scenarios, noise levels, episodes and
seeds as `configs/e1_baselines.yaml`, so its rows are appended to
`results/e1_baselines.csv` next to the other methods. The runner appends and skips
rows already in the file; it never overwrites it. If that file was written before
the oracle columns existed, the runner keeps appending in the file's own format
and prints a note: the `oracle_*` columns are then not saved, but all the common
metrics are.
