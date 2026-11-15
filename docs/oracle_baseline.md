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
| Candidates | every passing order of the group's CAVs: K! (2, 6, 24, 120) | the what-if method's candidates: one per CAV going first (with the CAVs queued ahead of it in its lane), plus "everyone yields" |
| Evaluated on a copy | until every CAV of the group has cleared its conflict points (+ 1.5 s), cap 20 s | held for 4 s (20 decision steps) |
| Cost (3rd key) | total clearing time: sum over the group's CAVs (total delay, as Xu et al.) | minus the progress: metres driven by the group's CAVs in 4 s |
| Re-planned | every 2 s, and at once when the group changes | every step |
| Exhaustive? | yes, over passing orders | no |

A group with a single CAV (a conflict with a human-driven vehicle only) has one
order, so nothing is simulated: that CAV drives FASTER, and the safety layer still
applies.

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
* a CAV that has left the scene is dropped from the order.

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
4. ties: the order closest to the arrival (TTCP) order, i.e. first come, first
   served. In mode B: the CAV going first that arrives earliest, with "everyone
   yields" last. Deterministic.

**Pruning (speed-up, same result).** Orders are evaluated in arrival order first.
Once a candidate with no collision and no severe conflict is known, an order whose
total clearing time is already larger, counting each CAV not yet cleared at the
current time, can no longer win, so its evaluation stops early. The choice is
identical to the one without pruning (test `test_pruning_does_not_change_the_choice`).

### Plan once, then follow (mode A)

The chosen order is stored for the group and followed at every step with the
order-following rule. It is re-planned every `replan_period` = 2 s (as Xu et al.),
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

* `oracle_orders_evaluated`: mode A order evaluations (pruned ones included);
* `oracle_options_evaluated`: mode B option evaluations;
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
* `evaluations` (when a plan is made): per candidate, collision, severe conflicts,
  cost or progress, and whether it was pruned or reached the cap.

## Limitations

* **Not a proven global optimum.** The candidates follow a fixed order-following
  rule: drive FASTER, or yield by braking to a stop. A better trajectory for the
  same order may exist (for example slowing down early instead of stopping), and
  it is not searched.
* **One group at a time.** Each conflict group is planned on its own. In the copy,
  **CAVs outside the group drive with the base default (FASTER)**, while in
  reality they may be slowed by their own arbitration. So the "known future" is
  exact only for the group being planned; the rest of the scene in the copy is an
  approximation. In the dense scene, the copies show crashes of other CAVs
  driving FASTER. Such crashes do not count as a collision for the option (only a
  crash of a CAV of the group does), and the copy keeps running with the crashed
  cars as obstacles. They can still delay the group or cause a group crash that
  would not happen in reality.
* **Human drivers are not ordered.** Only the group's CAVs are permuted. A
  human-driven vehicle in conflict is simulated exactly, but the oracle cannot
  choose "this CAV waits for that human driver" except through the order of the
  CAVs and the safety layer. A group with one CAV and a human driver has no
  choice: the CAV drives FASTER.
* **Mode B is not exhaustive.** For groups larger than 5 CAVs only K + 1 options
  are compared, over 4 s; the oracle is weaker there.
* **Re-planning.** Between re-plans (2 s) the stored order is followed even if the
  scene changes, unless the group changes.
* **Same at every noise level.** By design: it ignores the noisy observation.

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
