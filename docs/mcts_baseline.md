# The MCTS baseline: our re-implementation vs Lin et al. (2025)

CAVBench's MCTS baseline (`cavbench/arbiters/mcts.py`) re-implements

> Z. Lin, J. Lan, C. Anagnostopoulos, Z. Tian, D. Flynn, "Safety-Critical Multi-Agent
> MCTS for Mixed Traffic Coordination at Unsignalized Intersections", *IEEE Trans.
> Intelligent Transportation Systems*, vol. 26, no. 11, pp. 18992–19005, Nov. 2025,
> doi:10.1109/TITS.2025.3598727.

The authors' code (MATLAB 2024a) is not public. We implemented the method from
the accepted manuscript (University of Glasgow repository, eprints.gla.ac.uk/362388,
CC BY). Equation numbers below refer to that paper.

There are two versions:

* kind `mcts_calibrated`, the **main baseline** (label "MCTS" in E1). It is the
  strict version plus two calibrated values that the paper does not fix (see
  "Calibration" below).
* kind `mcts`, the **strict** version (label "MCTS-strict"): Table I plus the
  choices in the table below, nothing tuned.

Both run the same algorithm (Algorithm 1) with the same Table I values.

## What the paper does

A central planner chooses the actions of all automated vehicles (AVs) together
with a Monte Carlo tree search (Algorithm 1).

* **Tree nodes.** Each node holds a joint action of all AVs. Its children are all
  joint actions (Eq. 20: 5 accelerations × 5 steering rates per AV).
* **Safety check.** When a node is expanded, each child is marked safe "o" or
  unsafe "x" (Eq. 22, 24). Three checks are used:
  * V2V risk between two AVs (Eq. 7–12): a dynamic safe distance, an
    instantaneous risk and a temporal risk.
  * V2H risk between an AV and a human-driven vehicle (HDV) (Eq. 13–19): the
    HDV is predicted with IDM plus a Gaussian uncertainty that depends on its
    driving style, and the collision probability C_ih is added.
  * V2R: the distance to the road boundary (Eq. 6).
* **Search.** Only safe children are selected (UCB, Eq. 25). Rollouts follow
  random safe joint actions (Eq. 26–27).
* **Value.** Values are updated with Eq. 28. The reward combines safety,
  efficiency, comfort and cooperation (Eq. 29–35).
* **Decision.** The executed action is the root child with the largest
  N + ε·Q (Eq. 41).
* **Settings.** All parameters are in Table I: K = 1000 iterations, depth 8,
  c_exp = 1, γ_r = 0.95, β_dec = 0.75, and so on.

## What we implemented

For each conflict group found by CAVBench's shared `ConflictDetector`, the
arbiter runs one tree search over the joint actions of the group's CAVs. Each
CAV chooses SLOWER, IDLE or FASTER, so a group of K CAVs has 3^K joint actions.
The search follows Algorithm 1 with the Table I values.

**The scene inside the tree.** It contains:
* the group's CAVs;
* every vehicle within 60 m of one of them;
* the HDVs in conflict with the group.

**How the vehicles move inside the tree.** Every vehicle moves along its own
planned route (1-D, arc length), in steps of 0.1 s:
* the group's CAVs track the target speed chosen by their action, with
  HighwayEnv's controller (gain 1/0.6, acceleration between −5 and 3 m/s²);
* other CAVs keep their current target speed;
* HDVs follow IDM, with the paper's driving-style scaling (Eq. 14). Their
  leader is a vehicle ahead on a shared stretch of lane, as in UCRA's rollouts.

**After the tree search.** The arbiter returns the first joint action of the
best child. As for every CAVBench arbiter, the shared car-following safety
layer (`FollowingSafety`) is applied afterwards.

## Adaptations and values we had to choose

| # | Item | Paper | CAVBench | Reason |
|---|---|---|---|---|
| 1 | Actions | 5 accelerations × 5 steering rates per AV (Eq. 20) | SLOWER / IDLE / FASTER per CAV (target speed one level down / same / up); 3^K joint actions | the benchmark's action space, the same for every arbiter; vehicles follow their lanes, so there is no steering |
| 2 | Who is planned jointly | all AVs in the scene | the CAVs of each conflict group (one tree per group); the CAVs that are not in a group drive FASTER | the same interface as the other arbiters (`arbitrate(obs, groups)`); it also keeps 3^K manageable |
| 3 | Information | exact states | the same noisy, possibly delayed shared states as every arbiter, **treated as exact**: σ_pos, σ_vel and the V2X message age are ignored | intentional: the paper has no model of sensor or V2X noise, and this is the difference we study |
| 4 | Tree step | not stated | Δt_tree = 0.5 s, d_max = 8 → 4 s look-ahead (`dt_tree`, `d_max`) | 4 s equals the horizon of UCRA's rollouts |
| 5 | Motion model | 2-D kinematic model (Eq. 3–4) | 1-D along the planned route, positions and headings from the route polyline, inner step 0.1 s | vehicles follow lane centres in CAVBench |
| 6 | Rollout policy | α·π_model + (1−α)·π_random, with a learned value predictor (Eq. 26) | α = 0: uniform over the **safe** joint actions at every step, or uniform over all of them if none is safe | we have no learned predictor. Eq. 27 writes U(Π_joint) and Eq. 26 writes U(Π_safe); we use the safe set (Eq. 26) |
| 7 | UCB | printed as Q_n/N_n + c_exp·√(ln N_n0 / N_n) (Eq. 25) | Q_n + c_exp·√(ln N_parent / N_n); +∞ if N_n = 0; only "o" children | Q_n is already a running mean (Eq. 28), so dividing it by N_n again would count the visits twice; standard UCT uses the parent's visits. Ties go to the lowest index |
| 8 | ε in Eq. 41 | ε ∈ (0, 1), value not stated | ε = 0.5 | we assume it is Table I's "policy balance factor" η = 0.5 |
| 9 | No safe root child | not covered | the child with the lowest maximum risk ratio (Q_risk / Q_th over all pairs); counted in `mcts_no_safe_root` | the paper has no fallback; we add only this one, and count it |
| 10 | Tree reuse (Eq. 43) | the chosen subtree is kept | rebuilt from scratch every 0.2 s (`reuse_tree=False`; `True` is not implemented) | the scene is re-observed, with noise, every 0.2 s, and the tree step (0.5 s) is not the decision step (0.2 s) |
| 11 | Iteration that reaches depth d_max, or a node whose children are all "x" | Algorithm 1 leaves the iteration without an update (the while loop ends, or "Terminate", Eq. 23) | the node's value is still backed up: at depth d_max with the discounted rewards of its path; with all children unsafe, with a rollout from the node | without an update, UCB never changes and every later iteration would follow the same path without effect |
| 12 | Pairs that are checked and rewarded | all AV–AV and AV–HDV pairs | pairs of a group CAV and a vehicle **whose route meets its own** (crossing, merging or same lane) | vehicles on lanes that never meet cannot collide in CAVBench. Otherwise a car passing in the adjacent opposite lane (2 m clearance, high relative speed, d_safe ≈ 8 m) would mark every action unsafe |
| 13 | Distance d_ij | minimum distance between the four vertices of the two vehicles (Eq. 1) | clearance between the 5 m × 2 m cars, each covered by three circles of radius 1 m | the footprint model already used by UCRA's rollouts |
| 14 | Δv_ij | "relative velocity", written \|Δv\| (Eq. 7–8) and ‖Δv‖ (Eq. 9) | norm of the relative velocity **vector** ‖v_i − v_j‖ | the norm in Eq. 9 suggests a vector. With the plain speed difference, two cars crossing at the same speed would have Δv = 0 and a collision would pass the V2V check (Q ≤ 0.6·1 + 0.4·0.29 < 0.8) |
| 15 | v_ref (Eq. 8) | not stated | top CAV target speed (9 m/s intersection, 12 m/s merge) | the speed scale of the scenario |
| 16 | Conflict zone, Eq. 5 | Ω_int: \|x\|, \|y\| ≤ R_EX; Ω_app: \|x\| ≤ 2R_EX **or** \|y\| ≤ 2R_EX, around the intersection centre | intersection: centre (0, 0), R_EX = 11 m (HighwayEnv's junction square: 9 m turn radius + half a 4 m lane). Merge: centre at the merge point (200, 0), R_EX = 11 m | the junction square is the area where routes cross. At the merge, the ramp is within about 2 m of the main lane over its last 11 m, which is where car bodies can overlap. The "or" in Ω_app is kept as printed, so every vehicle on an approach road is in Ω_app |
| 17 | α_3 for a pair | "1 when vehicles are in the zones" | a zone counts if **at least one** of the two vehicles is in it | our reading; the more cautious choice |
| 18 | T_p, ρ and R_Tp (Eq. 10–11) | T_p = 5.0 (Table I) | 5 predicted points, 1 s apart (t = 1…5 s, weight 1/(1+t)); constant speed along the route; ρ uses the predicted distance at each t and the current d_safe | T_p is given but its unit and step are not; 1 s steps give the 5 s horizon of "5.0" |
| 19 | AV prediction in C_ih | constant acceleration | the node's state, which is the tree's own prediction | the tree already predicts the CAVs |
| 20 | HDV prediction for C_ih (Eq. 13–18) | Gaussian around the IDM prediction, Σ_h(t, η_h) | the HDV's state in the node (IDM inside the tree), with Σ_h at t = depth × 0.5 s (Eq. 15–16, Table I values; ε_x, ε_y not scaled by style, since no formula is given); samples bounded as in Eq. 17; C_ih = share of 20 samples (`n_samples`) closer than d_safe | Monte Carlo estimate of the integral in Eq. 18. The same 20 standard-normal draws are used for all nodes of one decision (from the arbiter's seeded RNG) |
| 21 | d_safe in C_ih | "style-dependent safety threshold" | Eq. 7 with the sampled HDV speed and heading, × (1 − α_d·η_h) | Eq. 14 gives the style scaling of the safety distance |
| 22 | Driving style η_h | known per HDV (sampled from Beta(2, 2) in the paper's Case 2) | η_h = 0.5 for every HDV (also used in the IDM scaling of Eq. 14) | CAVBench does not give drivers' styles to arbiters; 0.5 is the mean of the paper's Beta(2, 2) |
| 23 | V2R (Eq. 6) | distance to road boundaries ≥ d_min = 1 m | always satisfied; Q_v2r = 0, w_v2r = 0 | on a lane centre a 2 m wide car is exactly 1 m from each edge of a 4 m lane, and actions are longitudinal only |
| 24 | Sign of the risk terms in the reward (Eq. 29) | Q_v2v = Σ(φ_v2v + λ_T·R_Tp), Q_v2h = Σ(φ_v2h + λ_T·R_Tp + λ_c·C_ih) | **−** λ_T·R_Tp and **−** λ_c·C_ih (`risk_as_penalty=True`; `False` gives the printed signs) | as printed, more risk would *raise* the safety reward, while φ is a penalty (−1). We read the "+" as part of the sum of penalties |
| 25 | w_v2v, w_v2h, w_v2r (Eq. 29) | not stated | 1.0, 1.0, 0 | equal weights; V2R is always satisfied (row 23) |
| 26 | μ_s, κ_1…κ_4, η_1, η_2 (φ_v2h) | not stated | μ_s = 1; κ_1 = 1, κ_2 = κ_3 = κ_4 = 0; η_1 = η_2 = 0.2 | κ_1 = 1 makes the penalty below d_th equal to φ_v2v (−1); η like β_1, β_2 |
| 27 | Efficiency (Q_vel, Q_acc, Q_ref) | Table I weights | v_des = top CAV target speed; a_des = 0; acceleration = mean acceleration over the tree step; Q_ref = 0; Q_eff = w_vel·Q_vel + Q_acc | lane following, so there is no reference-path error |
| 28 | Comfort (Eq. 31) | −w_jerk·\|ȧ\|² − w_yaw·\|θ̈\|² | jerk = change of the mean acceleration between tree steps / 0.5 s (0 before the first step); yaw term 0 | no steering |
| 29 | w_ij (Eq. 32) | not stated | 1 / (number of other simulated vehicles) | equal weights that sum to 1 |
| 30 | Q_safety_j of other vehicles in Q_other | from all their interactions | from their pairs with the group's CAVs (row 12) | only those pairs are computed |
| 31 | α_wait, v_thres (Eq. 33) | not stated | α_wait = 0.1 per s, v_thres = 1.0 m/s; T_wait = seconds the CAV has been below v_thres in this episode (noisy speeds; reset each episode); kept fixed inside the tree | the paper sums an indicator over time steps; we use seconds so that α_wait does not depend on the step |
| 32 | λ_i | λ_base = 0.5 (Table I) | 0.5·exp(−α_wait·T_wait) | Eq. 33 |
| 33 | HDV IDM base values | not stated | a = 3 m/s², b = 5 m/s², T = 1.5 s, s0 = 2 m, δ = 4, desired speed 8 m/s (UCRA's `RolloutConfig`), then style-scaled (Eq. 14, α_a = 0.3, α_T = 0.4, α_d = 0.3) | the same human-driver model as UCRA's rollouts |

All other values are Table I's: β_1 = β_2 = 0.2, κ_v = 0.3, λ_v = 0.3, d_base = 1.5 m,
w^v2v = (0.6, 0.4), w^v2h = (0.4, 0.3, 0.3), Q_th^v2v = 0.8, Q_th^v2h = 0.6,
λ_T = 0.4, λ_c = 0.3, the HDV uncertainty values (σ_x = σ_y = 0.1, ε = 0.05,
σ_v = 0.2, σ_θ = 0.1, ρ_xv = ρ_yθ = 0.3, β_x = β_y = 0.5, β_v = 0.6, β_θ = 0.4),
w_1, w_2, w_3 = 0.5, 0.3, 0.2, w_jerk = 0.1, w_vel = 0.4, w_acc = 0.3,
α_v = 1.5, β_v = 1.0 and v_tol = 2.0. Every number is a field of `MCTSConfig`,
marked "(Table I)" or "(our choice)", and can be overridden from a YAML
config (e.g. `params: {n_iterations: 200}`).

## A property of the paper's V2V check (not a change)

With the formulas as printed, the V2V check does **not** reject every overlap.

* Two overlapping cars (d = 0) give r = 1 + λ_v·‖Δv‖/v_max (Eq. 9).
* R_Tp is at most (1/5)·Σ_{t=1..5} 1/(1+t) ≈ 0.29 (Eq. 10).
* So Q_v2v ≤ 0.6·(1 + 0.3·‖Δv‖/v_max) + 0.4·0.29, which is below the 0.8 threshold
  when the relative speed is small (about ‖Δv‖ < 4 m/s at v_max = 9 m/s).

The tree therefore relies partly on the reward (φ_v2v = −1 below d_safe) to avoid
these cases. The V2H check is stricter, because a sure collision has C_ih = 1:
0.4 + 0.3 > 0.6. This is reported as part of the baseline, not corrected.

## Observed behaviour: the reward balance

In the smoke test, the intersection episode with seed 2 ends in a CAV–CAV
collision at both noise levels. No coordination also collides on this seed, while
FCFS and UCRA succeed.

**Cause.** With Table I's weights, the comfort and efficiency terms of the reward
are much larger than the safety term. Example for one CAV over one 0.5 s tree step:

| Option | Terms | Q_self |
|---|---|---|
| Braking from 9 m/s to target 4.5 m/s (mean acceleration about −4.6 m/s²) | Q_acc = −0.3·4.6² ≈ −6.3; jerk ≈ 9 m/s³, so Q_comfort ≈ −8.5 | ≈ 0.3·(−0.9 − 6.3) + 0.2·(−8.5) ≈ **−3.9** |
| Keeping speed towards a collision | φ_v2v = −1 (Eq. 30), small R_Tp | ≈ 0.5·(−1.04) ≈ **−0.5** |

So the search avoids braking, and only the safety check prevents crashes. That
check is lenient at low relative speed (see the previous section).

**Fixed by the paper, not by us:** Table I's weights (w_1, w_2, w_3, w_acc, w_jerk)
and the size of φ_v2v (−1).

**Our choices that matter here** (see the table):
* w_v2v = w_v2h = 1;
* the jerk definition (the change of the mean acceleration between 0.5 s tree steps);
* SLOWER/FASTER change the target speed by 4.5 m/s, and the env's CAV braking
  limit is 5 m/s², while the paper's actions are at most ±3 m/s².

The strict version (kind `mcts`) keeps these values. The calibrated version
re-scales them as documented in "Calibration" below.

## Calibration

### Why

On the paper's Case 1 (4 CAVs, one per approach, no human drivers; the paper
reports 0 % collisions), the strict version collided in 9 of 20 episodes and timed
out in 2 (`configs/mcts_case1_check.yaml`, tuning seeds 100000–100019, noise none,
ideal V2X). FCFS and UCRA succeeded in all 20.

The diagnosis of every collision found **no bug**:

* every chosen option had passed the safety check as the paper defines it;
* a copy of the arbiter's own `evaluate` gave identical totals.

The causes:

* **Our adaptation (main cause).** The paper's actions are accelerations of at
  most ±3 m/s², with a medium value of ±1.5 m/s². Our SLOWER changes the target speed
  by 4.5 m/s, and the speed controller then brakes at about −4.8 m/s² on average over
  a 0.5 s tree step. With Table I's w_acc and w_jerk, one such step costs about −4 per
  CAV, against about −0.5 to −1 for driving on towards a collision (φ_v2v = −1).
  Braking therefore always looked worse to the search.
* **The two time-outs** have the same cause. CAVs that had stopped to give way
  stayed stopped for the rest of the episode, because the cost of starting again
  (+3 m/s², a jump in jerk) was larger than the speed penalty for standing.
* **Values the paper does not give** (w_v2v, the T_p point spacing, the jerk
  definition) set how strongly the safety term counts.
* **The paper's V2V check is lenient.** It lets an overlap through when ‖Δv‖ is
  moderate (section above). Our crashes passed it with Q_v2v = 0.72–0.80.

### The rule

* Never change a Table I value or Algorithm 1.
* Change only values tied to our action adaptation, or values the paper does not
  give, in this order (smallest change first):
  1. a comfort scale;
  2. w_v2v and w_v2h (1, 2, 5, 10), the T_p point spacing (1.0 → 0.5 s), and the jerk
     definition.
* Use only the tuning seeds 100000–100019, never the evaluation seeds.
* Use a greedy search:
  * stop at the first setting that meets the target on the Case 1 check: at most
    1 collision in 20 (paper: 0 %), at least 90 % success (paper: 94.1 %), and as few
    PET-violation episodes as possible;
  * among settings with the same number of changes, prefer higher success, then
    higher mean speed.

### The comfort scale (formula)

`MCTSConfig.a_med` (None in the strict version) switches it on:

    comfort_scale = (a_med / a_step)^2
    Q_acc  = -comfort_scale · w_acc  · acc^2
    Q_jerk = -comfort_scale · w_jerk · jerk^2

* **a_med:** a_med = 1.5 m/s², Table I's medium acceleration a_med (the paper's action set is built from it, Eq. 20).
* **a_step:** a_step is the mean |acceleration| of one of our SLOWER tree steps
  (`slower_step_acceleration`). The CAV starts steady at a target-speed level and is
  sent to the level below, using the tree's own speed controller (kp = 1/0.6,
  bounds [−5, 3] m/s², 5 × 0.1 s). The result is averaged over the pairs of
  adjacent levels:
  * intersection (0 / 4.5 / 9 m/s): a_step = 4.83 m/s², so comfort_scale = 0.096;
  * merge (0 / 6 / 12 m/s): a_step = 5.0 m/s², so comfort_scale = 0.090.

Scaling the squared terms by (a_med / a_step)² is the same as measuring our
accelerations and jerks in units where one of our coarse steps equals the paper's
medium acceleration. One SLOWER step then costs what a_med would cost in the paper.
Table I's w_acc, w_jerk, w_1, w_2 and w_3 are unchanged.

### Settings tried (Case 1 check, 20 episodes each, tuning seeds)

| # | Setting | Changes | Success | Collisions | Time-outs | PET-violation episodes | Mean min PET [s] | Mean speed [m/s] | Decision time mean / p95 [ms] |
|---|---|---|---|---|---|---|---|---|---|
| 0 | strict (`mcts`) | 0 | 45 % | 9 | 2 | 7 | 1.04 | 6.89 | 5454 / 11495 |
| 1 | comfort scale | 1 | 90 % | 2 | 0 | 8 | 1.48 | 7.43 | 9887 / 18815 |
| 2 | comfort + w_v2v = 2 | 2 | 95 % | 0 | 1 | 2 | 1.96 | 6.88 | 6705 / 12734 |
| 3 | **comfort + w_v2v = 5** | 2 | **100 %** | **0** | **0** | **0** | 2.30 | 6.85 | 5693 / 11693 |
| 4 | comfort + w_v2v = 10 | 2 | 95 % | 0 | 1 | 0 | 2.57 | 6.29 | 5025 / 10852 |
| 5 | comfort + T_p spacing 0.5 s | 2 | 95 % | 1 | 0 | 8 | 1.47 | 7.33 | 8813 / 17102 |
| 6 | comfort + jerk = change per step (`jerk_def: delta`) | 2 | 80 % | 4 | 0 | 8 | 1.46 | 7.37 | 8459 / 15200 |

Notes on the search:

* Step 1 (comfort scale alone) missed the target by one collision.
* In step 2, settings 2–5 meet it. Setting 3 has the highest success and no
  PET-violation episode, so it is chosen; no combination was needed.
* w_v2h was not varied, because Case 1 has no human drivers and w_v2h cannot change
  anything there. It stays at 1.
* Mean min PET is over the episodes that have a PET pair.

### Chosen values (`MCTS_CALIBRATED`, kind `mcts_calibrated`)

| Value | Strict | Calibrated | Type |
|---|---|---|---|
| a_med (comfort scale) | off | 1.5 m/s² = Table I's a_med (scale 0.096 intersection, 0.090 merge) | fixes our action adaptation (row 1) |
| w_v2v | 1 | 5 | not given in the paper (row 25) |

Everything else is identical, including all Table I values, w_v2h = 1, the T_p
spacing of 1 s and the jerk definition. The test `test_strict_kind_unchanged`
checks that kind `mcts` still takes exactly the actions of the version before the
calibration.

### Final Case 1 results (tuning seeds 100000–100019)

| Method | Success | Collisions | Time-outs | PET-violation episodes | Mean min PET [s] | Mean speed [m/s] | Decision time mean / p95 [ms] |
|---|---|---|---|---|---|---|---|
| MCTS (calibrated) | 100 % | 0 | 0 | 0 | 2.30 | 6.85 | 5693 / 11693 |
| MCTS-strict | 45 % | 9 | 2 | 7 | 1.04 | 6.89 | 5454 / 11495 |
| FCFS | 100 % | 0 | 0 | 0 | 3.29 | 6.50 | 3 / 13 |
| UCRA | 100 % | 0 | 0 | 0 | 2.56 | 5.87 | 72 / 147 |

The paper's Case 1 numbers are 0 % collisions and 94.1 % success. The calibrated
version reproduces them on these seeds.

### Sanity check with human drivers (tuning seeds 100000–100019)

This run used the normal `intersection` scenario (CAVs and human drivers, spawning
on), 20 episodes per row. It is not used for choosing values, only to see whether
the calibrated version also behaves better when human drivers are present.

| Noise | Method | Success | Collisions | Time-outs | PET-violation episodes | Mean speed [m/s] | Decision time mean / p95 [ms] | No-safe-root decisions |
|---|---|---|---|---|---|---|---|---|
| none | MCTS (calibrated) | **80 %** | 2 | 2 | 2 | 5.80 | 25725 / 69041 | 261 |
| none | MCTS-strict | 50 % | 5 | 5 | 6 | 5.51 | 27883 / 82791 | 42 |
| none | FCFS | 100 % | 0 | 0 | 0 | 5.68 | 7 / 30 | – |
| none | UCRA | 95 % | 0 | 1 | 1 | 5.43 | 139 / 263 | – |
| high | MCTS (calibrated) | **65 %** | 6 | 1 | 3 | 5.68 | 26221 / 73765 | 129 |
| high | MCTS-strict | 60 % | 6 | 2 | 7 | 5.59 | 26253 / 86858 | 101 |
| high | FCFS | 100 % | 0 | 0 | 0 | 5.58 | 7 / 29 | – |
| high | UCRA | 95 % | 1 | 0 | 0 | 5.46 | 137 / 273 | – |

What the sanity check shows:

* **Better than strict.** With human drivers, the calibrated version is better than
  the strict one: +30 points of success without noise, +5 with high noise, fewer
  time-outs and fewer PET violations.
* **Still behind FCFS and UCRA.** It is clearly worse than both, especially with high
  noise. The MCTS treats the noisy shared states as exact (row 3).
* **The V2H side is now the limit.** It has many decisions with no safe root child
  (V2H pairs). That side could not be calibrated on Case 1, since Case 1 has no human
  drivers, so w_v2h stays at 1.

The calibration was fixed before this check and was not changed after it.

## Speed-ups that do not change the algorithm

* **Shared motion of independent vehicles.** Vehicles whose motion cannot depend on
  the joint action move the same way in every branch. These are the other CAVs and
  every HDV that is not queued behind a group CAV. They are simulated once per
  decision, and the tree only re-simulates the group's CAVs and the HDVs that follow
  them. The numbers are identical; this was checked with a fingerprint of 34
  decisions.
* **Batched safety checks.** All children of a node are safety-checked in one
  NumPy batch.
* **Rollout batches.** A rollout step checks the joint actions of a random order in
  batches of 1, 2, 4, … and takes the first safe one. This is exactly "uniform over
  the safe set" (`rollout_first`, `rollout_chunk` change only the speed).

## Metrics (per episode, in the CSV)

| Column | Meaning |
|---|---|
| `mcts_iterations` | total tree iterations (1000 per conflict group per decision step) |
| `mcts_nodes` | nodes created (roots + all children of every expanded node) |
| `mcts_pruned_fraction` | share of the created nodes marked "x" |
| `mcts_no_safe_root` | decisions where no root child was safe (least-unsafe choice used) |
| `mcts_max_group_size` | largest group planned jointly (3^K children) |

`decision_ms_*` is measured by the base class, as for all arbiters.

## Run time

The search runs in Python with NumPy (the paper used MATLAB). With Table I's 1000
iterations it is much slower than the other baselines. See the timing check in
the README section "MCTS baseline (Lin et al., 2025)".
