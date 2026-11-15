# FCFS baseline (classical first-come, first-served)

Code: `cavbench/arbiters/fcfs.py` (kind `fcfs`). Tests: `tests/test_fcfs.py`.

## What it does

The conflict area is one shared resource that only one vehicle may occupy at a
time. At the intersection this is HighwayEnv's junction box (the 22 m × 22 m
square around the centre: 9 m turn radius + half a 4 m lane); at the merge it is
the same square around the merge point.

1. **Queue.** A CAV joins the queue when it comes within 50 m of the area (the
   benchmark's reaction range). Its place never changes afterwards: CAVs are
   served in the order in which they joined; among CAVs joining at the same
   step, the one closer to the area goes first. A CAV leaves the queue once its
   centre is 3 m past the area.
2. **One vehicle at a time.** Only the first CAV of the queue may enter, and only
   when no other vehicle (CAV or human-driven) is inside the area.
3. **Waiting.** The other queued CAVs approach at the middle target speed
   (4.5 m/s) and brake (comfort deceleration 3 m/s², 0.4 s reaction) to stop with
   their centre 3 m before the area.
4. **Human drivers** cannot be queued: they do not communicate and do not follow
   the order. The first CAV therefore also waits while a human-driven vehicle
   would reach the area before the CAV could clear it plus 1 s (arrival time =
   distance / max(speed, 1 m/s); clearing time with 3 m/s² acceleration). Only
   the first human-driven vehicle on each lane is considered, since vehicles
   queued behind it cannot arrive first, and vehicles on the CAV's own lane are
   left to car following.
5. A CAV that was given permission always continues until it has cleared the
   area (also one that is clearly inside it without permission, e.g. pushed in
   by its leader).

**Noise.** Positions are noisy, so a vehicle counts as inside the area from
2 σ_pos before it, and as cleared only 2 σ_pos after it: the conflict
detector's margin (`noise_k = 2`). Waiting CAVs stop that much earlier, and a
CAV that has started to stop keeps stopping until it gets permission. Without
these margins, a waiting CAV stopped at the edge of the area could be
measured "inside" and drive in, and noise made waiting CAVs alternate between
approaching and stopping (creeping forward); in a first version without them,
high noise caused CAV-CAV collisions in 40-75 % of intersection episodes
(tuning seeds), which would have made the baseline unfairly weak.

CAVs outside the control range drive at the top target speed. Like every
arbiter, FCFS uses the noisy (and possibly delayed) shared states, so noise can
change the order in which CAVs join the queue. The FollowingSafety layer is
applied afterwards, as for every arbiter.

## Why this version

FCFS is the standard rule-based baseline for unsignalized intersections: the
intersection is reserved as a whole and served in arrival order. CoDrivingLLM
(Fang et al., 2025) uses FCFS as its rule-based baseline but does not describe
it and its released code does not contain it. The behaviour it reports for FCFS
(90 % success, the lowest average speed of all methods, 2.7 m/s, and the
largest average PET, 18.1 s) is what this policy produces: safe, with large
gaps between vehicles, and slow.

An earlier CAVBench version used a per-conflict-point variant that re-decided
the order every 0.2 s and let vehicles on non-crossing routes pass together.
That variant is closer to a reservation-based scheme than to classical FCFS and
was replaced before any final experiment was run.

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `control_range` | 50 m | distance at which a CAV joins the queue |
| `half_width` | 11 m | half-width of the conflict area |
| `clear_margin` | 3 m | past the area by this much = cleared |
| `stop_offset` | 3 m | waiting CAVs stop this far before the area |
| `comfort_decel`, `reaction` | 3 m/s², 0.4 s | braking model of waiting CAVs |
| `hdv_buffer` | 1 s | margin to a human-driven vehicle's arrival |
| `min_speed` | 1 m/s | speed floor in arrival-time estimates |
| `noise_k` | 2 | position margin = `noise_k` × σ_pos |

## Check on tuning seeds

Seeds 100000-100019, 20 episodes per row; bad V2X = 0.3 s delay and 30 %
packet loss. All rows: 0 collisions.

| Scene | Noise | Success | Time-outs | Severe-conflict episodes | Mean min PET [s] | Mean speed [m/s] |
|---|---|---|---|---|---|---|
| intersection | none | 100 % | 0 | 0 | 3.81 | 3.94 |
| intersection | high | 100 % | 0 | 0 | 4.53 | 3.66 |
| intersection, bad V2X | high | 100 % | 0 | 0 | 4.49 | 3.67 |
| dense | none | 100 % | 0 | 0 | 3.64 | 2.97 |
| dense | high | 75 % | 5 | 0 | 4.39 | 2.65 |
| dense, bad V2X | high | 75 % | 5 | 0 | 4.26 | 2.68 |
| merge | none | 100 % | 0 | 0 | 2.60 | 4.90 |
| merge | high | 100 % | 0 | 0 | 3.16 | 4.41 |

The dense time-outs come from the 50 s time limit: serving eight CAVs and the
human drivers one at a time, with the larger noise margins, takes longer.

## Limitations

* One vehicle at a time is conservative by design: cars on non-crossing routes
  also wait. This is what makes the classical policy slow.
* Human-driven vehicles are handled by the arrival-time check above, not by the
  queue; they do not yield to CAVs.
* At the merge, vehicles on the same lane follow each other through the area;
  this is car following, not two reservations.
