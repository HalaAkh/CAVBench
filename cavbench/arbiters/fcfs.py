"""First-come, first-served (FCFS), classical version.

The conflict area (the junction box at the intersection, the merging area at
the merge) is treated as one shared resource that only one vehicle may occupy
at a time:

1. **Queue.** A CAV joins the queue when it comes within ``control_range``
   (50 m, the benchmark's reaction range) of the conflict area. Its place is
   fixed from then on: the order of joining, and among CAVs joining at the same
   step, the one closer to the area first. A CAV leaves the queue once it has
   cleared the area.
2. **One vehicle at a time.** Only the first CAV in the queue may enter, and
   only when no other vehicle (CAV or human-driven) is inside the area.
3. **Waiting.** The other queued CAVs approach at the middle target speed and
   brake to stop before the area.
4. **Human drivers** cannot be queued (they do not communicate and do not
   follow the order), so the first CAV also waits for a human-driven vehicle
   that would reach the area before it could clear it (estimated arrival time
   from distance / speed). Only the first human-driven vehicle on each lane is
   considered: the ones behind it cannot arrive earlier.
5. A CAV that was given permission always continues, so that it clears the
   area (also one that is clearly inside, e.g. pushed in by its leader).

Noise: positions are noisy, so a vehicle counts as inside the area from
``noise_k * sigma_pos`` before it, and as cleared only ``noise_k * sigma_pos``
after it (the conflict detector's rule, 2 sigma). Waiting CAVs stop that much
earlier, and a CAV that has started to stop keeps stopping until it gets
permission, so that noise cannot make it creep forward.

All quantities come from the same noisy (and possibly delayed) shared states
as every other arbiter, so perception noise can change the order. The
low-level FollowingSafety layer is applied afterwards, as for every arbiter.
CAVs outside the control range drive at the top target speed.

This is the standard FCFS policy for unsignalized intersections (a single
reservation of the whole intersection, served in arrival order). The
CoDrivingLLM paper (Fang et al., 2025) uses FCFS as a rule-based baseline
without specifying it; its reported FCFS (lowest speed, 2.7 m/s, and largest
mean PET, 18.1 s, of all methods) is consistent with this behaviour.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .base import FASTER, Arbiter, toward_speed


def conflict_area_centre(obs) -> np.ndarray:
    """Centre of the conflict area of the scenario (same rule as the MCTS baseline)."""
    nodes = {n for st in obs.states.values() for lane in st.path.key for n in lane}
    if any(str(n).startswith(("ir", "il")) for n in nodes):
        return np.zeros(2)  # HighwayEnv intersection: the junction is centred at the origin
    from ..envs.merge import MERGE_X  # CAVBench merge: the ramp joins at (MERGE_X, 0)
    return np.array([MERGE_X, 0.0])


class FCFSArbiter(Arbiter):
    name = "fcfs"

    def __init__(self, control_range: float = 50.0, half_width: float = 11.0,
                 clear_margin: float = 3.0, stop_offset: float = 3.0, comfort_decel: float = 3.0,
                 reaction: float = 0.4, hdv_buffer: float = 1.0, min_speed: float = 1.0,
                 noise_k: float = 2.0, **kw) -> None:
        """
        control_range: [m] distance to the area at which a CAV joins the queue.
        half_width:    [m] the area is the square of this half-width around the centre
                       (11 m = HighwayEnv's junction box: 9 m turn radius + half a lane).
        clear_margin:  [m] a vehicle has cleared the area when its centre is this far past
                       it (half a car length + 0.5 m).
        stop_offset:   [m] waiting CAVs stop with their centre this far before the area.
        comfort_decel, reaction: braking model used to decide when to start braking.
        hdv_buffer:    [s] safety margin between the first CAV clearing the area and a
                       human-driven vehicle reaching it.
        min_speed:     [m/s] floor of the speed in arrival-time estimates (as the
                       conflict detector).
        noise_k:       position margin = noise_k * sigma_pos (as the conflict detector).
        """
        super().__init__(**kw)
        self.control_range = control_range
        self.half_width = half_width
        self.clear_margin = clear_margin
        self.stop_offset = stop_offset
        self.comfort_decel = comfort_decel
        self.reaction = reaction
        self.hdv_buffer = hdv_buffer
        self.min_speed = min_speed
        self.noise_k = noise_k
        self.queue: list[int] = []
        self.granted: set[int] = set()  # CAVs given permission to enter
        self.holding: set[int] = set()  # CAVs stopping before the area
        self._spans: dict = {}
        self._margin = 0.0

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self.queue = []
        self.granted, self.holding = set(), set()
        self._spans = {}

    # --- geometry ---------------------------------------------------------
    def _span(self, path, centre) -> tuple[float, float] | None:
        """Arc lengths where the path enters and leaves the area (None: never inside)."""
        key = (path.key, round(path.length, 3))
        if key not in self._spans:
            s = np.arange(0.0, path.length + 0.25, 0.25)
            inside = np.max(np.abs(path.position(s) - centre[None, :]), axis=1) <= self.half_width
            idx = np.flatnonzero(inside)
            self._spans[key] = (float(s[idx[0]]), float(s[idx[-1]])) if len(idx) else None
        return self._spans[key]

    def _distances(self, obs, centre) -> dict[int, tuple[float, float]]:
        """vid -> (distance to the area entry, distance until cleared) for vehicles
        that have not (clearly, beyond the noise margin) cleared the area yet."""
        out = {}
        for vid, st in obs.states.items():
            if st.arrived:
                continue
            span = self._span(st.path, centre)
            if span is None:
                continue
            d_in, d_out = span[0] - st.s, span[1] + self.clear_margin - st.s
            if d_out > -self._margin:
                out[vid] = (d_in, d_out)
        return out

    def _time_to_cover(self, distance: float, v0: float, v_max: float) -> float:
        """Time to drive ``distance`` from speed v0, accelerating at the comfort rate to v_max."""
        a = self.comfort_decel
        v0 = max(v0, 0.0)
        if v0 >= v_max:
            return distance / v_max
        d_acc = (v_max ** 2 - v0 ** 2) / (2 * a)
        if distance <= d_acc:
            return (math.sqrt(v0 ** 2 + 2 * a * distance) - v0) / a
        return (v_max - v0) / a + (distance - d_acc) / v_max

    def _first_on_lane(self, obs, vid, dist) -> bool:
        """True if no uncleared vehicle is ahead of ``vid`` on its current lane."""
        me = obs.states[vid]
        return not any(u != vid and obs.states[u].path.key[:1] == me.path.key[:1]
                       and obs.states[u].s > me.s for u in dist)

    # --- decision ---------------------------------------------------------
    def act(self, obs, all_cavs):
        t0 = time.perf_counter()
        centre = conflict_area_centre(obs)
        m = self._margin = self.noise_k * obs.sigma_pos
        dist = self._distances(obs, centre)
        cavs = [v for v in all_cavs if v in obs.states]

        # 1. Queue: drop CAVs that have cleared, append the newcomers (fixed order).
        self.queue = [v for v in self.queue if v in cavs and v in dist]
        self.granted &= set(self.queue)
        self.holding &= set(self.queue)
        new = [v for v in cavs if v not in self.queue and v in dist
               and dist[v][0] <= self.control_range]
        new.sort(key=lambda v: (dist[v][0], v))
        self.queue += new

        actions = {v: FASTER for v in cavs}  # outside the control range: drive on
        goer = None
        for i, v in enumerate(self.queue):
            st = obs.states[v]
            d_in, d_out = dist[v]
            if v in self.granted or d_in < -m:  # permitted, or clearly inside: clear the area
                actions[v] = FASTER
                continue
            if i == 0 and self._may_enter(obs, v, dist):
                actions[v] = FASTER
                self.granted.add(v)
                self.holding.discard(v)
                goer = v
                continue
            # Wait: approach at the middle speed, brake in time to stop before the area.
            d_stop = d_in - self.stop_offset - m
            v_now = st.speed
            braking = v_now * self.reaction + v_now ** 2 / (2 * self.comfort_decel) + 1.0
            if d_stop <= braking:
                self.holding.add(v)
            desired = 0.0 if v in self.holding else obs.v_mid
            actions[v] = toward_speed(st.target_speed, desired)

        if self.queue:
            self.decision_times.append((time.perf_counter() - t0) * 1000.0)
            self.log.append({"time": obs.time, "mode": "fcfs", "queue": list(self.queue),
                             "goer": goer})
        if self.safety is not None:
            actions = self.safety.apply(obs, actions)
        return actions

    def _may_enter(self, obs, v, dist) -> bool:
        """The first CAV may enter if the area is empty and no human driver comes first."""
        for u, (d_in, _) in dist.items():
            if u != v and (d_in <= self._margin or u in self.granted):
                return False  # someone is (or may be) inside the area
        st = obs.states[v]
        t_clear = self._time_to_cover(dist[v][1], st.speed, obs.v_max)
        for u, (d_in, _) in dist.items():
            other = obs.states[u]
            if other.is_cav or d_in > self.control_range:
                continue
            if other.path.key[:1] == st.path.key[:1]:
                continue  # same lane: car following, handled by the safety layer
            if not self._first_on_lane(obs, u, dist):
                continue  # queued behind another vehicle: cannot arrive first
            if d_in / max(other.speed, self.min_speed) < t_clear + self.hdv_buffer:
                return False
        return True

    def arbitrate(self, obs, groups):  # not used: act() handles the whole queue
        return {}
