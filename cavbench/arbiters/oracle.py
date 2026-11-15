"""Oracle: an upper bound with perfect information (not a realistic method).

The oracle "cheats on purpose", to show the best result achievable in each
situation:

* it sees the TRUE vehicle states (no perception noise, no V2X delay or loss),
  and it detects conflicts on them (same ConflictDetector, sigma = 0);
* it evaluates each option on an exact COPY of the real simulation
  (``copy.deepcopy`` of the environment, including its random generator), so
  it knows the real future of that option: true physics, true human drivers.

Mode A, passing-order enumeration (groups of K <= ``max_enum_cavs`` CAVs), as
the global-optimum reference of H. Xu, Y. Zhang, L. Li, W. Li, "Cooperative
driving at unsignalized intersections using tree search", IEEE T-ITS 21(11),
4563-4571, 2020 (doi:10.1109/TITS.2019.2940641). Every passing order (K!) is
run on a copy with the order-following rule (:meth:`OracleArbiter.order_actions`)
until all CAVs of the group have cleared their conflict points; the order is
chosen lexicographically (no collision, fewest severe conflicts, smallest
total clearing time, then the arrival (TTCP) order). The chosen order is
followed at every step and re-planned every ``replan_period`` (2 s, as Xu et
al.) and whenever the group changes.

Mode B, option search (K > ``max_enum_cavs``): the candidate set of the
what-if method (one option per CAV going first, with the CAVs queued ahead of
it, plus "everyone yields"), each held for ``option_horizon`` (4 s) on a copy,
chosen with the same rule (progress instead of clearing time), re-planned
every step. Not exhaustive.

The runner gives the oracle the environment through the existing
``attach(env)`` hook (called after ``env.reset``). The perception-noise
generator is never passed to it, so all methods keep matched noise samples.
See docs/oracle_baseline.md for the limitations.
"""
from __future__ import annotations

import copy
import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from ..conflicts import ConflictGroup
from ..metrics import pet_values
from ..state import SceneObservation, SharedState
from .base import FASTER, IDLE, SLOWER, Arbiter
from .commitment import arrival_time, platoon_ahead

CHOICE_KEYS = ("collision", "severe", "cost")


@dataclass
class OracleConfig:
    max_enum_cavs: int = 5  # mode A up to this group size (K! orders), mode B above
    max_eval_time: float = 20.0  # [s] cap of one order evaluation (mode A)
    replan_period: float = 2.0  # [s] mode A re-plans this often (Xu et al.)
    option_horizon: float = 4.0  # [s] mode B: each option is held this long
    post_clear_time: float = 1.5  # [s] keep simulating after everyone cleared, so
                                  # that a vehicle entering a conflict area just
                                  # behind still counts as a severe conflict
    pet_threshold: float = 1.5  # [s] severe conflict, as in the metrics
    stop_speed: float = 1.0  # [m/s] a yielding CAV below this, with target 0, sends IDLE
    choice: tuple = CHOICE_KEYS  # lexicographic keys, lower first; ties -> TTCP order
    prune: bool = True  # stop evaluating an order that can no longer win (same choice)


# -- perfect information ----------------------------------------------------
def true_observation(env) -> SceneObservation:
    """The scene as it really is: true positions and speeds, no V2X effects,
    sigma = 0. Uses no random numbers."""
    states = {}
    for v in env.road.vehicles:
        vid = env.vid(v)
        path = env.path_cache.path_for(v)
        pos = np.array(v.position, dtype=float)
        s, _ = path.project(pos)
        is_cav = v in env.controlled_vehicles
        states[vid] = SharedState(
            vid=vid, is_cav=is_cav, position=pos, speed=float(v.speed), path=path, s=s,
            target_speed=float(v.target_speed) if is_cav else None,
            arrived=bool(is_cav and env.has_arrived(v)),
        )
    return SceneObservation(
        time=float(env.time), states=states, sigma_pos=0.0, sigma_vel=0.0,
        cav_target_speeds=env.cav_target_speeds,
        policy_dt=1.0 / env.config["policy_frequency"], extra={},
    )


def copy_env(env):
    """Exact, independent copy of the running simulation (vehicles, human
    drivers' parameters, random generator, time, V2X history). The viewer is
    not copied; route polylines (read-only) are shared; the copied vehicles
    keep their ids. Nothing in ``env`` is changed."""
    memo: dict = {}
    for name in ("viewer", "_record_video_wrapper"):
        obj = getattr(env, name, None)
        if obj is not None:
            memo[id(obj)] = None
    trajectory = env.trajectory
    memo[id(trajectory)] = []  # copied below as a flat list (its rows are tuples)
    for p in list(env.path_cache._paths.values()) + list(env.vehicle_paths.values()):
        memo[id(p)] = p
    sim = copy.deepcopy(env, memo)
    sim.trajectory = list(trajectory)
    sim.render_mode = None
    vid = dict(env._vid)  # ids are keyed by object identity: re-key them
    for v in list(env.road.vehicles) + list(env.controlled_vehicles):
        if id(v) in vid and id(v) in memo:
            vid[id(memo[id(v)])] = vid.pop(id(v))
    sim._vid = vid
    return sim


def choose(results: list[dict], keys: tuple = CHOICE_KEYS) -> int:
    """Index of the best result, lexicographically (lower first) on ``keys``,
    then on ``tiebreak``. collision: False < True; cost is rounded to 1e-6."""
    def key(r):
        k = []
        for name in keys:
            x = r[name]
            k.append(round(float(x), 6) if name == "cost" else int(x))
        return tuple(k) + (r["tiebreak"],)
    return min(range(len(results)), key=lambda i: key(results[i]))


@dataclass
class _Plan:
    """A passing order to follow. ``hdv_yield``: the CAVs also give way to human
    drivers that would reach a shared conflict point earlier (yield at the conflict point)."""
    order: tuple
    hdv_yield: bool
    time: float


class OracleArbiter(Arbiter):
    name = "oracle"

    def __init__(self, oracle: dict | OracleConfig | None = None, **kw) -> None:
        super().__init__(**kw)
        self.cfg = oracle if isinstance(oracle, OracleConfig) else OracleConfig(**(oracle or {}))
        self.env = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.plans: dict[frozenset, _Plan] = {}  # mode A plans being followed
        self.context: dict[frozenset, _Plan] = {}  # every group's current plan, used in copies
        self.n_orders = 0  # candidate evaluations in mode A
        self.n_options = 0  # candidate evaluations in mode B
        self.n_plans_a = 0
        self.n_plans_b = 0
        self.n_cap = 0  # evaluations that reached max_eval_time
        self.n_other_crash = 0  # evaluations in which a CAV outside the group crashed
        self.max_group = 0

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self._reset_state()

    def attach(self, env) -> None:
        """Called by the runner after env.reset(): the oracle reads (and copies)
        the real simulation. It never steps it."""
        self.env = env

    def extra_metrics(self) -> dict:
        plans = self.n_plans_a + self.n_plans_b
        return {
            "oracle_orders_evaluated": self.n_orders,
            "oracle_options_evaluated": self.n_options,
            "oracle_enum_share": self.n_plans_a / plans if plans else float("nan"),
            "oracle_replans": plans,
            "oracle_cap_reached": self.n_cap,
            "oracle_max_group_size": self.max_group,
            "oracle_other_crash_evals": self.n_other_crash,  # diagnostics, not a CSV column
        }

    # -- one decision step ------------------------------------------------
    def act(self, obs: SceneObservation, all_cavs: list[int]) -> dict[int, int]:
        """Same structure as Arbiter.act, but on the TRUE states: the noisy
        ``obs`` is ignored (conflicts, decisions and the safety layer all use
        the true scene)."""
        if self.env is None:
            raise RuntimeError("oracle needs attach(env) after env.reset(); "
                               "cavbench.runner.run_episode does this.")
        t0 = time.perf_counter()
        true = true_observation(self.env)
        groups = self.detector.groups(true)
        actions = {vid: FASTER for vid in all_cavs}
        if groups:
            actions.update(self.arbitrate(true, groups))
            self.decision_times.append((time.perf_counter() - t0) * 1000.0)
        if self.safety is not None:
            actions = self.safety.apply(true, actions)
        return actions

    @staticmethod
    def _arrival_order(g) -> tuple:
        ttcp = {v: arrival_time(g, v) for v in g.cavs}
        return tuple(sorted(g.cavs, key=lambda v: (ttcp[v], v)))

    def arbitrate(self, obs, groups):
        actions = {}
        keys = {frozenset(g.cavs) for g in groups}
        self.plans = {k: p for k, p in self.plans.items() if k in keys}
        # Current plan of every group, used for the other groups' CAVs in the
        # copies: the stored plan or, if not planned yet, the arrival order with
        # yielding to earlier human drivers (yield at the conflict point).
        self.context = {frozenset(g.cavs): self.plans.get(frozenset(g.cavs))
                        or _Plan(self._arrival_order(g), True, obs.time) for g in groups}
        for g in groups:
            self.max_group = max(self.max_group, len(g.cavs))
            if len(g.cavs) <= self.cfg.max_enum_cavs:
                acts, entry = self._mode_a(obs, g)
            else:
                acts, entry = self._mode_b(obs, g)
            actions.update(acts)
            entry.update({"time": obs.time, "group": list(g.cavs),
                          "actions": {int(v): int(a) for v, a in acts.items()}})
            self.log.append(entry)
        return actions

    # -- mode A: passing orders ---------------------------------------------
    @staticmethod
    def enumerate_orders(cavs) -> list[tuple]:
        return list(itertools.permutations(cavs))

    def candidates(self, g) -> list[tuple[tuple, bool]]:
        """Mode A candidates: every passing order of the group's CAVs (K!), each
        with the CAVs giving way to earlier human drivers (True) or going before
        them (False): 2 K! candidates, the arrival order with yielding first."""
        rank = {v: i for i, v in enumerate(self._arrival_order(g))}
        orders = sorted(self.enumerate_orders(g.cavs), key=lambda o: tuple(rank[v] for v in o))
        return [(o, y) for o in orders for y in (True, False)]

    def _mode_a(self, obs, g):
        key = frozenset(g.cavs)
        plan = self.plans.get(key)
        entry = {"mode": "enum_follow"}
        if plan is None or obs.time - plan.time >= self.cfg.replan_period - 1e-9:
            order, hdv_yield, evals = self.plan_order(obs, g)
            plan = self.plans[key] = _Plan(order, hdv_yield, obs.time)
            entry = {"mode": "enum_plan", "evaluations": evals}
        self.context[key] = plan
        entry["order"] = [int(v) for v in plan.order]
        entry["hdv_yield"] = plan.hdv_yield
        return self.order_actions(obs, plan.order, plan.hdv_yield), entry

    def plan_order(self, obs, g) -> tuple[tuple, bool, list[dict]]:
        """Evaluate every candidate on a copy of the simulation; return the chosen
        order, its human-driver rule and a short summary of each evaluation."""
        self.n_plans_a += 1
        rank = {v: i for i, v in enumerate(self._arrival_order(g))}
        cands = self.candidates(g)
        results, best = [], None
        for order, hdv_yield in cands:  # "yield" first: a good bound early prunes more
            r = self.simulate_order(obs, g, order, best, hdv_yield)
            r["tiebreak"] = (tuple(rank[v] for v in order), 0 if hdv_yield else 1)
            results.append(r)
            i = choose(results, self.cfg.choice)
            if not results[i]["collision"] and results[i]["severe"] == 0:
                best = results[i]["cost"]
        k = choose(results, self.cfg.choice)
        summary = [{"order": [int(v) for v in o], "hdv_yield": y, "collision": r["collision"],
                    "severe": r["severe"], "cost": round(r["cost"], 3), "pruned": r["pruned"],
                    "cap": r["cap"]}
                   for (o, y), r in zip(cands, results)]
        return cands[k][0], cands[k][1], summary

    def copy_actions(self, obs, g: ConflictGroup, group_actions: dict) -> dict[int, int]:
        """Actions of all CAVs in a copy: the group being evaluated gets
        ``group_actions``; the CAVs of the other groups follow their own current
        plan (same order-following rule); the remaining CAVs drive FASTER (the
        base default); then the safety layer, as in Arbiter.act."""
        acts = {v: FASTER for v, st in obs.states.items() if st.is_cav and not st.arrived}
        mine = frozenset(g.cavs)
        for key, plan in self.context.items():
            if key != mine:
                acts.update({v: a for v, a in self.order_actions(obs, plan.order, plan.hdv_yield).items()
                             if v in acts})
        acts.update({v: a for v, a in group_actions.items() if v in acts})
        if self.safety is not None:
            acts = self.safety.apply(obs, acts)
        return acts

    def simulate_order(self, obs, g: ConflictGroup, order, best_cost: float | None = None,
                       hdv_yield: bool = True) -> dict:
        """Run ``order`` on a copy until every CAV of the group has cleared its
        conflict points (plus ``post_clear_time``), a CAV of the group crashes,
        or ``max_eval_time``. cost = sum of the clearing times (total delay)."""
        cfg = self.cfg
        self.n_orders += 1
        sim = copy_env(self.env)
        dt = 1.0 / sim.config["policy_frequency"]
        members = list(g.cavs)
        targets = set(g.cavs) | set(g.others)
        clear_t: dict[int, float] = {}
        t, cur = 0.0, obs
        collision = cap = pruned = False
        t_all = None
        while True:
            for v in members:
                if v not in clear_t and self.has_cleared(cur, v, targets):
                    clear_t[v] = t
            if len(clear_t) == len(members):
                t_all = t if t_all is None else t_all
                if t - t_all >= cfg.post_clear_time - 1e-9:
                    break
            if t >= cfg.max_eval_time - 1e-9:
                cap = True
                break
            if cfg.prune and best_cost is not None and t_all is None:
                bound = sum(clear_t.values()) + t * (len(members) - len(clear_t))
                if bound > best_cost + 1e-9:
                    pruned = True
                    break
            acts = self.copy_actions(cur, g, self.order_actions(cur, order, hdv_yield))
            _, _, _, truncated, _ = sim.step(tuple(int(acts.get(v, FASTER)) for v in sim.cav_vids()))
            t += dt
            cur = true_observation(sim)
            if self._group_crashed(sim, members):
                collision = True
                break
            if truncated or _all_arrived(sim):  # end of the episode in the copy
                for v in members:
                    if v not in clear_t and self.has_cleared(cur, v, targets):
                        clear_t[v] = t
                cap = len(clear_t) < len(members)
                break
        self._count_other_crash(sim, members)
        if cap:
            self.n_cap += 1
        if pruned:  # cannot win: its cost is already above the best clean candidate
            return {"collision": False, "severe": 0, "cost": bound, "pruned": True, "cap": False}
        cost = sum(clear_t.get(v, cfg.max_eval_time) for v in members)
        return {"collision": collision, "severe": self._severe(sim), "cost": cost,
                "pruned": False, "cap": cap}

    def _severe(self, sim) -> int:
        """Severe conflicts (PET below the threshold), same code as the metrics."""
        return int(sum(p["pet"] < self.cfg.pet_threshold for p in pet_values(sim)))

    # -- the order-following rule -------------------------------------------
    def _points(self, a: SharedState, b: SharedState):
        """Conflict points of two routes: the ConflictDetector's meeting points,
        skipping the start of a shared approach lane (car following)."""
        for mp in self.detector._meetings(a, b):
            if mp.s_i < 1e-6 and mp.s_j < 1e-6:
                continue
            yield mp.s_i - a.s, mp.s_j - b.s  # distance to the point for a and b

    def blocks(self, obs, v: int, p: int) -> bool:
        """Must CAV v wait for p? Yes while there is a conflict point that v has
        not reached yet and p has not cleared (passed by the detector's
        ``passed_margin``, 4.5 m)."""
        a, b = obs.states.get(v), obs.states.get(p)
        if a is None or b is None or b.arrived:
            return False
        margin = self.detector.cfg.passed_margin
        return any(d_v >= 0.0 and d_p >= -margin for d_v, d_p in self._points(a, b))

    def hdv_blocks(self, obs, v: int) -> bool:
        """Yield-at-the-conflict-point rule for human drivers (true states): CAV v waits for a
        human-driven vehicle that would reach their first shared conflict point
        ahead no later than v (TTCP = d / max(speed, 1 m/s), both within the
        detector's reaction range) and has not cleared it yet."""
        a = obs.states.get(v)
        if a is None:
            return False
        cfg = self.detector.cfg
        for b in obs.states.values():
            if b.is_cav or a.path.key[0] == b.path.key[0]:
                continue  # same current lane: car following (the safety layer)
            for d_v, d_h in self._points(a, b):
                if d_v < 0.0 or d_h < -cfg.passed_margin:
                    continue  # v is already at the point, or the driver has cleared it
                if d_v > cfg.reaction_range or d_h > cfg.reaction_range:
                    break
                t_v = d_v / max(a.speed, cfg.min_speed)
                t_h = max(d_h, 0.0) / max(b.speed, cfg.min_speed)
                if t_h <= t_v:
                    return True
                break  # only the first point ahead counts, as in the detector
        return False

    def has_cleared(self, obs, v: int, targets) -> bool:
        """CAV v has passed (by passed_margin) every conflict point it has with
        the vehicles of its group, or has left the scene."""
        a = obs.states.get(v)
        if a is None or a.arrived:
            return True
        margin = self.detector.cfg.passed_margin
        for u in targets:
            b = obs.states.get(u)
            if u == v or b is None:
                continue
            if any(d_v >= -margin for d_v, _ in self._points(a, b)):
                return False
        return True

    def _wait_action(self, obs, st) -> int:
        nearly_stopped = st.speed < self.cfg.stop_speed and (st.target_speed or 0.0) <= min(
            obs.cav_target_speeds) + 1e-6
        return IDLE if nearly_stopped else SLOWER

    def order_actions(self, obs, order, hdv_yield: bool = False) -> dict[int, int]:
        """The first CAV of the order drives FASTER; every next CAV yields
        (SLOWER, or IDLE once it is nearly stopped) while an earlier CAV of the
        order has not cleared a conflict point they share, then drives FASTER.
        With ``hdv_yield`` every CAV also yields to human drivers as in
        :meth:`hdv_blocks`. CAVs that have left the scene are dropped."""
        acts, earlier = {}, []
        for v in order:
            st = obs.states.get(v)
            if st is None or st.arrived:
                continue
            wait = any(self.blocks(obs, v, p) for p in earlier) or (
                hdv_yield and self.hdv_blocks(obs, v))
            acts[v] = self._wait_action(obs, st) if wait else FASTER
            earlier.append(v)
        return acts

    # -- mode B: option search ----------------------------------------------
    def _mode_b(self, obs, g):
        """One option per CAV going first (with the CAVs queued ahead of it), with
        and without yielding to earlier human drivers, plus 'everyone yields'
        (2K + 1 options), each held for option_horizon on a copy; re-planned
        every step."""
        self.n_plans_b += 1
        cands = [(goer, y) for goer in self._arrival_order(g) for y in (True, False)] + [(None, True)]
        results = []
        for goer, y in cands:
            moving = {goer, *platoon_ahead(obs, g, goer)} if goer is not None else set()
            r = self.simulate_option(obs, g, moving, y)
            r["tiebreak"] = (arrival_time(g, goer) if goer is not None else math.inf, 0 if y else 1)
            results.append(r)
        k = choose(results, self.cfg.choice)
        goer, y = cands[k]
        moving = {goer, *platoon_ahead(obs, g, goer)} if goer is not None else set()
        acts = self.option_actions(obs, g, moving, y)
        # For the copies of the other groups: this choice written as an order
        # (the moving CAVs first, front to back, then the arrival order).
        front = tuple(sorted(moving, key=lambda v: -obs.states[v].s))
        rest = tuple(v for v in self._arrival_order(g) if v not in moving)
        self.context[frozenset(g.cavs)] = _Plan(front + rest, y, obs.time)
        entry = {"mode": "option" if goer is not None else "fallback", "goer": goer, "hdv_yield": y,
                 "evaluations": [{"goer": go, "hdv_yield": yy, "collision": r["collision"],
                                  "severe": r["severe"], "progress": round(-r["cost"], 2)}
                                 for (go, yy), r in zip(cands, results)]}
        return acts, entry

    def option_actions(self, obs, g, moving: set, hdv_yield: bool) -> dict[int, int]:
        """Mode B option: the moving CAVs drive FASTER (with ``hdv_yield``: unless
        they must give way to a human driver), the other CAVs of the group SLOWER."""
        acts = {}
        for v in g.cavs:
            st = obs.states.get(v)
            if st is None or st.arrived:
                continue
            go = v in moving and not (hdv_yield and self.hdv_blocks(obs, v))
            acts[v] = FASTER if go else SLOWER
        return acts

    def simulate_option(self, obs, g: ConflictGroup, moving: set, hdv_yield: bool = True) -> dict:
        """Hold one option for option_horizon on a copy. cost = - progress (metres
        travelled by the group's CAVs; a CAV that has left counts at top speed)."""
        self.n_options += 1
        sim = copy_env(self.env)
        dt = 1.0 / sim.config["policy_frequency"]
        steps = int(round(self.cfg.option_horizon / dt))
        cur, progress, collision = obs, 0.0, False
        for _ in range(steps):
            acts = self.copy_actions(cur, g, self.option_actions(cur, g, moving, hdv_yield))
            _, _, _, truncated, _ = sim.step(tuple(int(acts.get(v, FASTER)) for v in sim.cav_vids()))
            cur = true_observation(sim)
            for v in g.cavs:
                st = cur.states.get(v)
                progress += (cur.v_max if st is None or st.arrived else st.speed) * dt
            if self._group_crashed(sim, g.cavs):
                collision = True
                break
            if truncated or _all_arrived(sim):
                break
        self._count_other_crash(sim, g.cavs)
        return {"collision": collision, "severe": self._severe(sim), "cost": -progress}

    @staticmethod
    def _group_crashed(sim, members) -> bool:
        """A collision counts for an option only if a CAV of the group crashed.
        A crash among other vehicles does not stop the copy (the crashed cars
        stay on the road as obstacles), so it cannot hide the group's result."""
        mem = set(members)
        return any(v.crashed and sim.vid(v) in mem for v in sim.controlled_vehicles)

    def _count_other_crash(self, sim, members) -> None:
        mem = set(members)
        if any(v.crashed and sim.vid(v) not in mem for v in sim.controlled_vehicles):
            self.n_other_crash += 1


def _all_arrived(sim) -> bool:
    return all(sim.has_arrived(v) for v in sim.controlled_vehicles)
