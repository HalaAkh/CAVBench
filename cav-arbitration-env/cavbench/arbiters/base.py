"""Common interface for all arbiters, plus the shared low-level safety layer.

An arbiter receives the noisy :class:`SceneObservation` every policy step
(0.2 s) and returns one high-level action per CAV: SLOWER, IDLE or FASTER
(HighwayEnv's ``DiscreteMetaAction``, the benchmark's action space).

Every arbiter, including the baselines, passes through the same car-following
safety layer, so that differences in results come from the arbitration
policy and not from different collision-avoidance logic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..conflicts import ConflictConfig, ConflictDetector, ConflictGroup
from ..geometry import is_leader, route_relation
from ..state import SceneObservation

SLOWER, IDLE, FASTER = 0, 1, 2
ACTION_NAMES = {SLOWER: "SLOWER", IDLE: "IDLE", FASTER: "FASTER"}


def toward_speed(current_target: float | None, desired: float) -> int:
    """Action that moves a CAV's target speed towards ``desired``."""
    if current_target is None or abs(current_target - desired) < 1e-6:
        return IDLE
    return FASTER if current_target < desired else SLOWER


@dataclass
class FollowingSafety:
    """Brake for vehicles ahead on the CAV's own route (noisy states)."""

    s0: float = 3.0  # [m] standstill gap
    time_gap: float = 1.2  # [s]
    lateral_tol: float = 2.5  # [m] counts as "on my route" (car width 2 m + 0.5 m;
                              # adjacent lanes are 4 m apart)
    look_ahead: float = 40.0  # [m]
    vehicle_length: float = 5.0

    leader_range: float = 30.0  # same leader rule as the what-if rollouts

    def cap(self, obs: SceneObservation, vid: int) -> int:
        me = obs.states[vid]
        gap = np.inf
        for oid, other in obs.states.items():
            if oid == vid:
                continue
            disp = other.position - me.position
            dist = float(np.hypot(*disp))
            if dist > self.look_ahead + 5:
                continue
            # (a) anything physically on my route ahead
            s_o, lat = me.path.project(other.position)
            ds = s_o - me.s
            if lat < self.lateral_tol and 0 < ds < self.look_ahead:
                gap = min(gap, ds - self.vehicle_length)
            # (b) a vehicle already on a shared stretch of lane, ahead of me
            elif dist < self.leader_range and is_leader(
                    route_relation(me.path, other.path)[1], me.s, other.s):
                gap = min(gap, dist - self.vehicle_length)
        desired = self.s0 + me.speed * self.time_gap
        if gap < desired:
            return SLOWER
        if gap < desired + me.speed * 1.0:
            return IDLE
        return FASTER

    def apply(self, obs: SceneObservation, actions: dict[int, int]) -> dict[int, int]:
        return {vid: min(a, self.cap(obs, vid)) for vid, a in actions.items()}


class Arbiter:
    """Base class. Subclasses implement :meth:`arbitrate`."""

    name = "base"

    def __init__(self, conflict: dict | ConflictConfig | None = None, safety: bool = True,
                 seed: int | None = None) -> None:
        cfg = conflict if isinstance(conflict, ConflictConfig) else ConflictConfig(**(conflict or {}))
        self.detector = ConflictDetector(cfg)
        self.safety = FollowingSafety() if safety else None
        self.rng = np.random.default_rng(seed)
        self.decision_times: list[float] = []  # [ms], steps with at least one conflict
        self.log: list[dict] = []

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.decision_times = []
        self.log = []

    def arbitrate(self, obs: SceneObservation, groups: list[ConflictGroup]) -> dict[int, int]:
        raise NotImplementedError

    def act(self, obs: SceneObservation, all_cavs: list[int]) -> dict[int, int]:
        t0 = time.perf_counter()
        groups = self.detector.groups(obs)
        actions = {vid: FASTER for vid in all_cavs}  # no conflict: proceed
        if groups:
            actions.update(self.arbitrate(obs, groups))
            self.decision_times.append((time.perf_counter() - t0) * 1000.0)
        if self.safety is not None:
            actions = self.safety.apply(obs, actions)
        return actions
