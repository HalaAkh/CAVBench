"""Conflict detection (Step 2) and grouping of conflicting CAVs.

Two vehicles are in conflict when their remaining routes meet ahead of both
of them, both are within the reaction range of the meeting point, and their
times to reach it differ by less than ``dttcp_threshold``:

    TTCP_i = d_i / v_i,     dTTCP_ij = |TTCP_i - TTCP_j| < 8 s

This follows CoDrivingLLM (Fang et al., 2025): its severity scale is
dTTCP <= 2 s serious, 2-5 s general, 5-8 s slight, >= 8 s no conflict; it
treats routes as meeting when their points come within 1 m and uses a 50 m
reaction range at the intersection. Distances here are measured along the
route (arc length) rather than in a straight line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import meeting_points
from .state import SceneObservation, SharedState


@dataclass
class ConflictConfig:
    dttcp_threshold: float = 8.0  # [s]
    meet_distance: float = 1.0  # [m] routes "meet" when closer than this
    reaction_range: float = 50.0  # [m] max distance to the conflict point
    passed_margin: float = 4.5  # [m] half a car length + half a car width + buffer:
                                # a vehicle blocks the point until it is this far past it
    min_speed: float = 1.0  # [m/s] floor used in TTCP = d / v
    include_hdv: bool = True  # CAV-HDV conflicts form groups of their own


@dataclass
class Conflict:
    i: int  # always a CAV
    j: int  # CAV or HDV
    d_i: float  # [m] distance along route to the conflict point
    d_j: float
    ttcp_i: float
    ttcp_j: float

    @property
    def dttcp(self) -> float:
        return abs(self.ttcp_i - self.ttcp_j)

    @property
    def severity(self) -> str:
        d = self.dttcp
        return "serious" if d <= 2 else "general" if d <= 5 else "slight" if d < 8 else "none"


@dataclass
class ConflictGroup:
    """Connected set of CAVs that must be arbitrated together (the set C)."""

    cavs: list[int]
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def others(self) -> list[int]:
        """Non-member vehicles (HDVs) in conflict with a member."""
        mem = set(self.cavs)
        return sorted({c.j for c in self.conflicts if c.j not in mem})


class ConflictDetector:
    def __init__(self, config: ConflictConfig | None = None) -> None:
        self.cfg = config or ConflictConfig()
        self._cache: dict[tuple, list] = {}

    def _meetings(self, a: SharedState, b: SharedState):
        key = (a.path.key, b.path.key)
        if key not in self._cache:
            self._cache[key] = meeting_points(a.path, b.path, self.cfg.meet_distance)
        return self._cache[key]

    def pair(self, a: SharedState, b: SharedState) -> Conflict | None:
        cfg = self.cfg
        for mp in self._meetings(a, b):
            d_a, d_b = mp.s_i - a.s, mp.s_j - b.s
            if d_a < -cfg.passed_margin or d_b < -cfg.passed_margin:
                continue  # this meeting point is behind one of them
            if mp.s_i < 1e-6 and mp.s_j < 1e-6:
                continue  # routes start together: same approach, car following
            if mp.shared and (d_a < 0 or d_b < 0):
                continue  # one has entered the shared lane: the order is settled
                          # (car following); the 4.5 m margin is for crossings only
            if d_a > cfg.reaction_range or d_b > cfg.reaction_range:
                return None
            t_a = max(d_a, 0.0) / max(a.speed, cfg.min_speed)
            t_b = max(d_b, 0.0) / max(b.speed, cfg.min_speed)
            if abs(t_a - t_b) < cfg.dttcp_threshold:
                return Conflict(a.vid, b.vid, d_a, d_b, t_a, t_b)
            return None
        return None

    def detect(self, obs: SceneObservation) -> list[Conflict]:
        cavs = [obs.states[v] for v in obs.cav_ids]
        others = [st for st in obs.states.values() if not (st.is_cav and not st.arrived)]
        out = []
        for idx, a in enumerate(cavs):
            for b in cavs[idx + 1 :]:
                if a.path.key[0] == b.path.key[0]:
                    continue  # same current lane: car following (benchmark rule)
                c = self.pair(a, b)
                if c:
                    out.append(c)
            if self.cfg.include_hdv:
                for b in others:
                    if b.is_cav or a.path.key[0] == b.path.key[0]:
                        continue
                    c = self.pair(a, b)
                    if c:
                        out.append(c)
        return out

    def groups(self, obs: SceneObservation) -> list[ConflictGroup]:
        conflicts = self.detect(obs)
        cav_set = set(obs.cav_ids)
        parent = {v: v for v in cav_set}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for c in conflicts:
            if c.j in cav_set:
                parent[find(c.i)] = find(c.j)
        members: dict[int, list[int]] = {}
        for c in conflicts:
            for v in (c.i, c.j):
                if v in cav_set:
                    members.setdefault(find(v), [])
                    if v not in members[find(v)]:
                        members[find(v)].append(v)
        groups = []
        for root, cavs in members.items():
            group_conf = [c for c in conflicts if find(c.i) == root]
            groups.append(ConflictGroup(sorted(cavs), group_conf))
        return groups
