"""Route geometry: vehicle routes as arc-length-parameterised polylines.

Every vehicle in HighwayEnv follows a planned route (a list of lanes). This
module turns that route into a :class:`Path`, a polyline sampled every
``step`` metres, so that conflict detection and the what-if rollouts can work
in one dimension (distance travelled along the path) instead of in 2-D.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_STEP = 0.5  # [m] spacing of route points (report, Step 2)


def lane_polyline(lane, step: float = DEFAULT_STEP) -> np.ndarray:
    """Sample a HighwayEnv lane along its centre line every ``step`` metres."""
    n = max(int(np.ceil(lane.length / step)), 1)
    s = np.linspace(0.0, lane.length, n + 1)
    return np.array([lane.position(si, 0.0) for si in s])


@dataclass
class Path:
    """A route as a polyline with cumulative arc length."""

    points: np.ndarray  # (P, 2)
    key: tuple = ()
    s: np.ndarray = field(init=False)  # (P,) arc length of each point
    tangents: np.ndarray = field(init=False)  # (P, 2) unit tangent at each point

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float)
        # Drop duplicate consecutive points (lane ends coincide with next lane starts).
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6
        pts = pts[keep]
        self.points = pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        d = np.gradient(pts, axis=0) if len(pts) > 1 else np.array([[1.0, 0.0]])
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        self.tangents = d / np.maximum(norm, 1e-9)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def project(self, xy: np.ndarray) -> tuple[float, float]:
        """Map-match a 2-D point: return (arc length, lateral distance)."""
        d = np.linalg.norm(self.points - np.asarray(xy)[None, :], axis=1)
        i = int(np.argmin(d))
        # Refine on the neighbouring segment.
        best_s, best_d = self.s[i], d[i]
        for j in (i - 1, i):
            if 0 <= j < len(self.points) - 1:
                a, b = self.points[j], self.points[j + 1]
                ab = b - a
                t = np.clip(np.dot(xy - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
                p = a + t * ab
                dist = np.linalg.norm(xy - p)
                if dist < best_d:
                    best_d, best_s = dist, self.s[j] + t * np.linalg.norm(ab)
        return float(best_s), float(best_d)

    def position(self, s: np.ndarray) -> np.ndarray:
        """Positions at arc lengths ``s`` (any shape) -> shape s.shape + (2,)."""
        s = np.asarray(s, dtype=float)
        x = np.interp(s, self.s, self.points[:, 0])
        y = np.interp(s, self.s, self.points[:, 1])
        return np.stack([x, y], axis=-1)

    def tangent(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        tx = np.interp(s, self.s, self.tangents[:, 0])
        ty = np.interp(s, self.s, self.tangents[:, 1])
        n = np.maximum(np.hypot(tx, ty), 1e-9)
        return np.stack([tx / n, ty / n], axis=-1)


def remaining_route(vehicle) -> list:
    """Lanes the vehicle will still drive on, starting with its current lane."""
    current = vehicle.lane_index
    route = list(getattr(vehicle, "route", None) or [])
    for i, lane in enumerate(route):
        if lane[:2] == current[:2]:
            return [current] + route[i + 1 :]
    return [current] + route


class PathCache:
    """Build each distinct route polyline once per episode."""

    def __init__(self, network, step: float = DEFAULT_STEP) -> None:
        self.network = network
        self.step = step
        self._paths: dict[tuple, Path] = {}

    def path_for(self, vehicle) -> Path:
        lanes = remaining_route(vehicle)
        key = tuple((a, b) for a, b, *_ in lanes)
        if key not in self._paths:
            pts = [lane_polyline(self.network.get_lane((a, b, 0)), self.step) for a, b in key]
            self._paths[key] = Path(np.concatenate(pts, axis=0), key=key)
        return self._paths[key]


@dataclass
class MeetingPoint:
    """Start of a stretch where two paths come within ``meet_distance``."""

    s_i: float  # arc length on path i
    s_j: float  # arc length on path j
    length: float = 0.0  # length of the shared stretch along path i

    @property
    def shared(self) -> bool:
        """True for merging / same-lane stretches, False for a crossing."""
        return self.length >= SHARED_MIN_LENGTH


SHARED_MIN_LENGTH = 4.0  # [m] a crossing overlaps for ~2 m; a shared lane for much longer


def meeting_points(pi: Path, pj: Path, meet_distance: float) -> list[MeetingPoint]:
    """All places where path j comes within ``meet_distance`` of path i.

    Crossing paths give one short stretch; merging paths give one long stretch
    that starts at the merge point. Each stretch is reported by its start.
    """
    diff = pi.points[:, None, :] - pj.points[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    close = dist.min(axis=1) < meet_distance
    out: list[MeetingPoint] = []
    starts = np.flatnonzero(close & ~np.concatenate([[False], close[:-1]]))
    ends = np.flatnonzero(close & ~np.concatenate([close[1:], [False]]))
    for a, e in zip(starts, ends):
        # Entry point on path j: the first point of j within meet_distance of
        # this stretch of i (as in the benchmark, both vehicles' distances are
        # measured to where they enter the conflict zone).
        near_j = np.flatnonzero(dist[a:e + 1].min(axis=0) < meet_distance)
        b = int(near_j[0]) if len(near_j) else int(np.argmin(dist[a]))
        out.append(MeetingPoint(float(pi.s[a]), float(pj.s[b]), float(pi.s[e] - pi.s[a])))
    return out


_RELATION_CACHE: dict[tuple, tuple] = {}


def route_relation(pa: Path, pb: Path, meet_distance: float = 1.0):
    """How two routes relate, cached by route.

    Returns ``(meet, shared)``: ``meet`` is True if the routes cross or share a
    lane anywhere; ``shared`` is the first shared stretch as a
    :class:`MeetingPoint` (start on each route and length), or None.
    """
    key = (pa.key, pb.key)
    if key not in _RELATION_CACHE:
        ab = meeting_points(pa, pb, meet_distance)
        ba = meeting_points(pb, pa, meet_distance)
        _RELATION_CACHE[key] = (bool(ab), follow_stretch(pa, pb))
        _RELATION_CACHE[(pb.key, pa.key)] = (bool(ba) or bool(ab), follow_stretch(pb, pa))
    return _RELATION_CACHE[key]


FOLLOW_DISTANCE = 2.2  # [m] about one car width
FOLLOW_ALIGNMENT = 0.9  # cos(~25 deg): paths heading the same way


def follow_stretch(pa: Path, pb: Path) -> MeetingPoint | None:
    """First stretch on which a vehicle on route a can run into one on route b
    from behind: the routes are within a car width of each other AND heading the
    same way. This covers sharing a lane, merging, and two routes that leave the
    same lane and diverge gradually (their car bodies still overlap for several
    metres). Crossing routes are excluded by the alignment condition."""
    diff = pa.points[:, None, :] - pb.points[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    jn = dist.argmin(axis=1)
    near = dist[np.arange(len(jn)), jn] < FOLLOW_DISTANCE
    aligned = np.einsum("ij,ij->i", pa.tangents, pb.tangents[jn]) > FOLLOW_ALIGNMENT
    close = near & aligned
    starts = np.flatnonzero(close & ~np.concatenate([[False], close[:-1]]))
    ends = np.flatnonzero(close & ~np.concatenate([close[1:], [False]]))
    for a, e in zip(starts, ends):
        if pa.s[e] - pa.s[a] >= SHARED_MIN_LENGTH:
            return MeetingPoint(float(pa.s[a]), float(pb.s[jn[a]]), float(pa.s[e] - pa.s[a]))
    return None


def is_leader(rel_shared, s_me: float, s_other: float) -> bool:
    """Is the other vehicle my leader on a shared stretch of lane?

    Only once it has entered the shared stretch (and is still on it) and is
    further along it than me. Before that, who goes first is the arbiter's
    decision, as at a crossing.
    """
    if rel_shared is None:
        return False
    p_me = s_me - rel_shared.s_i
    p_other = s_other - rel_shared.s_j
    return 0.0 <= p_other <= rel_shared.length and p_other > p_me
