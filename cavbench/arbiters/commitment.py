"""Decision rules used by the what-if arbiter.

* Priority persistence: once a CAV has been given priority in a conflict, it
  keeps it while its option stays within ``margin`` of the best option.
  Without this, near-equal options make the chosen vehicle change every
  0.2 s, so all vehicles alternately speed up and brake.
* Committed vehicles: in fallback, a CAV that can no longer stop before its
  conflict point (braking distance v^2 / (2 |a_yield|)) is allowed to clear
  the conflict area; stopping it inside would block the other routes.
"""
from __future__ import annotations

from ..conflicts import ConflictGroup
from ..state import SceneObservation


def pick_with_persistence(values: list[float], goers: list[int], previous: int | None,
                          margin: float, higher_is_better: bool, tiebreak: list[tuple]) -> int:
    """Index of the chosen option. ``tiebreak`` gives extra keys (lower first)."""
    sign = -1.0 if higher_is_better else 1.0
    best = min(range(len(values)), key=lambda i: (sign * values[i],) + tiebreak[i])
    if previous is not None and previous in goers:
        p = goers.index(previous)
        if sign * (values[p] - values[best]) <= margin:
            return p
    return best


def committed_cavs(obs: SceneObservation, group: ConflictGroup, a_yield: float,
                   margin: float = 1.0) -> list[int]:
    """Group CAVs that cannot stop before their nearest conflict point."""
    out = []
    for v in group.cavs:
        d = [c.d_i for c in group.conflicts if c.i == v] + \
            [c.d_j for c in group.conflicts if c.j == v]
        if not d:
            continue
        speed = obs.states[v].speed
        stopping = speed * speed / (2.0 * abs(a_yield))
        if min(d) < stopping + margin:
            out.append(v)
    return out
