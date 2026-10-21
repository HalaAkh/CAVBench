"""First-come, first-served: the vehicle expected to reach the conflict point
first goes; every CAV that would arrive later than a conflicting vehicle
yields. Arrival times come from the same noisy shared states as every other
arbiter (TTCP = d / v), so noise can flip the order.
"""
from __future__ import annotations

from .base import FASTER, SLOWER, Arbiter


class FCFSArbiter(Arbiter):
    name = "fcfs"

    def arbitrate(self, obs, groups):
        actions = {}
        for g in groups:
            for v in g.cavs:
                yields = False
                for c in g.conflicts:
                    if c.i == v:
                        mine, theirs, other = c.ttcp_i, c.ttcp_j, c.j
                    elif c.j == v:
                        mine, theirs, other = c.ttcp_j, c.ttcp_i, c.i
                    else:
                        continue
                    if theirs < mine or (theirs == mine and other < v):
                        yields = True
                        break
                actions[v] = SLOWER if yields else FASTER
        return actions
