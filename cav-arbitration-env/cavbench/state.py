"""The shared (V2X) view of the scene that every arbiter receives.

Arbiters never touch the simulator's true vehicle states. They receive a
:class:`SceneObservation`: positions and speeds corrupted by the noise model,
plus information assumed to be shared exactly (vehicle identity, whether it is
a CAV, its planned route, and, for CAVs, its current target speed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Path


@dataclass
class SharedState:
    vid: int
    is_cav: bool
    position: np.ndarray  # noisy (2,)
    speed: float  # noisy
    path: Path  # planned route (shared exactly)
    s: float  # noisy arc length on the path (map-matched noisy position)
    target_speed: float | None = None  # CAV intent (shared exactly)
    arrived: bool = False


@dataclass
class SceneObservation:
    time: float
    states: dict[int, SharedState]
    sigma_pos: float  # noise level the arbiter assumes for its imagined errors
    sigma_vel: float
    cav_target_speeds: tuple[float, ...] = (0.0, 4.5, 9.0)
    policy_dt: float = 0.2
    extra: dict = field(default_factory=dict)

    @property
    def cav_ids(self) -> list[int]:
        return [v for v, st in self.states.items() if st.is_cav and not st.arrived]

    @property
    def v_max(self) -> float:
        return float(max(self.cav_target_speeds))

    @property
    def v_mid(self) -> float:
        speeds = sorted(self.cav_target_speeds)
        return float(speeds[len(speeds) // 2])
