"""Gaussian perception noise on shared vehicle states (report, Eq. 1).

    s_hat_i = s_i + eps_i,   eps_i ~ N(0, sigma^2)

Position noise is added independently to x and y; speed noise to the speed.
A fresh noise sample is drawn for every vehicle at every decision step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The four noise levels of the experiments: (sigma_position [m], sigma_speed [m/s]).
NOISE_LEVELS: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.0),
    "low": (0.2, 0.1),
    "medium": (0.5, 0.3),
    "high": (1.5, 1.0),
}


@dataclass(frozen=True)
class GaussianNoise:
    sigma_pos: float = 0.0
    sigma_vel: float = 0.0

    @classmethod
    def from_level(cls, level: str) -> "GaussianNoise":
        if level not in NOISE_LEVELS:
            raise ValueError(f"Unknown noise level {level!r}; choose from {list(NOISE_LEVELS)}")
        return cls(*NOISE_LEVELS[level])

    @property
    def is_zero(self) -> bool:
        return self.sigma_pos == 0.0 and self.sigma_vel == 0.0

    def perturb(
        self, positions: np.ndarray, speeds: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return noisy copies of ``positions`` (M, 2) and ``speeds`` (M,)."""
        positions = np.asarray(positions, dtype=float)
        speeds = np.asarray(speeds, dtype=float)
        noisy_pos = positions + rng.normal(0.0, self.sigma_pos, size=positions.shape)
        noisy_spd = np.maximum(speeds + rng.normal(0.0, self.sigma_vel, size=speeds.shape), 0.0)
        return noisy_pos, noisy_spd
