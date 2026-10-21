"""No coordination: every CAV drives on at its top target speed and only the
shared car-following safety layer reacts to vehicles ahead on its route."""
from __future__ import annotations

from .base import Arbiter


class NoCoordinationArbiter(Arbiter):
    name = "no_coordination"

    def arbitrate(self, obs, groups):
        return {}
