"""Optional V2X communication channel: message delay and packet loss.

The channel acts on the TRUE vehicle states, before the Gaussian perception
noise is added:

    true states -> [V2X: delay + loss] -> [Gaussian noise] -> arbiter

* **Delay**: the arbiter receives, for each vehicle, the newest message sent at
  least ``delay`` seconds ago.
* **Packet loss**: with probability ``loss_prob`` the newest message of a
  vehicle is lost; the arbiter then keeps the last message it received from
  that vehicle, which becomes older and older (its *age* grows).

Assumptions:

* every vehicle broadcasts its state at the simulation frequency (15 Hz);
* the first message of each vehicle always arrives (the arbiter always knows
  every vehicle on the road);
* losses are independent between vehicles and between decision steps.

With the default settings (no delay, no loss) the channel is *ideal* and is
skipped entirely, so results are identical to running without it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

#: V2X levels: (delay [s], packet-loss probability).
#: * typical = the 3GPP requirement for V2X safety messages (3GPP TR 22.885,
#:   Release 14): at most 100 ms latency and at least 90% message reception
#:   (90% for NLOS/urban roads, 95% at urban intersections), i.e. a link that
#:   just meets the standard;
#: * degraded = a stress case beyond the requirement (3x the latency budget,
#:   30% loss), as seen with congestion or when buildings block the signal:
#:   in a DSRC field test, packet reception at an NLOS intersection stayed
#:   >= 90% only up to 320 m at 5 dBm (Bae et al., IEEE Access 9, 2021,
#:   doi:10.1109/ACCESS.2020.3044358) and fell further beyond.
V2X_LEVELS: dict[str, tuple[float, float]] = {
    "ideal": (0.0, 0.0),
    "typical": (0.1, 0.1),
    "degraded": (0.3, 0.3),
}

_EPS = 1e-9  # tolerance for floating-point simulation times


@dataclass
class V2XChannel:
    delay: float = 0.0
    loss_prob: float = 0.0
    # internal state (not constructor arguments)
    _history: dict = field(default_factory=dict, init=False, repr=False)
    _last: dict = field(default_factory=dict, init=False, repr=False)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.delay = float(self.delay)
        self.loss_prob = float(self.loss_prob)
        if self.delay < 0:
            raise ValueError(f"V2X delay must be >= 0, got {self.delay}")
        if not 0.0 <= self.loss_prob < 1.0:
            raise ValueError(f"V2X loss_prob must be in [0, 1), got {self.loss_prob}")

    @classmethod
    def from_level(cls, level: str) -> "V2XChannel":
        if level not in V2X_LEVELS:
            raise ValueError(f"Unknown V2X level {level!r}; choose from {list(V2X_LEVELS)}")
        return cls(*V2X_LEVELS[level])

    @property
    def is_ideal(self) -> bool:
        return self.delay == 0.0 and self.loss_prob == 0.0

    def reset(self) -> None:
        """Forget everything (call at the start of every episode)."""
        self._history = {}
        self._last = {}
        self._rng = None

    def record(self, t: float, vid: int, position, speed: float) -> None:
        """Store the TRUE state broadcast by vehicle ``vid`` at time ``t``."""
        hist = self._history.setdefault(vid, deque())
        hist.append((float(t), np.array(position, dtype=float), float(speed)))
        # Drop messages that can never be delivered again: keep only the newest
        # one that is already older than ``delay`` (plus all the newer ones).
        cutoff = t - self.delay + _EPS
        while len(hist) >= 2 and hist[1][0] <= cutoff:
            hist.popleft()

    def _candidate(self, t: float, vid: int):
        """Newest message sent at or before ``t - delay`` (else the oldest one)."""
        hist = self._history[vid]
        cutoff = t - self.delay + _EPS
        best = hist[0]
        for msg in hist:
            if msg[0] <= cutoff:
                best = msg
            else:
                break
        return best

    def receive(self, t: float, vids, rng: np.random.Generator):
        """States the arbiter receives at time ``t`` for vehicles ``vids``.

        Returns ``positions`` (M, 2), ``speeds`` (M,) and ``ages`` (M,), where
        age = t - send time of the message used.

        The loss randomness comes from a child generator spawned from ``rng``
        once per episode, so the samples drawn from ``rng`` itself (the
        perception noise) are not consumed and stay matched across experiments.
        """
        if self._rng is None:
            self._rng = rng.spawn(1)[0]
        vids = list(vids)
        positions = np.zeros((len(vids), 2))
        speeds = np.zeros(len(vids))
        ages = np.zeros(len(vids))
        for i, vid in enumerate(vids):
            msg = self._candidate(t, vid)
            if vid not in self._last:
                self._last[vid] = msg  # the first message always arrives
            elif self._rng.random() >= self.loss_prob:
                self._last[vid] = msg  # delivered
            # else: lost -> keep the last received message
            send_t, pos, spd = self._last[vid]
            positions[i] = pos
            speeds[i] = spd
            ages[i] = t - send_t
        return positions, speeds, ages
