"""Fuzzy decision layer (Step 6): acceptability A of each option.

Inputs per option:  risk R_hat (Low / Medium / High), uncertainty U (Low /
Medium / High), waiting time W of the conflict group (Short / Long).
Output: acceptability A in [0, 1] (Reject / Cautious / Accept).
Mamdani inference: AND = min, implication = min (clipping), aggregation =
max, centroid defuzzification. Membership functions are triangular or
trapezoidal (scikit-fuzzy ``trimf`` / ``trapmf``).

The membership functions are chosen so that two guarantees can be proved
(see :func:`acceptance_bounds` and docs/fuzzy_layer.md):

* Only the Accept set extends above A = 0.7, so A > 0.7 requires an Accept
  rule to fire. Accept rules require R_hat Low (< 0.2), or R_hat Medium
  (< 0.5) after a Long wait. Hence A > 0.7 with W <= 1 s implies R_hat < 0.2,
  so the accepted risk estimate is below 0.2.
* Only Cautious and Accept extend above A = 0.4, so A > 0.4 requires a
  non-Reject rule; with W <= 1 s those require R_hat < 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import skfuzzy as fuzz

# Universes of discourse.
UNIVERSES = {
    "risk": (0.0, 1.0),
    "uncertainty": (0.0, 2.0),
    "wait": (0.0, 6.0),
    "acceptability": (0.0, 1.0),
}

# (shape, parameters) for each linguistic term.
MEMBERSHIP = {
    "risk": {
        "low": ("trap", [0.0, 0.0, 0.1, 0.2]),
        "medium": ("tri", [0.1, 0.3, 0.5]),
        "high": ("trap", [0.35, 0.6, 1.0, 1.0]),
    },
    "uncertainty": {
        "low": ("trap", [0.0, 0.0, 0.1, 0.3]),
        "medium": ("tri", [0.1, 0.35, 0.6]),
        "high": ("trap", [0.4, 0.8, 2.0, 2.0]),
    },
    "wait": {
        "short": ("trap", [0.0, 0.0, 1.0, 3.0]),
        "long": ("trap", [1.0, 3.0, 6.0, 6.0]),
    },
    "acceptability": {
        "reject": ("trap", [0.0, 0.0, 0.2, 0.4]),
        "cautious": ("tri", [0.3, 0.5, 0.7]),
        "accept": ("trap", [0.6, 0.8, 1.0, 1.0]),
    },
}

# 13 rules: (antecedents, consequent). Antecedents are combined with AND.
RULES: list[tuple[dict[str, str], str]] = [
    ({"risk": "low", "uncertainty": "low"}, "accept"),
    ({"risk": "low", "uncertainty": "medium"}, "accept"),
    ({"risk": "low", "uncertainty": "high", "wait": "short"}, "cautious"),
    ({"risk": "low", "uncertainty": "high", "wait": "long"}, "accept"),
    ({"risk": "medium", "uncertainty": "low", "wait": "short"}, "cautious"),
    ({"risk": "medium", "uncertainty": "low", "wait": "long"}, "accept"),
    ({"risk": "medium", "uncertainty": "medium"}, "cautious"),
    ({"risk": "medium", "uncertainty": "high", "wait": "short"}, "reject"),
    ({"risk": "medium", "uncertainty": "high", "wait": "long"}, "cautious"),
    ({"risk": "high", "wait": "short"}, "reject"),
    ({"risk": "high", "uncertainty": "low", "wait": "long"}, "cautious"),
    ({"risk": "high", "uncertainty": "medium", "wait": "long"}, "reject"),
    ({"risk": "high", "uncertainty": "high"}, "reject"),
]

A_LOW = 0.4  # below: fallback (all slow down)
A_HIGH = 0.7  # above: accept (the vehicle going first speeds up)


def _mf(universe: np.ndarray, shape: str, params: list[float]) -> np.ndarray:
    return fuzz.trapmf(universe, params) if shape == "trap" else fuzz.trimf(universe, params)


@dataclass
class FuzzyAcceptability:
    resolution: int = 1001
    membership: dict = field(default_factory=lambda: MEMBERSHIP)
    rules: list = field(default_factory=lambda: RULES)

    def __post_init__(self) -> None:
        self.x = {k: np.linspace(lo, hi, self.resolution) for k, (lo, hi) in UNIVERSES.items()}
        self.mf = {
            var: {term: _mf(self.x[var], *spec) for term, spec in terms.items()}
            for var, terms in self.membership.items()
        }

    def degree(self, var: str, term: str, value) -> np.ndarray:
        lo, hi = UNIVERSES[var]
        value = np.clip(np.asarray(value, dtype=float), lo, hi)
        return np.interp(value, self.x[var], self.mf[var][term])

    def rule_strengths(self, risk, uncertainty, wait) -> np.ndarray:
        """Firing strength of every rule: shape (n_rules,) + input shape."""
        inputs = {"risk": risk, "uncertainty": uncertainty, "wait": wait}
        shape = np.broadcast(np.asarray(risk), np.asarray(uncertainty), np.asarray(wait)).shape
        out = np.empty((len(self.rules),) + shape)
        for r, (ante, _) in enumerate(self.rules):
            strength = np.ones(shape)
            for var, term in ante.items():
                strength = np.minimum(strength, self.degree(var, term, inputs[var]))
            out[r] = strength
        return out

    def infer(self, risk, uncertainty, wait) -> np.ndarray:
        """Acceptability A (centroid of the aggregated output set)."""
        strengths = self.rule_strengths(risk, uncertainty, wait)
        ax = self.x["acceptability"]
        agg = np.zeros(strengths.shape[1:] + ax.shape)
        for r, (_, cons) in enumerate(self.rules):
            clipped = np.minimum(strengths[r][..., None], self.mf["acceptability"][cons])
            agg = np.maximum(agg, clipped)
        area = agg.sum(axis=-1)
        a = np.where(area > 0, (agg * ax).sum(axis=-1) / np.maximum(area, 1e-12), 0.0)
        return a

    def explain(self, risk: float, uncertainty: float, wait: float) -> list[tuple[str, float]]:
        """Human-readable fired rules, strongest first (for explainable decisions)."""
        strengths = self.rule_strengths(risk, uncertainty, wait)
        out = []
        for (ante, cons), w in zip(self.rules, strengths):
            if w > 0:
                text = " AND ".join(f"{k} is {v}" for k, v in ante.items())
                out.append((f"IF {text} THEN {cons}", float(w)))
        return sorted(out, key=lambda t: -t[1])


def band(a: float, a_low: float = A_LOW, a_high: float = A_HIGH) -> str:
    return "accept" if a > a_high else "cautious" if a > a_low else "reject"


def acceptance_bounds(fz: FuzzyAcceptability | None = None) -> dict[str, float]:
    """Largest R_hat for which each band is reachable, from the rule base.

    Derived symbolically: a band above a threshold needs a rule whose
    consequent's support extends above that threshold; the risk terms in
    those rules bound R_hat by the right end of their support.
    """
    fz = fz or FuzzyAcceptability()

    def support_right(var, term):
        shape, p = fz.membership[var][term]
        return p[-1]

    def support_left(var, term):
        shape, p = fz.membership[var][term]
        return p[0]

    def max_risk(consequents, allow_long):
        best = 0.0
        for ante, cons in fz.rules:
            if cons not in consequents:
                continue
            if not allow_long and ante.get("wait") == "long":
                continue
            term = ante.get("risk")
            best = max(best, 1.0 if term is None else support_right("risk", term))
        return best

    above_high = [c for c in fz.membership["acceptability"]
                  if fz.membership["acceptability"][c][1][-1] > A_HIGH]
    above_low = [c for c in fz.membership["acceptability"]
                 if fz.membership["acceptability"][c][1][-1] > A_LOW]
    short_max = support_left("wait", "long")  # W at or below this: Long has degree 0
    return {
        "accept_short_wait": max_risk(above_high, allow_long=False),
        "accept_any_wait": max_risk(above_high, allow_long=True),
        "proceed_short_wait": max_risk(above_low, allow_long=False),
        "proceed_any_wait": max_risk(above_low, allow_long=True),
        "short_wait_limit": short_max,
    }
