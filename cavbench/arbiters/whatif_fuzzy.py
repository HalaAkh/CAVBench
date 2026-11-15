"""What-if arbitration with the fuzzy decision layer (proposed method).

Steps 1-5 simulate every option; Step 6 decides with fuzzy inference:

    A_k = Mamdani(R_hat_k, U_k, W),   k* = argmax_k A_k
    A_k* >  0.7       -> Accept:   the vehicle going first speeds up, others yield
    0.4 < A_k* <= 0.7 -> Cautious: it proceeds at a moderate constant speed
    A_k* <= 0.4       -> Reject:   fallback, conflicting CAVs slow down

W is how long the group has been held in fallback. Ties in A are broken by
lower R_hat, lower U, then earlier arrival. "k goes" includes the CAVs queued
ahead of k in its lane. See commitment.py for priority persistence and for
letting CAVs that can no longer stop clear the conflict area in fallback.

Deadlock breaker (liveness rule): fallback is the safe choice while traffic is
moving, but two CAVs can end up stopped in each other's way, e.g. one half-way
through its turn and the other at its stop line, so that every "k goes" option
passes the other car with slightly less than d_safe = 2 m of clearance and is
rejected for ever. If the group has been in fallback for ``deadlock_wait``
seconds, the option with the least risk of actual contact (rollouts with
footprint clearance below ``deadlock_clearance`` = 0.5 m) is released at the
cautious speed, provided that risk is at most ``deadlock_risk``; that CAV keeps
the right of way until it has left the conflict (or the contact risk rises
above ``deadlock_risk``). All other rules, the rollouts and the shared safety
layer still apply.
Ablations: use_uncertainty=False (U fixed at 0), use_fallback=False,
use_deadlock_breaker=False.
"""
from __future__ import annotations

import numpy as np

from .base import FASTER, SLOWER, Arbiter, toward_speed
from .commitment import arrival_time, committed_cavs, pick_with_persistence, platoon_ahead
from .fuzzy import A_HIGH, A_LOW, FuzzyAcceptability, band
from .whatif_core import RolloutConfig, evaluate_options


class WhatIfFuzzyArbiter(Arbiter):
    name = "whatif_fuzzy"

    def __init__(self, use_uncertainty: bool = True, use_fallback: bool = True,
                 a_low: float = A_LOW, a_high: float = A_HIGH,
                 persistence_margin: float = 0.05, rollout: dict | None = None,
                 use_deadlock_breaker: bool = True, deadlock_wait: float = 2.0,
                 deadlock_clearance: float = 0.5, deadlock_risk: float = 0.1, **kw) -> None:
        super().__init__(**kw)
        self.use_uncertainty = use_uncertainty
        self.use_fallback = use_fallback
        self.a_low, self.a_high = a_low, a_high
        self.persistence_margin = persistence_margin
        self.rollout = RolloutConfig(**(rollout or {}))
        self.fuzzy = FuzzyAcceptability()
        self.use_deadlock_breaker = use_deadlock_breaker
        self.deadlock_wait = deadlock_wait
        self.deadlock_clearance = deadlock_clearance
        self.deadlock_risk = deadlock_risk
        self.wait: dict[int, float] = {}
        self.prev_goers: set[int] = set()
        self.released: set[int] = set()  # CAVs given the right of way by the deadlock breaker
        self.n_deadlock = 0

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self.wait, self.prev_goers, self.released = {}, set(), set()
        self.n_deadlock = 0

    def extra_metrics(self) -> dict:
        return {"n_deadlock": self.n_deadlock}

    def _contact_risk(self, r) -> float:
        """Share of rollouts in which the footprints come closer than
        deadlock_clearance (near-contact), the criterion of the deadlock breaker."""
        return float(np.mean(np.asarray(r.d_min) < self.deadlock_clearance))

    def arbitrate(self, obs, groups):
        actions, in_group, goers_now = {}, set(), set()
        for g in groups:
            in_group.update(g.cavs)
            res = evaluate_options(obs, g, self.rollout, self.rng, include_fallback=True)
            fb = res.pop()  # the "everyone yields" option
            w = max(self.wait.get(v, 0.0) for v in g.cavs)
            risk = [r.risk for r in res]
            unc = [r.uncertainty if self.use_uncertainty else 0.0 for r in res]
            acc = [float(a) for a in self.fuzzy.infer(risk, unc, [w] * len(res))]
            goers = [r.goer for r in res]
            prev = next((v for v in g.cavs if v in self.prev_goers), None)
            k = pick_with_persistence(acc, goers, prev, self.persistence_margin, True,
                                      [(risk[i], unc[i], arrival_time(g, goers[i]))
                                       for i in range(len(res))])
            mode = band(acc[k], self.a_low, self.a_high)
            if mode == "reject" and not self.use_fallback:
                mode = "cautious"
            goer = res[k].goer
            held = next((v for v in g.cavs if v in self.released), None)
            if held is not None and not (mode == "accept" and goer == held):
                # keep the right of way given by the deadlock breaker, while safe
                if self._contact_risk(res[goers.index(held)]) <= self.deadlock_risk:
                    goer, mode = held, "deadlock"
                else:
                    self.released.discard(held)
            if mode == "reject":
                goer = None
                committed = committed_cavs(obs, g, self.rollout.a_yield)
                if committed:  # a CAV that cannot stop may clear the area, if safer
                    c = min(committed, key=lambda v: res[goers.index(v)].risk)
                    if res[goers.index(c)].risk < fb.risk:
                        goer, mode = c, "reject_clear"
            if mode == "reject" and self.use_deadlock_breaker and w >= self.deadlock_wait:
                contact = [self._contact_risk(r) for r in res]
                j = min(range(len(res)), key=lambda i: (contact[i], arrival_time(g, goers[i])))
                if contact[j] <= self.deadlock_risk:
                    goer, mode = goers[j], "deadlock"
                    self.released.add(goer)
                    self.n_deadlock += 1
            moving = {goer, *platoon_ahead(obs, g, goer)} if goer is not None else set()
            for v in g.cavs:
                if v not in moving:
                    actions[v] = SLOWER
                elif mode in ("accept", "reject_clear"):
                    actions[v] = FASTER
                else:  # cautious / deadlock release: moderate constant speed
                    actions[v] = toward_speed(obs.states[v].target_speed, obs.v_mid)
                rejected = mode.startswith("reject")
                self.wait[v] = self.wait.get(v, 0.0) + obs.policy_dt if rejected else 0.0
            if goer is not None:
                goers_now.add(goer)
            self.log.append({
                "time": obs.time, "group": list(g.cavs), "goer": goer, "risk": risk,
                "uncertainty": unc, "acceptability": acc, "fallback_risk": fb.risk, "wait": w,
                "mode": "fallback" if mode == "reject" else mode,
            })
        for v in list(self.wait):
            if v not in in_group:
                self.wait[v] = 0.0
        self.released &= in_group  # the released CAV has left the conflict
        self.prev_goers = goers_now
        return actions
