"""What-if arbitration with the fuzzy decision layer (proposed method).

Steps 1-5 simulate every option; Step 6 decides with fuzzy inference:

    A_k = Mamdani(R_hat_k, U_k, W),   k* = argmax_k A_k
    A_k* >  0.7       -> Accept:   the vehicle going first speeds up, others yield
    0.4 < A_k* <= 0.7 -> Cautious: it proceeds at a moderate constant speed
    A_k* <= 0.4       -> Reject:   fallback, conflicting CAVs slow down

Hard risk cap: an option with R_hat >= risk_cap (0.5) never proceeds, whatever
A is, so the bound R_hat < 0.5 holds for every waiting time (docs/fuzzy_layer.md).
The CAV that had priority keeps it while it can no longer stop before its
conflict point (also across fallback steps).

W is how long the group has been held in fallback. Ties in A are broken by
lower R_hat, lower U, then earlier arrival. "k goes" includes the CAVs queued
ahead of k in its lane. See commitment.py for priority persistence and for
letting CAVs that can no longer stop clear the conflict area in fallback.
Ablations: use_uncertainty=False (U fixed at 0), use_fallback=False.
"""
from __future__ import annotations

from .base import FASTER, SLOWER, Arbiter, toward_speed
from .commitment import arrival_time, committed_cavs, pick_with_persistence, platoon_ahead
from .fuzzy import A_HIGH, A_LOW, FuzzyAcceptability, band
from .whatif_core import RolloutConfig, evaluate_options


class WhatIfFuzzyArbiter(Arbiter):
    name = "whatif_fuzzy"

    def __init__(self, use_uncertainty: bool = True, use_fallback: bool = True,
                 a_low: float = A_LOW, a_high: float = A_HIGH,
                 persistence_margin: float = 0.05, risk_cap: float = 0.5,
                 rollout: dict | None = None, **kw) -> None:
        super().__init__(**kw)
        self.use_uncertainty = use_uncertainty
        self.use_fallback = use_fallback
        self.a_low, self.a_high = a_low, a_high
        self.persistence_margin = persistence_margin
        self.risk_cap = risk_cap  # an option with R_hat >= risk_cap never proceeds
        self.rollout = RolloutConfig(**(rollout or {}))
        self.fuzzy = FuzzyAcceptability()
        self.wait: dict[int, float] = {}
        self.prev_goers: set[int] = set()
        self.holders: set[int] = set()  # had priority and cannot stop any more

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self.wait, self.prev_goers, self.holders = {}, set(), set()

    def arbitrate(self, obs, groups):
        actions, in_group, goers_now, committed_all = {}, set(), set(), set()
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
            committed = committed_cavs(obs, g, self.rollout.a_yield)
            committed_all.update(committed)
            holder = next((v for v in g.cavs if v in self.holders and v in committed), None)
            if holder is not None and holder in goers:
                prev = holder
            if prev in committed and prev in goers:
                # The CAV that had priority keeps it while it can no longer stop
                # before its conflict point: taking priority away would leave it
                # standing in the conflict area, where it blocks the others.
                k = goers.index(prev)
            mode = band(acc[k], self.a_low, self.a_high)
            if self.use_fallback and risk[k] >= self.risk_cap:
                # Hard cap: a long wait W may raise A, but never lets an option
                # with a high estimated risk proceed (cf. docs/fuzzy_layer.md).
                mode = "reject"
            if mode == "reject" and not self.use_fallback:
                mode = "cautious"
            goer = res[k].goer
            if mode == "reject":
                goer = None
                if committed:  # a CAV that cannot stop may clear the area, unless
                    # that is riskier than everyone yielding. Ties go to clearing:
                    # a CAV standing in the conflict area otherwise blocks it for good.
                    c = (holder if holder in committed and holder in goers
                         else min(committed, key=lambda v: res[goers.index(v)].risk))
                    if res[goers.index(c)].risk <= fb.risk:
                        goer, mode = c, "reject_clear"
            moving = {goer, *platoon_ahead(obs, g, goer)} if goer is not None else set()
            for v in g.cavs:
                if v not in moving:
                    actions[v] = SLOWER
                elif mode in ("accept", "reject_clear"):
                    actions[v] = FASTER
                else:  # cautious: moderate constant speed
                    actions[v] = toward_speed(obs.states[v].target_speed, obs.v_mid)
                # W grows while the group is held in fallback and is reset only
                # when an option is accepted; a cautious step leaves it unchanged
                # (resetting it there let W cycle below "Long" in a deadlock).
                if mode.startswith("reject"):
                    self.wait[v] = self.wait.get(v, 0.0) + obs.policy_dt
                elif mode == "accept":
                    self.wait[v] = 0.0
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
        self.prev_goers = goers_now
        # Priority survives fallback steps, and steps in which the holder is in no
        # group, until the holder is in a group again but can still stop (i.e. it
        # has not entered the conflict area) or has left the scene.
        self.holders = goers_now | {v for v in self.holders if v in obs.cav_ids
                                    and (v not in in_group or v in committed_all)}
        return actions
