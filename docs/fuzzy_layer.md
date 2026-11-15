# Fuzzy decision layer

Implemented in `cavbench/arbiters/fuzzy.py`; used by `whatif_fuzzy.py`.

**Inputs** (per candidate maneuver *k*): risk R̂ₖ ∈ [0,1] (Low, Medium, High),
outcome uncertainty Uₖ ∈ [0,2] (Low, Medium, High), waiting time W ∈ [0,6] s
(Short, Long). **Output:** acceptability Aₖ ∈ [0,1] (Reject, Cautious, Accept).
Mamdani inference: AND = min, implication = min, aggregation = max, centroid
defuzzification.

| Variable | Term | Shape | Parameters |
|---|---|---|---|
| Risk | Low / Medium / High | trap / tri / trap | (0,0,0.1,0.2) / (0.1,0.3,0.5) / (0.35,0.6,1,1) |
| Uncertainty | Low / Medium / High | trap / tri / trap | (0,0,0.1,0.3) / (0.1,0.35,0.6) / (0.4,0.8,2,2) |
| Waiting time [s] | Short / Long | trap / trap | (0,0,1,3) / (1,3,6,6) |
| Acceptability | Reject / Cautious / Accept | trap / tri / trap | (0,0,0.2,0.4) / (0.3,0.5,0.7) / (0.6,0.8,1,1) |

**Decision.** k* = argmax Aₖ. A > 0.7: Accept (the vehicle going first speeds up);
0.4 < A ≤ 0.7: Cautious (it proceeds at 4.5 m/s); A ≤ 0.4: Reject (fallback,
all conflicting CAVs slow down). W grows by 0.2 s at each decision in fallback.

**Proposition 1.** Only the Accept set extends above A = 0.7, so A > 0.7 needs an
Accept rule to fire. With W ≤ 1 s, Long has degree 0, so only rules 1–2 (Risk Low,
R̂ < 0.2) can fire: hence R̂ < 0.2 and, with Theorem 1, R < 0.2 + γ. Similarly
A > 0.4 with W ≤ 1 s implies R̂ < 0.5, and A > 0.7 for any W implies R̂ < 0.5.
`acceptance_bounds()` derives these limits from the rule base, and
`tests/test_fuzzy.py` checks them numerically.

**Hard risk cap (for any waiting time).** For W > 1 s the rule base alone does
not bound the risk of a *Cautious* decision: after a long wait an option with
R̂ = 1 can reach A ≈ 0.41. The arbiter therefore rejects any option with
R̂ ≥ 0.5 (`risk_cap`), whatever A is. Hence, for every W: a vehicle proceeds
(Accept or Cautious) only if R̂ < 0.5, and with Theorem 1 (N = 100, K ≤ 4)
R < 0.5 + γ = 0.65 with probability ≥ 1 − δ = 0.9; an Accept with W ≤ 1 s still
gives R < 0.2 + γ = 0.35. The only exception is the existing *clear* rule: in
fallback, a CAV that can no longer stop before its conflict point may clear the
area if that is not riskier than everyone yielding (`reject_clear`). The cap is
part of the fallback, so the no-fallback ablation does not use it.
`tests/test_ucra_decision.py` checks the cap.

**Priority in the conflict area.** The CAV that had priority keeps it while it
can no longer stop before its conflict point, also across fallback steps and
steps in which the conflict groups change. Taking priority away from it would
leave it standing in the conflict area, where it blocks the other routes.

These two rules were added after a deadlock and a collision were traced on
tuning seeds (a CAV that had entered the junction lost priority, and after a
long wait another CAV proceeded "cautiously" with R̂ = 1). On 240 tuning
episodes they changed collisions from 1 to 0, time-outs from 6 to 5 and
success from 97.1 % to 97.9 %, with the same mean speed.
The remaining time-outs (about 2 %, all without perception noise) occur when
two CAVs stop close to each other's path and neither can proceed with
R̂ < 0.5; resolving them would need a new maneuver (e.g. reversing), which
the action space does not have.
