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
