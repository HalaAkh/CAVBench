# The LLM baseline: our faithful mode vs the original CoDrivingLLM

CAVBench's LLM baseline (`cavbench/arbiters/codrivingllm_wrapper.py`) follows
CoDrivingLLM (Fang et al., *IEEE Transactions on Vehicular Technology*, vol. 74,
no. 8, pp. 11894–11905, Aug. 2025). It has two modes:

* **simple** (default): one LLM call per conflict group, only when a conflict
  exists; the passing order becomes FASTER (first) / SLOWER (the others).
  Zero-shot, no memory.
* **faithful**: the CoDrivingLLM design. At every 0.2 s decision step:
  1. **Negotiation**: one call to a central conflict coordinator, which decides
     which vehicle of each conflicting pair passes first (advisory). Pairs are
     described by vehicle ids, CAV or human-driven, route (turning / straight),
     speed, distance to the conflict point, TTCP, dTTCP and severity
     (< 2 s serious, 2–5 s general, 5–8 s slight, ≥ 8 s no danger), with the
     rule "turning vehicles yield to vehicles going straight".
  2. **Action safety check** for each CAV (`allowed_actions` in
     `cavbench/arbiters/llm_faithful.py`): FASTER is removed if the vehicle
     ahead is close, and IDLE too if it is too close; FASTER is removed if the
     negotiation says the CAV must yield in a general conflict, and FASTER and
     IDLE in a serious one. SLOWER is always allowed.
  3. **Decision**: one call per CAV, with its own state, nearby vehicles, its
     conflicts, the advisory negotiation result, traffic rules, cautions, the
     allowed actions and, optionally, the *k* most similar past situations from
     memory.
  4. **Memory** (optional, `cavbench/arbiters/llm_memory.py`): (situation,
     action, feedback) items, retrieved by cosine similarity of text embeddings
     and shown as few-shot examples. *k* = 2 by default, because the paper's
     90 % success at the intersection is the 2-shot version (0-shot: 74 %).

## Differences that remain (to report in the paper)

**Prompts and code.** The CoDrivingLLM repository has no licence, so no code or
prompt text was copied. All prompts were written from the paper's description,
so their wording, length and level of detail differ from the original.

**Embeddings and memory store.** We store the memory in our own JSONL file
(`results/llm_memory/<name>.jsonl`) and search it by brute-force cosine
similarity, instead of the original's vector-database setup. With the real
model the embeddings come from OpenAI `text-embedding-3-small`. With the `mock`
model and in the tests, a free local embedding (hashed bag of words) is used.
Neither is necessarily the embedding model of the original.

**Memory protocol.** As we understand the original, its memory keeps growing
while it runs. We **build** the memory on separate training seeds (100000+,
20 episodes by default, noise none and high in turn;
`experiments/build_llm_memory.py`) and then **freeze** it during evaluation.
Evaluation episodes therefore stay independent and matched with the other
methods (same seeds, same noise samples), and the results do not depend on the
order in which episodes run. Only steps where a CAV had a conflict are stored.
Retrieval happens before every decision call, including steps without a conflict.

**Feedback rule.** After each step, a CAV's decision is judged with the new
(noisy) state. The feedback is negative ("Your action intensified the conflict;
similar actions should be avoided") if:

* one of its conflicts moved to a more severe class; or
* a near miss is imminent: dTTCP < 1.5 s and both vehicles reach the point
  within 1.5 s. This is our proxy for PET < 1.5 s, because the true PET is only
  known after both vehicles have passed.

A collision at the end of the episode also gives negative feedback. Otherwise
the feedback is positive. The original's exact evaluation rule may differ.

**Action space.** Only longitudinal actions: SLOWER / IDLE / FASTER
(HighwayEnv's target-speed levels 0, 4.5 and 9 m/s). There are no lane changes
at the intersection, because the routes are fixed.

**States.** The original works with exact vehicle states. CAVBench gives every
arbiter, including the LLM, **noisy shared states** (Gaussian position and speed
noise, and optionally V2X delay and loss). Conflicts, prompts and the safety
check are all computed from these noisy states.

**Conflict detection.** Pairs come from CAVBench's `ConflictDetector`:

* distances are measured along the route (arc length), not in a straight line;
* the reaction range is 50 m;
* there is a noise margin for "already past the point";
* CAV–CAV and CAV–human pairs are included; human–human pairs are not.

Whether a vehicle turns is decided from the heading change along its route
(more than 45°).

**Safety layers.** The action safety check reuses CAVBench's car-following rule
(`FollowingSafety.cap`), plus the yield rule above; the original's exact list of
removed actions may differ. As for every CAVBench arbiter, the same low-level
car-following layer is applied after the decisions.

**Decisions within a step.** The decision calls for the CAVs are made one after
the other and independently: a CAV does not see the other CAVs' decisions from
the same step.

**Unusable answers.**

* If the negotiation reply cannot be read, or misses a pair, that pair falls
  back to arrival order (earlier TTCP first).
* If a decision is invalid or not allowed, the CAV takes the most cautious
  allowed action (SLOWER).

Both cases are counted in `llm_parse_failures`, and API errors in `llm_failures`.

**Model and settings.** All CAVBench LLM runs use `gpt-4o-mini` at temperature 0.
Replies are cached by prompt (`results/llm_cache/`). The model and decoding
settings of the original may differ; check the paper before reporting them.

**Scenarios.** The faithful mode runs in CAVBench's HighwayEnv scenarios
(intersection: 4 CAVs plus human-driven traffic; policy step 0.2 s). These
match the original's CoDrivingLLM setup only approximately, so its 90 % / 74 %
figures are a reference, not a number we expect to reproduce exactly.

## Cost

A faithful intersection episode lasts about 50–100 steps (10–20 s) and makes
1 + (number of CAVs still driving) chat calls per step: about 200–400 calls,
measured with the mock model. With memory, there is one more (batched)
embedding request per step. `python experiments/llm_check.py --mode faithful`
measures the real cost and time on 5 episodes and extrapolates to
`configs/e1_llm_faithful.yaml`.
