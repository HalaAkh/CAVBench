# CoDrivingLLM in CAVBench: reproduced from the released code

`kind: codrivingllm_repro` (file `cavbench/arbiters/codrivingllm_repro.py`) is
the CoDrivingLLM baseline used in the paper. It re-implements the code of
<https://github.com/FanGShiYuu/CoDrivingLLM> (Fang et al., IEEE T-VT 74(8), 2025)
function by function, inside CAVBench, so that UCRA and CoDrivingLLM run in
the same environment, on the same seeds and the same (noisy) shared states.

The older modes `codrivingllm` (simple) and `codrivingllm_faithful` were written
from the paper's text; they are kept but are **not** the released method.

## What is identical to the released code

| Part | Released code (file) | CAVBench reproduction |
|---|---|---|
| Model and call | gpt-4o-mini, prompt as one *system* message, temperature not set (`llm_agent_action.py`, `llm_agent_negotiation_system.py`) | same |
| Calls per 0.2 s step | 1 negotiation call (even with no conflict) + 1 decision call per CAV (`Run_multi_CAV_LLM.py`) | same |
| Who negotiates | CAV-CAV pairs only at the intersection (`detect_conflicts`) | same |
| Conflict test | routes sampled 1000 points per lane, closer than 1 m; straight-line distance to the first such point; both vehicles within 50 m and before the point (`is_conflict`) | same (same arithmetic) |
| Negotiation prompt and parsing | "traffic police officer" prompt; `first_vehicle` / `second_vehicle` regex | same text, same regex |
| Speed cap | each CAV's speed set to 5 m/s if higher, just before its prompt is built (`llm_controller_run`) | same, on the simulated vehicle |
| Scenario text | available actions, lane, vehicle ahead/behind, TTC safety tools (accelerate / keep speed / decelerate), dTTCP conflict safety with `cal_ttcp` (a = 2 m/s², 5 m/s limit, levels 8/5/2 s), "most dangerous conflict" line (`prompt_llm.py`) | same text and numbers |
| Decision prompt and parsing | prefix, traffic rules, cautions, empty memory, negotiation advice; `extract_decision` | same text, same parsing |
| Memory | not used (`past_memory = ''`; the shipped memory database is empty) | not used |
| CAVs | MDPVehicle, target speeds 0/5/10/15/20 m/s, initial target 5 m/s, no acceleration limit, no safety layer (`safety_guarantee: False`) | same |

All prompt texts are the benchmark's, copied verbatim, with attribution.

**Check:** `tests/test_codrivingllm_repro.py` runs the original modules and the
reproduction side by side on the same vehicles at every step of CAVBench
episodes (with and without noise). It checks that both send the LLM exactly the
same prompts and turn the same answers into the same actions. Run it with a copy
of the benchmark in `third_party/CoDrivingLLM` (git-ignored):

    git clone https://github.com/FanGShiYuu/CoDrivingLLM third_party/CoDrivingLLM
    python -m pytest tests/test_codrivingllm_repro.py

## Remaining differences (all outside the method)

1. **Road and traffic.** CAVBench's own intersection generator (same layout,
   4 CAVs, 10 IDM vehicles, challenger, 50 s, 15/5 Hz, CAV start 65 ± 5 m at
   5 m/s), on HighwayEnv 1.12 rather than the benchmark's 2021 fork. One quirk is
   not copied: in the benchmark, a CAV's exit is drawn from o1-o3 only, so no CAV
   ever leaves towards o0. This is the same for every method.
2. **Shared states.** The method reads positions and speeds from CAVBench's shared
   observation. With `noise: none` and ideal V2X these are the true states, as in
   the benchmark. With noise or V2X delay, CoDrivingLLM gets the same corrupted
   states as every other method. That is the experimental condition, not a change
   to the method.
3. **Arrived CAVs leave the scene.** In the benchmark they stay on the road, and
   the LLM is still asked about them. This only affects cost.
4. **Unusable answer.** If the reply contains none of IDLE / FASTER / SLOWER, the
   benchmark's environment fails on action id -1 and the run stops. Here the CAV
   keeps its target speed (IDLE), and the case is counted in `llm_parse_failures`.
5. **Bug fix.** In the released `check_safety_in_current_lane`, the acceleration
   tool is called without its (unused) `env` argument. The original program
   therefore stops with a TypeError whenever a CAV has a vehicle ahead in its lane
   and none behind. The reproduction calls the tool as intended.
6. **Parallel decision calls.** The per-CAV decision calls do not depend on each
   other, so they are sent together. The prompts and answers are the same as
   sending them one by one.

## Options (defaults = as released)

`vehicle: cavbench` uses CAVBench's CAVs (speeds 0/4.5/9, acceleration limits
-5/+3 m/s²) instead of the benchmark's. `speed_cap: false` removes the 5 m/s cap.
`safety: follow` adds CAVBench's shared car-following layer. `temperature: 0`
sends a temperature. These options exist only for sensitivity checks. The paper
uses the defaults.

## Cost

An intersection episode takes about 70-150 steps. That is one negotiation call
per step plus about 230-380 decision calls (about 1 000 input tokens each). With
gpt-4o-mini this comes to about 0.1-0.25 USD per episode, so roughly 12-25 USD
for the 100 episodes of `configs/e1_codrivingllm.yaml`. Run 2 episodes first
(see that file) to measure the real cost.
