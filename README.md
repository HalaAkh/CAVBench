# CAVBench

**A simulation environment for multi-CAV conflict arbitration under perception noise, with what-if simulation and fuzzy decisions.**

CAVBench is the simulation environment of the study *"Uncertainty-aware counterfactual
risk arbitration for connected autonomous vehicles with fuzzy decision-making"*
(Kafel, El Akhrass and Watfa). It provides:

- **Scenarios:** an unsignalized four-way intersection configured after the
  CoDrivingLLM benchmark (Fang et al., 2025), a denser variant, and an on-ramp merge;
- **Perception noise:** Gaussian noise on the positions and speeds that vehicles
  share, at four levels;
- **Arbiters:** the proposed method (UCRA), first-come-first-served, no coordination,
  an LLM arbiter in the style of CoDrivingLLM and a safety-critical multi-agent MCTS
  (Lin et al., 2025), all using the same noisy shared states and the same low-level
  safety layer;
- **Metrics:** success, collisions, post-encroachment time (PET < 1.5 s = severe
  conflict), average speed and decision time;
- **Experiment protocol:** YAML configurations, matched random seeds, a parallel and
  resumable runner, and a statistics notebook;
- **Viewer:** watch any episode with its decisions, or save it as a video.

CAVBench uses the open-source [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
library (Leurent, 2018) for vehicle kinematics, road geometry and drawing, as does the
CoDrivingLLM benchmark.

## Installation

```bash
git clone https://github.com/HalaAkh/CAVBench.git
cd CAVBench
pip install -r requirements.txt
```

Python 3.10 or later. Run the tests with `pytest -q`.

## Quick start

```python
from cavbench.envs import make_env
from cavbench.arbiters import make_arbiter
from cavbench.runner import run_episode

env = make_env("intersection", {"noise_level": "high"})
print(run_episode(env, make_arbiter("whatif_fuzzy"), seed=0))
```

## Watching an episode

The experiments run without graphics, which is much faster. To **see** the roads,
the cars and the decisions of one episode, open it in a window:

```bash
python experiments/watch.py                                   # UCRA, intersection, high noise
python experiments/watch.py --arbiter fcfs --seed 3           # another method / situation
python experiments/watch.py --scenario intersection_dense --seed 7
python experiments/watch.py --scenario merge --noise none
python experiments/watch.py --seed 3 --save videos/ucra_seed3.mp4   # save a video (.mp4 or .gif)
```

Options: `--scenario` (intersection, intersection_dense, merge), `--arbiter`
(ucra, ucra_nou, ucra_nofallback, fcfs, nocoord, llm, llm_mock, llm_faithful,
llm_faithful_mock, mcts), `--noise` (none, low,
medium, high), `--seed`, `--speed` (playback speed), `--save` and `--no-trails`
(the recent path of each car is drawn by default, as in CoDrivingLLM's videos). The episode is
exactly the one the experiments run for the same settings.

On screen: blue cars are human drivers; numbered cars are CAVs, coloured by the
arbiter's command (green = go, yellow = cautious, orange = yield, red = crashed);
small rings show where vehicles *appear* to be in the noisy shared data; magenta
lines join CAVs in conflict; the top panel shows the latest decision with its risk,
uncertainty U and fuzzy acceptability A.

## Running the experiments

**Everything, with one command** (resumes where it stopped if interrupted):

```bash
python experiments/run_all.py              # E4, E5, E2, E3, E1 without the LLM, and E6
python experiments/run_all.py --with-llm   # the same, plus the LLM baseline (API key)
jupyter notebook analysis/statistics.ipynb # tables, statistical tests, figures
```

Use `--episodes 20` for a quick trial. The experiments one by one:

| Experiment | Config | Question |
|---|---|---|
| Smoke test | `configs/smoke.yaml` | does everything run? (2 episodes per condition) |
| E1 | `configs/e1_baselines.yaml` + `configs/e1_llm.yaml` + `configs/e1_mcts.yaml` | UCRA vs the LLM coordinator, MCTS, FCFS and no coordination (H1) |
| E2 | `configs/e2_uncertainty_ablation.yaml` | with vs without the uncertainty input U (H2) |
| E3 | `configs/e3_fallback_ablation.yaml` | with vs without the fallback (H3) |
| E4 | `configs/e4_rollout_sensitivity.yaml` | are N = 100 rollouts enough? (Theorem 1) |
| E5 | `configs/e5_realtime.yaml` | decision time (H4); run with `--workers 1` |
| E6 | `configs/e6_v2x_channel.yaml` | V2X delay and packet loss (H5) |

```bash
python experiments/run_experiment.py configs/e1_baselines.yaml --workers 8
python experiments/run_experiment.py configs/e5_realtime.yaml --workers 1
python experiments/rollout_sensitivity.py configs/e4_rollout_sensitivity.yaml
```

Results are written to `results/<experiment>.csv`, one row per episode, and every
method runs on the same seeds (same traffic, same noise samples). `--episodes N`
overrides the number of seeds (200 by default, from the power analysis).

### The LLM baseline (CoDrivingLLM-style)

As in CoDrivingLLM, which compared all its baselines inside its own simulator, the
LLM coordinator runs inside CAVBench on the same situations as the other methods.
It calls `gpt-4o-mini` through the OpenAI API:

1. Create a key at platform.openai.com (Settings → Billing: add about $10 of credit;
   API keys → Create new secret key). Never put the key in the code or on GitHub.
2. `pip install openai`, then give CAVBench the key, either way:
   * copy `.env.example` to `.env` (in the CAVBench folder) and write the key after
     `OPENAI_API_KEY=`. The `.env` file is git-ignored, so it never goes to GitHub;
   * or set the environment variable: Windows `setx OPENAI_API_KEY "sk-..."`
     (reopen the terminal); Mac/Linux `export OPENAI_API_KEY=sk-...`.
3. `python experiments/llm_check.py` makes one test call, runs 5 episodes and
   estimates the cost and time of the full run.
4. `python experiments/run_experiment.py configs/e1_llm.yaml --workers 8`, or
   `python experiments/run_all.py --with-llm`. The rows go into the same
   `results/e1_baselines.csv` as the other methods.

Every reply is cached in `results/llm_cache/`, so an interrupted run resumes without
paying twice; API errors are retried, and the run stops with a clear message if the
key is wrong or the API is unreachable. The CSV records the calls, tokens and cost
of each episode. To test the pipeline without a key, use the `mock` model
(`configs/llm_mock.yaml`, or `python experiments/watch.py --arbiter llm_mock`), which
answers with the arrival order. To run the original CoDrivingLLM code in its own
simulator, see `third_party/README.md`.

### LLM baseline: simple vs faithful mode

The LLM arbiter has two modes (`params: {mode: ...}`, or `kind: codrivingllm_faithful`):

| | simple (default) | faithful |
|---|---|---|
| When the LLM is called | only when a conflict exists | at **every** 0.2 s step |
| Calls per step | 1 per conflict group | 1 negotiation + 1 decision **per CAV** |
| What the LLM decides | the passing order → FASTER / SLOWER | advisory pair order, then each CAV's own action |
| Action safety check | – | unsafe actions removed before each decision |
| Memory (few-shot examples) | – | optional, `k = 2` similar past situations |
| Calls per intersection episode | ≈ 25 | ≈ 200–400 (+1 embedding request per step with memory) |

The faithful mode reproduces the CoDrivingLLM design (Fang et al., 2025) from the
paper's description; the remaining differences are listed in
[docs/llm_baseline.md](docs/llm_baseline.md). Because it calls the LLM about ten
times more often, run `python experiments/llm_check.py --mode faithful` first: it
estimates the cost and time of `configs/e1_llm_faithful.yaml` (intersection, noise
none and high, 50 seeds).

**Memory, built fairly.** The memory is built on separate *training* seeds (100000
and up, never used for evaluation) and then frozen, so evaluation episodes stay
independent and matched with the other methods:

```bash
python experiments/build_llm_memory.py --model gpt-4o-mini   # 20 training episodes -> results/llm_memory/
python experiments/llm_check.py --mode faithful              # cost / time estimate (uses the memory)
python experiments/run_experiment.py configs/e1_llm_faithful.yaml --workers 8
```

**Free test (no key, no network):** `configs/llm_faithful_mock.yaml`,
`python experiments/watch.py --arbiter llm_faithful_mock`, or
`python experiments/build_llm_memory.py --model mock` (local embeddings). The CSV
gets the extra columns `llm_calls_negotiation`, `llm_calls_decision`,
`llm_calls_embedding`, `memory_items` and `memory_hits`; `llm_cost_usd` includes
the embeddings. The key is read from `OPENAI_API_KEY` or the `.env` file, never
from code or configs.

### MCTS baseline (Lin et al., 2025)

`kind: mcts` re-implements the safety-critical multi-agent Monte Carlo tree search
of Lin, Lan, Anagnostopoulos, Tian and Flynn (*IEEE T-ITS* 26(11), 2025) from the
paper, because the authors' MATLAB code is not public. For each conflict group,
one tree search runs over the joint actions of the group's CAVs (SLOWER / IDLE /
FASTER each, so 3^K children per node). It uses Table I's settings: 1000
iterations, depth 8, UCB over the children that pass the V2V / V2H safety
check, random safe rollouts, and the multi-objective reward. Like the paper, it
treats the noisy shared states as exact. Every adaptation and every value the
paper does not give is listed in [docs/mcts_baseline.md](docs/mcts_baseline.md).

```bash
python experiments/run_experiment.py configs/mcts_smoke.yaml --workers 6   # 6 episodes, timing check
python experiments/run_experiment.py configs/e1_mcts.yaml --workers 8      # E1 rows -> results/e1_baselines.csv
python experiments/watch.py --arbiter mcts --seed 3
```

Any `MCTSConfig` value can be changed in a config, e.g.
`{label: MCTS-200, kind: mcts, params: {n_iterations: 200}}`. The CSV gets the
extra columns `mcts_iterations`, `mcts_nodes`, `mcts_pruned_fraction` (share of
nodes marked unsafe), `mcts_no_safe_root` (decisions with no safe option, where the
least unsafe one was used) and `mcts_max_group_size`. The search runs in Python,
so it is slow; see the run time in docs/mcts_baseline.md before starting E1.

## Method in brief (UCRA)

Every 0.2 s:

1. **Observe** the shared states with noise: ŝᵢ = sᵢ + εᵢ, εᵢ ~ N(0, σ²).
2. **Detect conflicts:** routes that meet, with ΔTTCP = |dᵢ/vᵢ − dⱼ/vⱼ| < 8 s. A vehicle
   counts as past a conflict point only once it is 4.5 m + 2σ beyond it, so position
   noise cannot make a conflict disappear for a moment.
3. **Candidate maneuvers:** one per conflicting CAV (it goes, the others yield),
   plus the fallback (everyone yields).
4. **What-if rollouts:** N = 100 simulations of each maneuver over T = 4 s, starting
   from the shared states plus imagined errors of at least σ_min (0.5 m, 0.3 m/s),
   so that the futures differ even with perfect perception. Human drivers follow IDM
   with their own desired speed in each rollout, and each one may give way: with
   probability 0.5 it follows the road's priority rule (stops when a conflict with a
   higher-priority or leading vehicle is predicted within 3 s), otherwise it drives on.
5. **Risk and uncertainty:** R̂ₖ = share of rollouts that are dangerous: two vehicles
   come within 2 m (three-circle footprints) **or** pass a conflict area less than
   1.5 s apart (PET, as in the evaluation); Uₖ = std of the closest gap / 5 m.
6. **Fuzzy decision:** 13 Mamdani rules over risk, uncertainty and waiting time give
   an acceptability Aₖ; > 0.7 accept, 0.4–0.7 proceed cautiously, ≤ 0.4 fall back.
   Ties go to the vehicle that arrives first, and "k goes" includes the CAVs queued
   ahead of k in its lane. See `docs/fuzzy_layer.md`.

## V2X channel: delay and packet loss (optional)

By default the shared states reach the arbiter instantly and without loss.
An optional V2X channel models an imperfect radio link:

```
true states -> [V2X: delay + loss] -> [Gaussian noise] -> [dead reckoning] -> arbiter
```

* **Delay**: the arbiter receives each vehicle's newest message sent at least
  `delay` seconds ago.
* **Packet loss**: with probability `loss` the newest message is lost and the
  arbiter keeps the last message it received from that vehicle (it gets older).
* **Dead reckoning** (for every arbiter, as in practice): the receiver moves each
  message forward along the vehicle's route by speed × age
  (`v2x_compensate: false` turns it off).
* **Age-aware uncertainty** (UCRA): in the rollouts, each vehicle's imagined error
  grows with the age of its message,
  σ_s(age)² = σ² + (σ_v·age)² + (a·age²/2)², σ_v(age)² = σ_v² + (a·age)², a = 3 m/s²,
  so stale information counts as less certain (`rollout: {age_aware: false}` turns
  it off; this is the UCRA-noAge ablation).

| Level | delay | loss probability | Meaning |
|---|---|---|---|
| ideal (default) | 0 s | 0 | perfect link |
| typical | 0.1 s | 0.1 | the 3GPP requirement for V2X safety messages (TR 22.885: ≤ 100 ms, ≥ 90% reception) |
| degraded | 0.3 s | 0.3 | stress case beyond the requirement (congestion, buildings blocking the signal) |

Assumptions: every vehicle broadcasts at 15 Hz, the first message of each vehicle
always arrives, and losses are independent. Use it through the env config:

```yaml
env: {v2x_level: typical}               # ideal / typical / degraded
env: {v2x_delay: 0.2, v2x_loss: 0.05}   # explicit values (override the level)
```

The age of the message used for each vehicle is in `obs.extra["v2x_age"]`. To watch
an episode: `python experiments/watch.py --v2x degraded`. Experiment E6
(`configs/e6_v2x_channel.yaml`) compares the three levels. The loss randomness uses
its own random generator, so the perception-noise samples stay matched across
experiments, and the ideal channel gives **identical results** to running without it.

## Low-level safety layer

All arbiters share the same low-level layer, so that differences come from the
arbitration. By default (`safety: follow`) it performs **car-following only**:
a CAV brakes behind a vehicle travelling the same way in its lane, in a merging
lane or in a lane that diverges from it. **Crossing conflicts are resolved only by
the arbiter**, as in the CoDrivingLLM benchmark. The following gap is measured
from noisy positions, so it includes a margin of 2·√2·σ (4.2 m at high noise) that
stops position errors from letting a CAV creep into the car ahead. With `params: {safety: full}` in a
config, CAVs also brake for any vehicle physically on their path ahead, including
crossing traffic; this stronger layer can hide arbitration errors.

## Repository layout

```
cavbench/
  envs/          intersection.py, merge.py, base.py (noise, trajectories, termination)
  noise/         gaussian.py (four noise levels), v2x.py (optional V2X delay + packet loss)
  arbiters/      whatif_core.py (rollouts), fuzzy.py, whatif_fuzzy.py (UCRA),
                 fcfs.py, no_coordination.py, codrivingllm_wrapper.py (LLM baseline),
                 llm_faithful.py (faithful-mode prompts and safety check),
                 llm_memory.py (memory of past decisions), mcts.py (MCTS baseline),
                 commitment.py
  metrics/       safety.py (success, collisions, PET, speed)
  conflicts.py   ΔTTCP conflict detection and grouping
  geometry.py    routes as polylines, meeting points
  runner.py      one episode with matched seeds
configs/         experiment configurations (YAML)
experiments/     run_all.py, run_experiment.py, rollout_sensitivity.py,
                 llm_check.py, build_llm_memory.py, watch.py (graphics)
analysis/        statistics.ipynb
docs/            fuzzy_layer.md, llm_baseline.md (faithful LLM mode vs the original),
                 mcts_baseline.md (MCTS baseline: adaptations and chosen values)
tests/           pytest suite
third_party/     how to run the original CoDrivingLLM
```

## Noise levels

| Level | σ position | σ speed |
|---|---|---|
| none | 0 | 0 |
| low | 0.2 m | 0.1 m/s |
| medium | 0.5 m | 0.3 m/s |
| high | 1.5 m | 1.0 m/s |

## Citation

If you use CAVBench, please cite the accompanying paper (details to be added on
publication) and HighwayEnv (Leurent, 2018).

## Licence

MIT, see `LICENSE`. CoDrivingLLM is not included (its repository has no licence).
