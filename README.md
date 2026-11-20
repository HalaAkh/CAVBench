# CAVBench

**A simulation benchmark for multi-CAV conflict arbitration under noisy, delayed V2X data, with what-if simulation and fuzzy decisions.**

CAVBench is the simulation environment of the study *"Uncertainty-aware counterfactual
risk arbitration for connected autonomous vehicles with fuzzy decision-making"*
(Kafel, El Akhrass and Watfa). It provides:

- **Scenarios:** an unsignalized four-way intersection configured after the
  CoDrivingLLM benchmark (Fang et al., 2025), a denser variant (8 CAVs), and an
  on-ramp merge, all with human-driven vehicles;
- **Benchmark mode** (default): the CoDrivingLLM simulator's rules for what counts as
  a crash and which traffic situations occur, so results can be compared with that
  paper ([docs/benchmark_mode.md](docs/benchmark_mode.md));
- **Imperfect shared data:** Gaussian noise on the positions and speeds that vehicles
  share (four levels) and an optional V2X channel with delay and packet loss;
- **Arbiters:** the proposed method (UCRA) and five baselines:
  * no coordination (lower bound);
  * classical first-come-first-served (FCFS), one vehicle in the conflict area at a
    time ([docs/fcfs_baseline.md](docs/fcfs_baseline.md));
  * CoDrivingLLM (Fang et al., IEEE T-VT 2025), **reproduced from the authors'
    released code** and verified against it, run as released and with its memory
    module ([docs/codrivingllm_repro.md](docs/codrivingllm_repro.md));
  * safety-critical multi-agent MCTS (Lin et al., IEEE T-ITS 2025)
    ([docs/mcts_baseline.md](docs/mcts_baseline.md));
  * an oracle with perfect information (upper bound)
    ([docs/oracle_baseline.md](docs/oracle_baseline.md));
- **Metrics:** success (no collision and all CAVs arrived), collision-free rate,
  collisions, time-outs, severe conflicts (PET < 1.5 s), episode time, speed, decision
  time, and CoDrivingLLM's Table I metrics (PET and travel velocity, average / max /
  min) ([docs/benchmark_metrics.md](docs/benchmark_metrics.md));
- **Experiment protocol:** YAML configurations, matched random seeds, a parallel and
  resumable runner, a statistics report with paired tests
  (`experiments/final_report.py`) and the paper's figures (`experiments/make_figures.py`);
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
print(run_episode(env, make_arbiter("whatif_fuzzy"), seed=1000))
```

## Watching an episode

The experiments run without graphics, which is much faster. To **see** the roads,
the cars and the decisions of one episode, open it in a window:

```bash
python experiments/watch.py                                   # UCRA, intersection, high noise
python experiments/watch.py --arbiter fcfs --seed 3           # another method / situation
python experiments/watch.py --scenario intersection_dense --seed 7
python experiments/watch.py --scenario merge --noise none
python experiments/watch.py --arbiter codrivingllm_mock       # CoDrivingLLM pipeline, no API key
python experiments/watch.py --seed 3 --save videos/ucra_seed3.mp4   # save a video (.mp4 or .gif)
```

Options: `--scenario` (intersection, intersection_dense, merge), `--arbiter`
(ucra, ucra_nou, ucra_nofallback, fcfs, nocoord, oracle, codrivingllm,
codrivingllm_mock, mcts, mcts_strict, and the earlier LLM variants llm, llm_mock,
llm_faithful, llm_faithful_mock), `--noise` (none, low, medium, high), `--v2x`
(ideal, typical, degraded), `--seed`, `--speed` (playback speed), `--save` and
`--no-trails` (the recent path of each car is drawn by default, as in CoDrivingLLM's
videos). The episode is exactly the one the experiments run for the same settings.

On screen: blue cars are human drivers; numbered cars are CAVs, coloured by the
arbiter's command (green = go, yellow = cautious, orange = yield, red = crashed);
small rings show where vehicles *appear* to be in the noisy shared data; magenta
lines join CAVs in conflict; the top panel shows the latest decision with its risk,
uncertainty U and fuzzy acceptability A.

## Running the experiments

**Everything, with one command** (resumes where it stopped if interrupted):

```bash
python experiments/run_all.py              # E4, E5, E2, E3, E1 (without CoDrivingLLM), E6, E7, E7b
python experiments/run_all.py --with-llm   # the same, plus CoDrivingLLM as released (API key)
python experiments/final_report.py         # tables and paired tests -> results/final_report.md
python experiments/make_figures.py         # figures -> results/figures/ (PDF + PNG)
```

MCTS, the oracle and CoDrivingLLM with memory are run separately (see their sections
below), because they are slow or paid. Use `--episodes 20` for a quick trial.

| Experiment | Config | Question |
|---|---|---|
| Smoke test | `configs/smoke.yaml` | does everything run? (2 episodes per condition) |
| E1 | `e1_baselines.yaml`, `e1_mcts.yaml`, `e1_oracle.yaml`, `e1_codrivingllm.yaml`, `e1_codrivingllm_memory_build.yaml` + `e1_codrivingllm_memory.yaml` | UCRA vs no coordination, FCFS, MCTS, CoDrivingLLM and the oracle (H1) |
| E2 | `configs/e2_uncertainty_ablation.yaml` | ablation: with vs without the uncertainty input U (H2) |
| E3 | `configs/e3_fallback_ablation.yaml` | ablation: with vs without the fallback (H3) |
| E4 | `configs/e4_rollout_sensitivity.yaml` | are N = 100 rollouts enough? (Theorem 1) |
| E5 | `configs/e5_realtime.yaml` | decision time within the 0.2 s interval (H4); run with `--workers 1` |
| E6 | `configs/e6_v2x_channel.yaml` | robustness to V2X delay and packet loss; age-aware spread ablation (H5) |
| E7 | `configs/e7_sensitivity.yaml` | sensitivity: 8 parameters changed one at a time |
| E7b | `configs/e7_noise_mismatch.yaml` | robustness: UCRA assumes a wrong noise level |

```bash
python experiments/run_experiment.py configs/e1_baselines.yaml --workers 8
python experiments/run_experiment.py configs/e5_realtime.yaml --workers 1
python experiments/rollout_sensitivity.py configs/e4_rollout_sensitivity.yaml
```

Results are written to `results/<experiment>.csv`, one row per episode, and every
method runs on the same seeds (same traffic, same noise samples). The reported
results use **evaluation seeds from 1000 onwards**; seeds 0–199 and 100000+ were used
for development and tuning. `--episodes N` overrides the number of seeds (200 by
default, from the power analysis).

### Statistics and figures

`experiments/final_report.py` reads all result files and writes
`results/final_report.md`: summary tables with Wilson 95 % intervals, paired tests
of UCRA against each method on the same episodes (exact McNemar test for success,
collisions and severe conflicts; Wilcoxon signed-rank test for episode time), the
Table I format of CoDrivingLLM, the gap to the oracle, robustness, ablation,
sensitivity and real-time results. Open it in VS Code (Ctrl+Shift+V) or on GitHub to
see the tables formatted. `experiments/make_figures.py` draws the paper's figures
from the same files into `results/figures/`.

### CoDrivingLLM (reproduced from its released code)

`kind: codrivingllm_repro` re-implements CoDrivingLLM's released code
([FanGShiYuu/CoDrivingLLM](https://github.com/FanGShiYuu/CoDrivingLLM)) step by step:
the negotiation call every step, one decision call per CAV with the authors' prompts
and safety tools, their vehicle model (target speeds 0/5/10/15/20 m/s, 5 m/s cap at
the intersection) and no extra safety layer, as released, with `gpt-4o-mini`. A test
runs the original modules side by side with the reproduction at every step and
checks that both send the same prompts and turn the same answers into the same
actions (`tests/test_codrivingllm_repro.py`; it needs a local copy of the original
repository in `third_party/CoDrivingLLM`, which is git-ignored because that
repository has no licence). Remaining differences: [docs/codrivingllm_repro.md](docs/codrivingllm_repro.md).

**API key.** Create a key at platform.openai.com and add credit. Never put the key
in the code or on GitHub: copy `.env.example` to `.env` and write it after
`OPENAI_API_KEY=` (`.env` is git-ignored), or set the environment variable.

```bash
python experiments/run_experiment.py configs/codrivingllm_mock.yaml      # free pipeline test (mock model)
python experiments/run_experiment.py configs/e1_codrivingllm.yaml --workers 4   # as released (0-shot)
```

**With its memory module (2-shot).** The released code contains the paper's memory
module but switches it off; its own comments say how to switch it on (line 46 builds
the memory, line 159 retrieves the two most similar past decisions). CAVBench
reproduces both functions line by line. The memory is built over 20 interaction
episodes on tuning seeds (100000+), as in the paper's Fig. 7, then frozen:

```bash
python experiments/check_codrivingllm_memory.py            # one test call: key, credit, models (< $0.001)
python experiments/run_experiment.py configs/e1_codrivingllm_memory_build.yaml --workers 4   # ~$1.4
python experiments/check_codrivingllm_memory.py --memory   # must report 20 episodes
python experiments/run_experiment.py configs/e1_codrivingllm_memory.yaml --workers 4         # ~$7
```

Every reply is cached in `results/llm_cache/` and the memory in `results/llm_memory/`,
so an interrupted run resumes without paying twice. The CSV records the calls,
tokens and cost of each episode.

*Earlier LLM variants.* `kind: codrivingllm` (simple and faithful modes, written from
the paper's description, with `experiments/llm_check.py` and
`experiments/build_llm_memory.py`; see [docs/llm_baseline.md](docs/llm_baseline.md))
were used during development. The paper uses the reproduction above.

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

Two versions exist. `kind: mcts` is the **strict** version (Table I plus our
earlier choices). `kind: mcts_calibrated` is the **main baseline**: the same
algorithm and Table I values, with three calibrated values that the paper does not
fix (acceleration and jerk costs on the paper's acceleration scale, w_v2v = 5 and
w_v2h = 10), chosen on tuning seeds only so that the paper's Case 1 and Case 2
results are reproduced. See "Calibration" in docs/mcts_baseline.md.

MCTS needs several seconds per decision in Python, so the paper runs it on the
intersection only, at noise levels none and high, with 100 seeds. Edit
`configs/e1_mcts.yaml` accordingly (`noise_levels: [none, high]`, intersection only)
before running:

```bash
python experiments/run_experiment.py configs/mcts_smoke.yaml --workers 6        # timing check
python experiments/run_experiment.py configs/e1_mcts.yaml --episodes 100 --workers 8
python experiments/watch.py --arbiter mcts --seed 3
```

The CSV gets the extra columns `mcts_iterations`, `mcts_nodes`,
`mcts_pruned_fraction`, `mcts_no_safe_root` and `mcts_max_group_size`.

### Oracle (upper bound)

`kind: oracle` is **not a realistic method**: it has perfect information, to show the
best result reachable in each situation. It reads the true vehicle states (no noise,
no V2X delay or loss) and tests each option on an exact copy of the running
simulation (same human drivers, same random state). For a conflict group of up to 5
CAVs it tries every passing order (K! orders, as the global-optimum reference of Xu
et al., IEEE T-ITS 2020), each with the CAVs waiting for earlier human drivers or
going before them, then follows the best one and re-plans every 2 s or when the
group changes. Larger groups compare one option per CAV going first, plus "everyone
yields", over 4 s. It is an upper bound with perfect information, not a proven
global optimum; its results are the same at every noise level, so it is run at noise
none. Details: [docs/oracle_baseline.md](docs/oracle_baseline.md).

```bash
python experiments/run_experiment.py configs/oracle_smoke.yaml --workers 6
python experiments/run_experiment.py configs/e1_oracle.yaml --workers 8
```

## Method in brief (UCRA)

Every 0.2 s:

1. **Observe** the shared states with noise: p̂ᵢ = pᵢ + εᵢ, εᵢ ~ N(0, σ_p²), v̂ᵢ = vᵢ + νᵢ,
   νᵢ ~ N(0, σ_v²); with V2X delay or loss, each message is moved forward by dead
   reckoning.
2. **Detect conflicts:** routes that meet within 1 m, both vehicles within 50 m of the
   point, and ΔTTCP = |dᵢ/vᵢ − dⱼ/vⱼ| < 8 s. A vehicle counts as past a conflict point
   only once it is 4.5 m + 2σ_p beyond it. Conflicting CAVs form groups.
3. **Candidate maneuvers:** one per conflicting CAV (it goes, the others yield),
   plus the fallback (everyone yields).
4. **What-if rollouts:** N = 100 simulations of each maneuver over T = 4 s (Δt = 0.1 s),
   starting from the shared states plus imagined errors of at least 0.5 m and
   0.3 m/s, growing with message age. Human drivers follow IDM with their own desired
   speed in each rollout and give way with probability 0.5. All maneuvers use the same
   random draws.
5. **Risk and uncertainty:** R̂ₖ = share of dangerous rollouts (two vehicles within
   2 m, three-circle footprints, **or** PET < 1.5 s); Uₖ = std of the closest gap / 5 m.
   Theorem 1 (Hoeffding + union bound) gives N ≥ 98 for accuracy 0.15 with probability
   0.9 over four maneuvers.
6. **Fuzzy decision:** 13 Mamdani rules over risk, uncertainty and waiting time give
   an acceptability Aₖ; > 0.7 accept, 0.4–0.7 proceed cautiously, ≤ 0.4 fall back. An
   option with R̂ ≥ 0.5 never proceeds. See `docs/fuzzy_layer.md`.
7. **Liveness rules:** the previous choice is kept unless clearly worse (margin 0.05);
   a CAV that can no longer stop keeps priority until it has cleared the area; after
   2 s in fallback, the deadlock breaker releases the CAV with the lowest near-contact
   probability (< 0.5 m in at most 10 % of rollouts).
8. **Safety layer:** the shared car-following check, then the actions are sent.

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
  σ_s(age)² = σ_p² + (σ_v·age)² + (a·age²/2)², σ_v(age)² = σ_v² + (a·age)², a = 3 m/s²,
  with σ_p ≥ 0.5 m and σ_v ≥ 0.3 m/s, so stale information counts as less certain
  (`rollout: {age_aware: false}` turns it off; this is the UCRA-noAge ablation).

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
an episode: `python experiments/watch.py --v2x degraded`. The loss randomness uses
its own random generator, so the perception-noise samples stay matched across
experiments, and the ideal channel gives **identical results** to running without it.

## Low-level safety layer

UCRA, FCFS, MCTS, the oracle and no coordination share the same low-level layer, so
that differences come from the arbitration. CoDrivingLLM is run as released, with
its own safety tools and no extra layer. By default (`safety: follow`) the layer
performs **car-following only**: a CAV brakes behind a vehicle travelling the same
way in its lane, in a merging lane or in a lane that diverges from it. **Crossing
conflicts are resolved only by the arbiter**, as in the CoDrivingLLM benchmark. The
following gap is measured from noisy positions, so it includes a margin of 2·√2·σ_p
(4.2 m at high noise). With `params: {safety: full}` in a config, CAVs also brake for
any vehicle physically on their path ahead, including crossing traffic; this
stronger layer can hide arbitration errors.

## Main results (evaluation seeds 1000+)

| Method | Category | Success | Notes |
|---|---|---|---|
| UCRA | proposed | 98.6 % (3 scenarios, 2400 episodes) | 37 % faster than FCFS; decides in 56–98 ms |
| FCFS | classic | 95.6 % | 1 collision but 105 time-outs; 83.7 % at high noise, 88 % in dense traffic |
| MCTS | latest | 96.5 % (intersection) | as safe as UCRA, 6 % faster, but about 8 s per decision |
| CoDrivingLLM | latest | 56 % as released, 65 % with memory (intersection) | 40 % on its own 20-seed protocol, as reported in its paper |
| Oracle | upper bound | 99.0 % | UCRA reaches 78–96 % of its efficiency |

Full tables, paired tests and the robustness, ablation and sensitivity analyses:
`results/final_report.md`.

## Repository layout

```
cavbench/
  envs/          intersection.py, merge.py, base.py (noise, benchmark mode, trajectories, termination)
  noise/         gaussian.py (four noise levels), v2x.py (optional V2X delay + packet loss)
  arbiters/      whatif_core.py (rollouts), fuzzy.py, whatif_fuzzy.py (UCRA),
                 fcfs.py, no_coordination.py, mcts.py (MCTS baseline),
                 oracle.py (upper bound with perfect information),
                 codrivingllm_repro.py (CoDrivingLLM reproduced, with its memory module),
                 codrivingllm_wrapper.py, llm_faithful.py, llm_memory.py (earlier LLM variants),
                 commitment.py
  metrics/       safety.py (success, collisions, PET, speed, Table I metrics)
  conflicts.py   ΔTTCP conflict detection and grouping
  geometry.py    routes as polylines, meeting points
  runner.py      one episode with matched seeds
configs/         experiment configurations (YAML)
experiments/     run_all.py, run_experiment.py, rollout_sensitivity.py,
                 final_report.py (tables and paired tests), make_figures.py (figures),
                 benchmark_table.py (CoDrivingLLM Table I format),
                 check_codrivingllm_memory.py, llm_check.py, build_llm_memory.py,
                 watch.py (graphics)
docs/            benchmark_mode.md, benchmark_metrics.md, codrivingllm_repro.md,
                 fcfs_baseline.md, fuzzy_layer.md, llm_baseline.md, mcts_baseline.md,
                 oracle_baseline.md, paper_notes.md
tests/           pytest suite
third_party/     how to run the original CoDrivingLLM (the code itself is not included)
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
publication), HighwayEnv (Leurent, 2018) and, for the CoDrivingLLM baseline,
Fang et al. (2025).

## Licence

MIT, see `LICENSE`. CoDrivingLLM's code is not included (its repository has no
licence); the reproduction cites its source and reuses its prompts only to reproduce
the published method.
