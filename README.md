# CAVBench

**Noisy multi-CAV conflict arbitration with what-if simulation and fuzzy decisions, built on [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).**

CAVBench is the simulation environment of the study *"Uncertainty-aware counterfactual
risk arbitration for connected autonomous vehicles with fuzzy decision-making"*
(Kafel, El Akhrass and Watfa). It provides:

- **Scenarios:** an unsignalized four-way intersection configured after the
  CoDrivingLLM benchmark (Fang et al., 2025), a denser variant, and an on-ramp merge;
- **Perception noise:** Gaussian noise on the positions and speeds that vehicles
  share, at four levels;
- **Arbiters:** the proposed method (UCRA), first-come-first-served, no coordination
  and an LLM arbiter in the style of CoDrivingLLM, all using the same noisy shared
  states and the same low-level safety layer;
- **Metrics:** success, collisions, post-encroachment time (PET < 1.5 s = severe
  conflict), average speed and decision time;
- **Experiment protocol:** YAML configurations, matched random seeds, a parallel and
  resumable runner, and a statistics notebook.

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
(ucra, ucra_nou, ucra_nofallback, fcfs, nocoord, llm), `--noise` (none, low,
medium, high), `--seed`, `--speed` (playback speed) and `--save`. The episode is
exactly the one the experiments run for the same settings.

On screen: blue cars are human drivers; numbered cars are CAVs, coloured by the
arbiter's command (green = go, yellow = cautious, orange = yield, red = crashed);
small rings show where vehicles *appear* to be in the noisy shared data; magenta
lines join CAVs in conflict; the top panel shows the latest decision with its risk,
uncertainty U and fuzzy acceptability A.

## Running the experiments

| Experiment | Config | Purpose |
|---|---|---|
| Smoke test | `configs/smoke.yaml` | 2 episodes per condition, checks that everything runs |
| E1 | `configs/e1_baselines.yaml` | UCRA vs LLM arbiter, FCFS and no coordination (H1) |
| E2 | `configs/e2_uncertainty_ablation.yaml` | with vs without the uncertainty input (H2) |
| E3 | `configs/e3_fallback_ablation.yaml` | with vs without the fallback (H3) |
| E4 | `configs/e4_rollout_sensitivity.yaml` | empirical check of Theorem 1 |
| E5 | `configs/e5_realtime.yaml` | decision time (H4); run with `--workers 1` |

```bash
python experiments/run_experiment.py configs/smoke.yaml
python experiments/run_experiment.py configs/e1_baselines.yaml --workers 8
python experiments/run_experiment.py configs/e5_realtime.yaml --workers 1
python experiments/rollout_sensitivity.py configs/e4_rollout_sensitivity.yaml
jupyter notebook analysis/statistics.ipynb
```

Results are written to `results/<experiment>.csv`, one row per episode. If a run is
interrupted, running the same command again continues where it stopped. The
`--episodes N` option overrides the number of seeds (200 by default, from the power
analysis).

The LLM arbiter calls `gpt-4o-mini` through the OpenAI API: `pip install openai`
and set `OPENAI_API_KEY`. It is listed in E1 but can be removed from the config to
run without an API key. To run the original CoDrivingLLM code instead, see
`third_party/README.md`.

## Method in brief (UCRA)

Every 0.2 s:

1. **Observe** the shared states with noise: ŝᵢ = sᵢ + εᵢ, εᵢ ~ N(0, σ²).
2. **Detect conflicts:** routes that meet, with ΔTTCP = |dᵢ/vᵢ − dⱼ/vⱼ| < 8 s.
3. **Candidate maneuvers:** one per conflicting CAV (it goes, the others yield),
   plus the fallback (everyone yields).
4. **What-if rollouts:** N = 100 simulations of each maneuver over T = 4 s, starting
   from the shared states plus imagined errors; human drivers follow IDM.
5. **Risk and uncertainty:** R̂ₖ = share of rollouts in which two vehicles come within
   2 m (three-circle footprints); Uₖ = std of the closest gap / 5 m.
6. **Fuzzy decision:** 13 Mamdani rules over risk, uncertainty and waiting time give
   an acceptability Aₖ; > 0.7 accept, 0.4–0.7 proceed cautiously, ≤ 0.4 fall back.
   Ties go to the vehicle that arrives first, and "k goes" includes the CAVs queued
   ahead of k in its lane. See `docs/fuzzy_layer.md`.

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
  noise/         gaussian.py (four noise levels)
  arbiters/      whatif_core.py (rollouts), fuzzy.py, whatif_fuzzy.py (UCRA),
                 fcfs.py, no_coordination.py, codrivingllm_wrapper.py, commitment.py
  metrics/       safety.py (success, collisions, PET, speed)
  conflicts.py   ΔTTCP conflict detection and grouping
  geometry.py    routes as polylines, meeting points
  runner.py      one episode with matched seeds
configs/         experiment configurations (YAML)
experiments/     run_experiment.py, rollout_sensitivity.py, watch.py (graphics)
analysis/        statistics.ipynb
docs/            fuzzy_layer.md
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
