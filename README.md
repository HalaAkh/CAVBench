# CAVBench: noisy multi-CAV conflict arbitration (built on HighwayEnv)

Status: work in progress (Week 7). Environments, noise model, conflict
detection, all five arbiters and metrics run end to end; configs, the
experiment CLI, tests and the statistics notebook are still to be added.

## Layout
- `cavbench/envs/` – `intersection` (benchmark settings: 4 CAVs, ~10 IDM HDVs,
  5 Hz policy / 15 Hz sim, speeds 0/4.5/9 m/s) and `merge` (on-ramp joins main lane)
- `cavbench/noise/` – Gaussian noise, levels none / low (0.2 m, 0.1 m/s) /
  medium (0.5, 0.3) / high (1.5, 1.0)
- `cavbench/conflicts.py` – ΔTTCP < 8 s conflict detection (benchmark rule)
- `cavbench/arbiters/` – `whatif_fuzzy` (proposed method), `fcfs`,
  `no_coordination`, `codrivingllm_wrapper` (LLM arbiter written from the paper)
- `cavbench/metrics/` – success, collision, PET (< 1.5 s severe), speed
- `cavbench/runner.py` – one episode with matched seeds

## Quick start
```bash
pip install highway-env scikit-fuzzy pandas scipy
python -c "from cavbench.envs import make_env; from cavbench.arbiters import make_arbiter; \
from cavbench.runner import run_episode; \
print(run_episode(make_env('intersection', {'noise_level':'high'}), make_arbiter('whatif_fuzzy'), seed=0))"
```

## CoDrivingLLM
The original repository has no licence, so none of its code is copied. Add it
as a submodule (`git submodule add https://github.com/FanGShiYuu/CoDrivingLLM
third_party/CoDrivingLLM`); it needs its own environment (gym 0.15, openai 1.2.2).
