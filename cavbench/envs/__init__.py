from gymnasium.envs.registration import register

from .base import CAVehicle, ArbitrationEnvMixin
from .intersection import CAVIntersectionEnv
from .merge import CAVMergeEnv

SCENARIOS = {"intersection": CAVIntersectionEnv, "merge": CAVMergeEnv}


def make_env(scenario: str, config: dict | None = None, render_mode: str | None = None):
    """Create a CAVBench scenario with configuration overrides.

    ``render_mode`` is HighwayEnv's: None (no graphics, used by the experiments),
    "rgb_array" (images) or "human" (a window). See ``experiments/watch.py``.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
    env = SCENARIOS[scenario](config=config or {}, render_mode=render_mode)
    return env


register(id="cavbench-intersection-v0", entry_point="cavbench.envs:CAVIntersectionEnv")
register(id="cavbench-merge-v0", entry_point="cavbench.envs:CAVMergeEnv")

__all__ = ["CAVIntersectionEnv", "CAVMergeEnv", "CAVehicle", "ArbitrationEnvMixin",
           "SCENARIOS", "make_env"]
