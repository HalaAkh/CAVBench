from .base import FASTER, IDLE, SLOWER, Arbiter, FollowingSafety
from .codrivingllm_wrapper import CoDrivingLLMArbiter
from .fcfs import FCFSArbiter
from .fuzzy import A_HIGH, A_LOW, FuzzyAcceptability
from .no_coordination import NoCoordinationArbiter
from .whatif_core import OptionResult, RolloutConfig, evaluate_options
from .whatif_fuzzy import WhatIfFuzzyArbiter

ARBITERS = {
    "whatif_fuzzy": WhatIfFuzzyArbiter,
    "fcfs": FCFSArbiter,
    "no_coordination": NoCoordinationArbiter,
    "codrivingllm": CoDrivingLLMArbiter,
}


def make_arbiter(kind: str, **params) -> Arbiter:
    if kind not in ARBITERS:
        raise ValueError(f"Unknown arbiter {kind!r}; choose from {list(ARBITERS)}")
    return ARBITERS[kind](**params)


__all__ = [
    "ARBITERS", "make_arbiter", "Arbiter", "FollowingSafety", "SLOWER", "IDLE", "FASTER",
    "WhatIfFuzzyArbiter", "FCFSArbiter", "NoCoordinationArbiter",
    "CoDrivingLLMArbiter", "FuzzyAcceptability", "A_LOW", "A_HIGH", "RolloutConfig",
    "OptionResult", "evaluate_options",
]
