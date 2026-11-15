from .base import FASTER, IDLE, SLOWER, Arbiter, FollowingSafety
from .codrivingllm_repro import CoDrivingLLMReproArbiter
from .codrivingllm_wrapper import CoDrivingLLMArbiter, CoDrivingLLMFaithfulArbiter
from .fcfs import FCFSArbiter
from .fuzzy import A_HIGH, A_LOW, FuzzyAcceptability
from .no_coordination import NoCoordinationArbiter
from .oracle import OracleArbiter, OracleConfig
from .whatif_core import OptionResult, RolloutConfig, evaluate_options
from .whatif_fuzzy import WhatIfFuzzyArbiter

ARBITERS = {
    "whatif_fuzzy": WhatIfFuzzyArbiter,
    "fcfs": FCFSArbiter,
    "no_coordination": NoCoordinationArbiter,
    "codrivingllm": CoDrivingLLMArbiter,
    "codrivingllm_faithful": CoDrivingLLMFaithfulArbiter,  # mode="faithful"
    "codrivingllm_repro": CoDrivingLLMReproArbiter,  # CoDrivingLLM as released (use this one)
    "oracle": OracleArbiter,  # upper bound with perfect information (docs/oracle_baseline.md)
}


def make_arbiter(kind: str, **params) -> Arbiter:
    if kind not in ARBITERS:
        raise ValueError(f"Unknown arbiter {kind!r}; choose from {list(ARBITERS)}")
    return ARBITERS[kind](**params)


__all__ = [
    "ARBITERS", "make_arbiter", "Arbiter", "FollowingSafety", "SLOWER", "IDLE", "FASTER",
    "WhatIfFuzzyArbiter", "FCFSArbiter", "NoCoordinationArbiter",
    "CoDrivingLLMArbiter", "CoDrivingLLMFaithfulArbiter", "CoDrivingLLMReproArbiter", "FuzzyAcceptability", "A_LOW",
    "A_HIGH", "RolloutConfig",
    "OptionResult", "evaluate_options", "OracleArbiter", "OracleConfig",
]
