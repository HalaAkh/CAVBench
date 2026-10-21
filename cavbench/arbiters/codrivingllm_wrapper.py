"""LLM-based arbitration baseline in the style of CoDrivingLLM.

Two ways of comparing with CoDrivingLLM (Fang et al., 2025) are supported:

1. **This module (in-loop, same environment).** An LLM decides the passing
   order of each conflict group from the *same noisy shared states* that the
   other arbiters receive. The prompt is written from the paper's
   description (conflicts ranked by dTTCP severity, passing order decided by
   the coordinator under traffic rules). No code or prompt text is copied
   from the CoDrivingLLM repository, which has no licence.

2. **The original code (separate environment).** ``third_party/`` holds the
   CoDrivingLLM repository as a git submodule and a runner that injects the
   same Gaussian noise into the states its LLM modules read. See
   ``third_party/README.md``. Use this for the headline comparison.

The default LLM is ``gpt-4o-mini`` (the benchmark's model) through the
OpenAI API (``OPENAI_API_KEY``). Any callable ``prompt -> str`` can be passed
as ``llm_fn`` instead, e.g. a local model or a mock for testing.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .base import FASTER, SLOWER, Arbiter

SYSTEM_PROMPT = (
    "You coordinate connected automated vehicles at a conflict zone. "
    "Given vehicles whose planned routes meet, decide the order in which they pass the "
    "conflict point so that no two vehicles occupy it at the same time and traffic keeps "
    "flowing. Conflicts are rated by the difference in their times to reach the conflict "
    "point (dTTCP): <= 2 s serious, 2-5 s general, 5-8 s slight. Human-driven vehicles "
    "cannot be instructed; automated vehicles can. Answer only with JSON of the form "
    '{"order": [vehicle ids, first to pass first], "reason": "<one sentence>"}.'
)


def describe_group(obs, group) -> str:
    lines = ["Vehicles:"]
    ids = list(group.cavs) + group.others
    for v in ids:
        st = obs.states[v]
        kind = "automated (CAV)" if st.is_cav else "human-driven"
        lines.append(f"- vehicle {v}: {kind}, speed {st.speed:.1f} m/s")
    lines.append("Conflicts:")
    for c in group.conflicts:
        lines.append(
            f"- vehicle {c.i} is {c.d_i:.1f} m from the conflict point (TTCP {c.ttcp_i:.1f} s); "
            f"vehicle {c.j} is {c.d_j:.1f} m from it (TTCP {c.ttcp_j:.1f} s); "
            f"dTTCP {c.dttcp:.1f} s, {c.severity}."
        )
    lines.append("Give the passing order of all listed vehicles.")
    return "\n".join(lines)


def openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> Callable[[str], str]:
    from openai import OpenAI  # optional dependency

    client = OpenAI()

    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""

    return call


def parse_order(text: str, valid: list[int]) -> list[int] | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        order = [int(v) for v in json.loads(m.group(0)).get("order", [])]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    order = [v for v in order if v in valid]
    return order + [v for v in valid if v not in order] if order else None


class CoDrivingLLMArbiter(Arbiter):
    name = "codrivingllm"

    def __init__(self, model: str = "gpt-4o-mini", llm_fn: Callable[[str], str] | None = None,
                 **kw) -> None:
        super().__init__(**kw)
        self.model = model
        self._llm_fn = llm_fn
        self.parse_failures = 0

    @property
    def llm(self) -> Callable[[str], str]:
        if self._llm_fn is None:
            self._llm_fn = openai_llm(self.model)
        return self._llm_fn

    def arbitrate(self, obs, groups):
        actions = {}
        for g in groups:
            ids = list(g.cavs) + g.others
            prompt = describe_group(obs, g)
            reply = self.llm(prompt)
            order = parse_order(reply, ids)
            if order is None:  # unusable reply: fall back to arrival order
                self.parse_failures += 1
                ttcp = {}
                for c in g.conflicts:
                    ttcp[c.i] = min(ttcp.get(c.i, 1e9), c.ttcp_i)
                    ttcp[c.j] = min(ttcp.get(c.j, 1e9), c.ttcp_j)
                order = sorted(ids, key=lambda v: ttcp.get(v, 1e9))
            first = order[0]
            for v in g.cavs:
                actions[v] = FASTER if v == first else SLOWER
            self.log.append({"time": obs.time, "group": list(g.cavs), "order": order,
                             "reply": reply[:500]})
        return actions
