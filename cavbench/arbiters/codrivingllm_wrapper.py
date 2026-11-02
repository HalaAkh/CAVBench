"""LLM-based arbitration baseline in the style of CoDrivingLLM.

CoDrivingLLM (Fang et al., 2025) detects conflicts from dTTCP, rates their
severity and lets an LLM coordinator decide the passing order. This module
reproduces that coordinator inside CAVBench, so that it runs in the same
environment, on the same noisy shared states and matched seeds as the other
arbiters (as CoDrivingLLM itself compared all its baselines inside one
simulator). The prompt is written from the paper's description; no code or
prompt text is copied from the CoDrivingLLM repository, which has no licence.
Its memory (retrieval of past decisions) is not reproduced: this is the
zero-shot coordinator.

The default LLM is ``gpt-4o-mini`` (the benchmark's model) through the OpenAI
API; set the ``OPENAI_API_KEY`` environment variable. ``model="mock"`` uses a
stand-in that answers with the arrival order and needs no key (to test the
pipeline). Any callable ``prompt -> str`` can be passed as ``llm_fn``.

Replies are cached on disk (``results/llm_cache/<model>.jsonl``), keyed by the
prompt: re-running or resuming an experiment does not pay twice, and the
cache documents every answer the LLM gave. API errors are retried with
back-off; if the LLM still fails or its answer cannot be parsed, the group
falls back to arrival order and the failure is counted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
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


# USD per million tokens (input, output); used only for the cost report.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.0), "mock": (0.0, 0.0)}
CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "llm_cache"


class LLMClient:
    """Calls the LLM, with an on-disk cache, retries and token accounting."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 cache: bool = True, max_retries: int = 6, timeout: float = 30.0) -> None:
        self.model, self.temperature = model, temperature
        self.max_retries, self.timeout = max_retries, timeout
        self.calls = self.cache_hits = self.tokens_in = self.tokens_out = 0
        self.api_seconds = 0.0
        self.consecutive_failures = 0
        self._client = None
        self._cache: dict[str, str] = {}
        self._cache_path = CACHE_DIR / f"{model}.jsonl" if cache and model != "mock" else None
        self._lock = threading.Lock()
        if self._cache_path is not None and self._cache_path.exists():
            with self._cache_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        self._cache[e["key"]] = e["reply"]
                    except (ValueError, KeyError):
                        continue

    def _key(self, prompt: str) -> str:
        raw = f"{self.model}|{self.temperature}|{SYSTEM_PROMPT}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _api(self, prompt: str) -> str:
        if self.model == "mock":
            return mock_reply(prompt)
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Create a key at platform.openai.com and set "
                    "it (Windows: setx OPENAI_API_KEY \"sk-...\", then reopen the terminal), "
                    "or use model 'mock' to test without the API.")
            from openai import OpenAI  # optional dependency: pip install openai

            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                r = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                )
                self.api_seconds += time.perf_counter() - t0
                self.consecutive_failures = 0
                if r.usage is not None:
                    self.tokens_in += r.usage.prompt_tokens
                    self.tokens_out += r.usage.completion_tokens
                return r.choices[0].message.content or ""
            except Exception as exc:  # rate limit, timeout, connection error
                name = type(exc).__name__
                if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                            "BadRequestError"):  # wrong key / model: stop, do not retry
                    raise RuntimeError(f"OpenAI API {name}: {exc}") from exc
                if attempt == self.max_retries:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= 5:
                        raise RuntimeError(f"OpenAI API unreachable ({name}) for 5 calls in a "
                                           f"row; stopping. Check the connection and re-run "
                                           f"(finished episodes are kept).") from exc
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        return ""

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        key = self._key(prompt)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        reply = self._api(prompt)
        self._cache[key] = reply
        if self._cache_path is not None:
            with self._lock:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "prompt": prompt, "reply": reply}) + "\n")
        return reply

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (self.tokens_in * pin + self.tokens_out * pout) / 1e6


def mock_reply(prompt: str) -> str:
    """Stand-in LLM for testing without an API key: arrival (TTCP) order."""
    ttcp: dict[int, float] = {}
    for m in re.finditer(r"vehicle (\d+) is [\d.]+ m from (?:the conflict point|it) "
                         r"\(TTCP ([\d.]+) s\)", prompt):
        v, t = int(m.group(1)), float(m.group(2))
        ttcp[v] = min(ttcp.get(v, 1e9), t)
    order = sorted(ttcp, key=ttcp.get)
    return json.dumps({"order": order, "reason": "mock: arrival order"})


def openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> Callable[[str], str]:
    return LLMClient(model, temperature)


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
        self.llm_failures = 0

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
            try:
                reply = self.llm(prompt)
            except RuntimeError:
                raise  # missing key: stop the experiment with a clear message
            except Exception as exc:  # the API still failing after the retries
                self.llm_failures += 1
                reply = f"<LLM error: {type(exc).__name__}>"
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

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self.parse_failures = self.llm_failures = 0
        c = self._llm_fn
        self._usage0 = ((c.calls, c.cache_hits, c.tokens_in, c.tokens_out, c.cost_usd)
                        if isinstance(c, LLMClient) else None)

    def extra_metrics(self) -> dict:
        m = {"llm_parse_failures": self.parse_failures, "llm_failures": self.llm_failures}
        c = self._llm_fn
        if isinstance(c, LLMClient):
            base = getattr(self, "_usage0", None) or (0, 0, 0, 0, 0.0)
            now = (c.calls, c.cache_hits, c.tokens_in, c.tokens_out, c.cost_usd)
            d = [b - a for a, b in zip(base, now)]
            m.update({"llm_calls": d[0], "llm_cache_hits": d[1], "llm_tokens_in": d[2],
                      "llm_tokens_out": d[3], "llm_cost_usd": round(d[4], 6)})
        return m
