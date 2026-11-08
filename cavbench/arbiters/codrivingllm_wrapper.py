"""LLM-based arbitration baseline in the style of CoDrivingLLM.

CoDrivingLLM (Fang et al., 2025) detects conflicts from dTTCP, rates their
severity and lets an LLM coordinator decide the passing order. This module
reproduces that coordinator inside CAVBench, so that it runs in the same
environment, on the same noisy shared states and matched seeds as the other
arbiters (as CoDrivingLLM itself compared all its baselines inside one
simulator). The prompt is written from the paper's description; no code or
prompt text is copied from the CoDrivingLLM repository, which has no licence.

Two modes:

* ``mode="simple"`` (default): one LLM call per conflict group, only when a
  conflict exists; the passing order becomes FASTER (first) / SLOWER (others).
  Zero-shot, no memory.
* ``mode="faithful"``: the CoDrivingLLM design at EVERY decision step: one
  negotiation call (central coordinator, advisory passing order of every
  conflicting pair), an action safety check per CAV, and one decision call per
  CAV; optionally a memory of past decisions retrieved as few-shot examples
  (``memory={"enabled": True, "k": 2, "name": ..., "learn": False}``).
  See :mod:`.llm_faithful`, :mod:`.llm_memory` and docs/llm_baseline.md.

The default LLM is ``gpt-4o-mini`` (the benchmark's model) through the OpenAI
API; set the ``OPENAI_API_KEY`` environment variable, or put the line
``OPENAI_API_KEY=...`` in a file ``.env`` in the repository root (git-ignored;
see ``.env.example``). Never write the key in code or configs. ``model="mock"`` uses a
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
import inspect
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

from .base import ACTION_NAMES, FASTER, SLOWER, Arbiter, FollowingSafety
from .llm_faithful import (
    DECISION_SYSTEM, NEGOTIATION_SYSTEM, SEVERITY_RANK, allowed_actions, decision_prompt,
    describe_cav, describe_pairs, mock_decision, mock_negotiation, parse_action,
    parse_negotiation,
)
from .llm_memory import FEEDBACK_BAD, FEEDBACK_CRASH, FEEDBACK_GOOD, LLMMemory

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
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "llm_cache"
ENV_FILE = ROOT / ".env"


def read_env_file(path: str | Path = ENV_FILE) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines of a .env file (comments and blank lines ignored)."""
    out: dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


def api_key(env_file: str | Path = ENV_FILE) -> str | None:
    """OPENAI_API_KEY from the environment, else from the .env file (then exported
    to the environment so that the OpenAI client finds it)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        key = read_env_file(env_file).get("OPENAI_API_KEY") or None
        if key:
            os.environ["OPENAI_API_KEY"] = key
    return key


def require_api_key() -> str:
    key = api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put the line OPENAI_API_KEY=sk-... in a file named .env "
            "in the CAVBench folder (copy .env.example), or set the environment variable "
            "(Windows: setx OPENAI_API_KEY \"sk-...\", then reopen the terminal). "
            "Use model 'mock' to test without the API.")
    return key


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

    def _key(self, prompt: str, system: str | None = None) -> str:
        # With the default system prompt the key is exactly the original one,
        # so the simple mode's cached replies stay valid.
        system = SYSTEM_PROMPT if system is None else system
        raw = f"{self.model}|{self.temperature}|{system}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _api(self, prompt: str, system: str | None = None) -> str:
        if self.model == "mock":
            return mock_reply(prompt, system)
        if self._client is None:
            require_api_key()
            from openai import OpenAI  # optional dependency: pip install openai

            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                r = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role": "system", "content": system or SYSTEM_PROMPT},
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

    def __call__(self, prompt: str, system: str | None = None) -> str:
        """Reply to ``prompt``; ``system`` replaces the default SYSTEM_PROMPT."""
        self.calls += 1
        key = self._key(prompt, system)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        reply = self._api(prompt) if system is None else self._api(prompt, system)
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


def mock_reply(prompt: str, system: str | None = None) -> str:
    """Stand-in LLM for testing without an API key: arrival (TTCP) order."""
    if system == NEGOTIATION_SYSTEM:
        return mock_negotiation(prompt)
    if system == DECISION_SYSTEM:
        return mock_decision(prompt)
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


MEMORY_DEFAULTS = {
    "enabled": False,
    "k": 2,  # the paper's 90 % intersection result is the 2-shot version
    "name": None,  # file results/llm_memory/<name>.jsonl
    "path": None,  # or an explicit file path
    "learn": False,  # True only when BUILDING the memory (training seeds)
    "embedding": "auto",  # "local" (free), "openai", or auto: local for the mock model
}


class CoDrivingLLMArbiter(Arbiter):
    name = "codrivingllm"

    def __init__(self, model: str = "gpt-4o-mini", llm_fn: Callable[[str], str] | None = None,
                 mode: str = "simple", memory: dict | None = None, **kw) -> None:
        super().__init__(**kw)
        if mode not in ("simple", "faithful"):
            raise ValueError(f"mode must be 'simple' or 'faithful', got {mode!r}")
        self.model = model
        self.mode = mode
        self._llm_fn = llm_fn
        self.parse_failures = 0
        self.llm_failures = 0
        # -- faithful mode only --
        self.n_negotiation = self.n_decision = 0
        self._pending: dict[int, dict] = {}  # last decisions awaiting feedback (learn mode)
        self._follow = self.safety or FollowingSafety()  # cap used by the action safety check
        self.memory_cfg = {**MEMORY_DEFAULTS, **(memory or {})}
        unknown = set(self.memory_cfg) - set(MEMORY_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown memory options {sorted(unknown)}")
        self.memory: LLMMemory | None = None
        self.memory_path: Path | None = None
        if self.memory_cfg["enabled"]:
            if mode != "faithful":
                raise ValueError("the LLM memory needs mode='faithful'")
            self.memory = self._make_memory()

    def _make_memory(self) -> LLMMemory:
        cfg = self.memory_cfg
        emb = cfg["embedding"]
        if emb == "auto":
            emb = "local" if self.model == "mock" else "openai"
        if cfg["path"]:
            path = Path(cfg["path"])
        elif cfg["name"]:
            path = LLMMemory.path_for(cfg["name"])
        else:
            path = None
        self.memory_path = path
        if path is not None and path.exists():
            return LLMMemory.load(path, emb, cfg["k"])
        if cfg["learn"]:
            return LLMMemory(emb, cfg["k"])  # start empty, filled during training episodes
        hint = f" --name {cfg['name']}" if cfg["name"] else ""
        raise FileNotFoundError(
            f"LLM memory file {path} not found. Build it first on the training seeds:\n"
            f"    python experiments/build_llm_memory.py --model {self.model}{hint}")

    @property
    def learning(self) -> bool:
        return self.memory is not None and bool(self.memory_cfg["learn"])

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

    # -- faithful mode ------------------------------------------------------
    def _ask(self, prompt: str, system: str) -> str:
        """One LLM call with a given system prompt (faithful mode)."""
        fn = self.llm
        try:
            if isinstance(fn, LLMClient) or _accepts_system(fn):
                return fn(prompt, system=system)
            return fn(prompt)  # a plain ``prompt -> str`` callable
        except RuntimeError:
            raise  # missing key / wrong model: stop with a clear message
        except Exception as exc:  # the API still failing after the retries
            self.llm_failures += 1
            return f"<LLM error: {type(exc).__name__}>"

    def act(self, obs, all_cavs):
        if self.mode != "faithful":
            return super().act(obs, all_cavs)
        t0 = time.perf_counter()
        conflicts = [c for g in self.detector.groups(obs) for c in g.conflicts]
        if self.learning:
            self._give_feedback(conflicts)

        # 1) negotiation: one call, every step (even without conflicts)
        self.n_negotiation += 1
        reply = self._ask(describe_pairs(obs, conflicts), NEGOTIATION_SYSTEM)
        advice, ok = parse_negotiation(reply, conflicts)
        if not ok:
            self.parse_failures += 1
        self.log.append({
            "time": obs.time, "mode": "negotiation",
            "group": sorted({v for c in conflicts for v in (c.i, c.j) if obs.states[v].is_cav}),
            "decisions": [(a.first, a.second, a.severity) for a in advice], "reply": reply[:300],
        })

        # 2) + 3) action safety check and one decision call per CAV
        actions = {vid: FASTER for vid in all_cavs}
        cavs = [v for v in all_cavs if v in obs.states and not obs.states[v].arrived]
        situations = {v: describe_cav(obs, v, conflicts, advice) for v in cavs}
        use_memory = self.memory is not None and len(self.memory) > 0 and self.memory.k > 0
        vectors = {}
        if use_memory and cavs:  # one (batched) embedding request per step
            vectors = dict(zip(cavs, self.memory.embed([situations[v] for v in cavs])))
        for v in cavs:
            yield_sev = [a.severity for a in advice if a.second == v]
            allowed = allowed_actions(self._follow.cap(obs, v), yield_sev)
            memories = self.memory.retrieve(situations[v], vector=vectors[v]) if use_memory else []
            self.n_decision += 1
            reply = self._ask(decision_prompt(situations[v], allowed, memories), DECISION_SYSTEM)
            a = parse_action(reply, allowed)
            if a is None:  # invalid or not allowed: the most cautious allowed action
                self.parse_failures += 1
                a = allowed[0]
            actions[v] = a
            self.log.append({"time": obs.time, "mode": "decision", "vid": v,
                             "action": ACTION_NAMES[a], "allowed": [ACTION_NAMES[x] for x in allowed],
                             "memories": len(memories), "reply": reply[:300]})
            if self.learning and any(v in (c.i, c.j) for c in conflicts):
                self._pending[v] = {
                    "text": situations[v], "action": ACTION_NAMES[a],
                    "ranks": {(c.j if c.i == v else c.i): SEVERITY_RANK[c.severity]
                              for c in conflicts if v in (c.i, c.j)},
                }
        self.decision_times.append((time.perf_counter() - t0) * 1000.0)
        if self.safety is not None:
            actions = self.safety.apply(obs, actions)
        return actions

    def _give_feedback(self, conflicts) -> None:
        """Memory augment: judge the previous step's decisions with the new state."""
        now: dict[int, dict] = {}
        for c in conflicts:
            now.setdefault(c.i, {})[c.j] = c
            now.setdefault(c.j, {})[c.i] = c
        for v, p in self._pending.items():
            mine = now.get(v, {})
            worse = any(o in mine and SEVERITY_RANK[mine[o].severity] > r for o, r in p["ranks"].items())
            # near miss (proxy for PET < 1.5 s): both reach the point within 1.5 s of each other, soon
            near = any(c.dttcp < 1.5 and min(c.ttcp_i, c.ttcp_j) < 1.5 for c in mine.values())
            self.memory.add(p["text"], p["action"], FEEDBACK_BAD if (worse or near) else FEEDBACK_GOOD)
        self._pending = {}

    def end_episode(self, env) -> None:
        """Feedback for the last decisions (a collision ends the episode)."""
        if self.learning and self._pending:
            crashed = {env.vid(v) for v in env.controlled_vehicles if v.crashed}
            for v, p in self._pending.items():
                self.memory.add(p["text"], p["action"], FEEDBACK_CRASH if v in crashed else FEEDBACK_GOOD)
        self._pending = {}

    # -- bookkeeping ----------------------------------------------------------
    def reset(self, seed=None) -> None:
        super().reset(seed)
        self.parse_failures = self.llm_failures = 0
        c = self._llm_fn
        self._usage0 = ((c.calls, c.cache_hits, c.tokens_in, c.tokens_out, c.cost_usd)
                        if isinstance(c, LLMClient) else None)
        self.n_negotiation = self.n_decision = 0
        self._pending = {}
        mem = self.memory
        self._mem0 = (mem.embed_calls, mem.embed_cost_usd, mem.hits) if mem is not None else (0, 0.0, 0)

    def extra_metrics(self) -> dict:
        m = {"llm_parse_failures": self.parse_failures, "llm_failures": self.llm_failures}
        c = self._llm_fn
        if isinstance(c, LLMClient):
            base = getattr(self, "_usage0", None) or (0, 0, 0, 0, 0.0)
            now = (c.calls, c.cache_hits, c.tokens_in, c.tokens_out, c.cost_usd)
            d = [b - a for a, b in zip(base, now)]
            m.update({"llm_calls": d[0], "llm_cache_hits": d[1], "llm_tokens_in": d[2],
                      "llm_tokens_out": d[3], "llm_cost_usd": round(d[4], 6)})
        if self.mode == "faithful":
            mem = self.memory
            m0 = getattr(self, "_mem0", (0, 0.0, 0))
            emb_cost = mem.embed_cost_usd - m0[1] if mem is not None else 0.0
            m.update({
                "llm_calls_negotiation": self.n_negotiation,
                "llm_calls_decision": self.n_decision,
                "llm_calls_embedding": mem.embed_calls - m0[0] if mem is not None else 0,
                "memory_items": len(mem) if mem is not None else 0,
                "memory_hits": mem.hits - m0[2] if mem is not None else 0,
            })
            m["llm_cost_usd"] = round(m.get("llm_cost_usd", 0.0) + emb_cost, 6)  # incl. embeddings
        return m


class CoDrivingLLMFaithfulArbiter(CoDrivingLLMArbiter):
    """Shortcut: ``kind: codrivingllm_faithful`` = ``codrivingllm`` with mode="faithful"."""

    name = "codrivingllm_faithful"

    def __init__(self, **kw) -> None:
        kw.setdefault("mode", "faithful")
        super().__init__(**kw)


def _accepts_system(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "system" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())
