"""Building blocks of the faithful CoDrivingLLM mode.

CoDrivingLLM (Fang et al., IEEE T-VT 74(8), 2025) runs, at every decision step:

1. a NEGOTIATION call: a central conflict coordinator decides, for each
   conflicting pair, which vehicle passes first (advisory);
2. an ACTION SAFETY CHECK per CAV: unsafe actions are removed from the list
   the LLM may choose from;
3. a DECISION call per CAV: the LLM chooses that CAV's action, given the scene,
   the advisory negotiation result, traffic rules, cautions, the allowed
   actions and (optionally) similar past situations from memory.

Everything here (prompts included) is written from the paper's description;
no code or prompt text is copied from the CoDrivingLLM repository, which has
no licence. See docs/llm_baseline.md for the remaining differences.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .base import ACTION_NAMES, FASTER, IDLE, SLOWER

SEVERITY_RANK = {"none": 0, "slight": 1, "general": 2, "serious": 3}

NEGOTIATION_SYSTEM = (
    "You are the central conflict coordinator of an area where connected automated vehicles "
    "(CAVs) and human-driven vehicles share the road. For every pair of vehicles whose routes "
    "will meet at a conflict point, decide which vehicle should pass the conflict point first. "
    "Conflict severity depends on dTTCP, the difference between the two vehicles' times to "
    "reach the conflict point: below 2 s serious, 2-5 s general, 5-8 s slight, 8 s or more no "
    "danger. Traffic rule: a turning vehicle yields to a vehicle going straight. Human-driven "
    "vehicles cannot be instructed, so do not rely on them yielding. Your result is advice that "
    "each CAV will take into account. Answer only with JSON of the form "
    '{"decisions": [{"first": <vehicle id>, "second": <vehicle id>, "reason": "<short>"}]}, '
    "with one entry per listed pair (an empty list if there are no conflicts)."
)

DECISION_SYSTEM = (
    "You drive one connected automated vehicle (CAV) through an intersection or a merge. Every "
    "0.2 s you choose one longitudinal action: FASTER (raise the target speed one level), IDLE "
    "(keep the target speed) or SLOWER (lower the target speed one level). The route is fixed; "
    "you do not change lanes.\n"
    "Traffic rules: a turning vehicle yields to a vehicle going straight; keep a safe distance "
    "from the vehicle ahead; never enter a conflict point at the same time as another vehicle.\n"
    "Cautions: positions and speeds you receive are measured with some error; human-driven "
    "vehicles may not yield; a serious conflict (dTTCP below 2 s) needs immediate action; do not "
    "stop without reason, because blocking the road also causes delays and conflicts. The "
    "coordinator's result is advice: follow it unless it is clearly unsafe.\n"
    "Choose only among the allowed actions listed. Answer only with JSON of the form "
    '{"action": "FASTER" | "IDLE" | "SLOWER", "reason": "<one sentence>"}.'
)

_ROUTE_CACHE: dict[tuple, str] = {}


def route_kind(path) -> str:
    """'going straight' or 'turning', from the heading change along the route."""
    key = getattr(path, "key", None) or (id(path),)
    if key not in _ROUTE_CACHE:
        a = path.tangent(0.0)
        b = path.tangent(path.length)
        cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
        _ROUTE_CACHE[key] = "turning" if cos < np.cos(np.radians(45)) else "going straight"
    return _ROUTE_CACHE[key]


def _who(obs, v: int) -> str:
    st = obs.states[v]
    return "CAV" if st.is_cav else "human-driven"


# -- negotiation ------------------------------------------------------------
@dataclass
class PairAdvice:
    first: int
    second: int
    severity: str
    reason: str = ""


def describe_pairs(obs, conflicts) -> str:
    if not conflicts:
        return "No conflicts at the moment. Answer with an empty list of decisions."
    lines = ["Conflicting pairs:"]
    for n, c in enumerate(conflicts, 1):
        parts = []
        for v, d, t in ((c.i, c.d_i, c.ttcp_i), (c.j, c.d_j, c.ttcp_j)):
            st = obs.states[v]
            parts.append(f"vehicle {v} ({_who(obs, v)}, {route_kind(st.path)}, speed "
                         f"{st.speed:.1f} m/s, {max(d, 0.0):.1f} m from the conflict point, "
                         f"TTCP {t:.1f} s)")
        lines.append(f"- pair {n}: {parts[0]} and {parts[1]}; dTTCP {c.dttcp:.1f} s, {c.severity}.")
    lines.append("Decide which vehicle of each pair passes first.")
    return "\n".join(lines)


def _json(text: str):
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (ValueError, json.JSONDecodeError):
        return None


def arrival_first(c) -> tuple[int, int]:
    """Fallback order of a pair: earlier TTCP first (ties: lower id)."""
    a, b = (c.i, c.ttcp_i), (c.j, c.ttcp_j)
    first, second = sorted([a, b], key=lambda x: (x[1], x[0]))
    return first[0], second[0]


def parse_negotiation(text: str, conflicts) -> tuple[list[PairAdvice], bool]:
    """Advice for every conflict (arrival order where the reply says nothing).

    Returns (advice, ok); ok is False if the reply was unusable or missed a pair.
    """
    data = _json(text)
    chosen: dict[frozenset, tuple[int, str]] = {}
    ok = isinstance(data, dict) and isinstance(data.get("decisions"), list)
    if ok:
        for d in data["decisions"]:
            try:
                first, second = int(d["first"]), int(d["second"])
            except (KeyError, TypeError, ValueError):
                ok = False
                continue
            chosen[frozenset((first, second))] = (first, str(d.get("reason", ""))[:200])
    advice = []
    for c in conflicts:
        got = chosen.get(frozenset((c.i, c.j)))
        if got is not None and got[0] in (c.i, c.j):
            first = got[0]
            advice.append(PairAdvice(first, c.j if first == c.i else c.i, c.severity, got[1]))
        else:
            ok = False
            f, s = arrival_first(c)
            advice.append(PairAdvice(f, s, c.severity, "fallback: arrival order"))
    return advice, ok


# -- action safety check ----------------------------------------------------
def allowed_actions(follow_cap: int, yield_severities) -> list[int]:
    """Actions the LLM may choose for one CAV, most cautious first.

    * ``follow_cap``: the low-level car-following cap for this CAV
      (:meth:`FollowingSafety.cap`): IDLE removes FASTER; SLOWER removes FASTER
      and IDLE (the vehicle ahead is too close).
    * ``yield_severities``: severities of the conflicts in which the
      negotiation says this CAV must yield. "general" removes FASTER;
      "serious" removes FASTER and IDLE.

    SLOWER is always allowed, so the set is never empty.
    """
    allowed = {SLOWER, IDLE, FASTER}
    sev = set(yield_severities)
    if follow_cap <= IDLE or "general" in sev or "serious" in sev:
        allowed.discard(FASTER)
    if follow_cap <= SLOWER or "serious" in sev:
        allowed.discard(IDLE)
    allowed.add(SLOWER)
    return sorted(allowed)


# -- decision ---------------------------------------------------------------
def describe_cav(obs, vid: int, conflicts, advice: list[PairAdvice], nearby_range: float = 50.0) -> str:
    """The situation of one CAV (also the text stored in / matched against memory)."""
    me = obs.states[vid]
    tgt = f"{me.target_speed:.1f}" if me.target_speed is not None else "unknown"
    levels = ", ".join(f"{s:g}" for s in obs.cav_target_speeds)
    lines = [f"You are vehicle {vid}, a CAV {route_kind(me.path)}. Speed {me.speed:.1f} m/s, "
             f"target speed {tgt} m/s (target speed levels: {levels} m/s)."]
    near = []
    for oid, st in obs.states.items():
        if oid == vid:
            continue
        dist = float(np.hypot(*(st.position - me.position)))
        if dist <= nearby_range:
            near.append((dist, oid, st))
    near.sort(key=lambda x: x[0])
    if near:
        lines.append("Nearby vehicles:")
        for dist, oid, st in near[:6]:
            lines.append(f"- vehicle {oid}: {_who(obs, oid)}, {route_kind(st.path)}, "
                         f"{dist:.1f} m away, speed {st.speed:.1f} m/s")
    else:
        lines.append("No vehicles nearby.")
    mine = [(c, a) for c, a in zip(conflicts, advice) if vid in (c.i, c.j)]
    if mine:
        lines.append("Your conflicts:")
        for c, a in mine:
            other = c.j if c.i == vid else c.i
            d_me, t_me = (c.d_i, c.ttcp_i) if c.i == vid else (c.d_j, c.ttcp_j)
            d_o, t_o = (c.d_j, c.ttcp_j) if c.i == vid else (c.d_i, c.ttcp_i)
            lines.append(f"- with vehicle {other} ({_who(obs, other)}): you are {max(d_me, 0):.1f} m "
                         f"from the conflict point (TTCP {t_me:.1f} s), it is {max(d_o, 0):.1f} m "
                         f"from it (TTCP {t_o:.1f} s); dTTCP {c.dttcp:.1f} s, {c.severity}.")
        lines.append("Coordinator's advice:")
        for c, a in mine:
            other = c.j if c.i == vid else c.i
            lines.append(f"- you pass first before vehicle {other}." if a.first == vid
                         else f"- you should yield to vehicle {other}.")
    else:
        lines.append("You have no conflicts at the moment.")
    return "\n".join(lines)


def decision_prompt(situation: str, allowed: list[int], memories: list[dict]) -> str:
    lines = [situation,
             "Allowed actions (after the safety check): " + ", ".join(ACTION_NAMES[a] for a in allowed) + "."]
    if memories:
        lines.append("Similar past situations and what happened (examples):")
        for n, m in enumerate(memories, 1):
            lines.append(f"Example {n}:\n{m['text']}\nAction taken: {m['action']}\n"
                         f"Feedback: {m['feedback']}")
    lines.append("Choose your action.")
    return "\n".join(lines)


def parse_action(text: str, allowed: list[int]) -> int | None:
    """The chosen action if it is valid and allowed, else None."""
    data = _json(text)
    if not isinstance(data, dict):
        return None
    name = str(data.get("action", "")).strip().upper()
    by_name = {n: a for a, n in ACTION_NAMES.items()}
    a = by_name.get(name)
    return a if a in allowed else None


# -- free stand-in LLM (model "mock") ---------------------------------------
_PAIR_RE = re.compile(r"vehicle (\d+) \([^)]*TTCP ([\d.]+) s\) and vehicle (\d+) \([^)]*TTCP ([\d.]+) s\)")


def mock_negotiation(prompt: str) -> str:
    decisions = []
    for m in _PAIR_RE.finditer(prompt):
        a, ta, b, tb = int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))
        first, second = (a, b) if (ta, a) <= (tb, b) else (b, a)
        decisions.append({"first": first, "second": second, "reason": "mock: arrival order"})
    return json.dumps({"decisions": decisions})


def mock_decision(prompt: str) -> str:
    m = re.search(r"Allowed actions \(after the safety check\): ([A-Z, ]+)\.", prompt)
    allowed = [s.strip() for s in m.group(1).split(",")] if m else ["SLOWER"]
    situation = prompt.split("Allowed actions")[0]  # not the memory examples
    if "you should yield" in situation:
        action = "SLOWER"
    else:
        action = next(a for a in ("FASTER", "IDLE", "SLOWER") if a in allowed)
    return json.dumps({"action": action, "reason": "mock"})
