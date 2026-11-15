"""UCRA decision rules: hard risk cap and priority for a CAV in the conflict area."""
import numpy as np

from cavbench.arbiters import FASTER, SLOWER, make_arbiter
from cavbench.arbiters import whatif_fuzzy as wf
from cavbench.arbiters.whatif_core import OptionResult
from cavbench.envs import make_env


def _scene_with_group():
    """A real observation with at least one conflict group of two or more CAVs."""
    env = make_env("intersection", {})
    env.reset(seed=100000)
    arb = make_arbiter("whatif_fuzzy")
    arb.reset(seed=1)
    rng = np.random.default_rng([100000, 2026])
    for _ in range(100):
        obs = env.observe(rng)
        groups = arb.detector.groups(obs)
        if any(len(g.cavs) >= 2 for g in groups):
            return arb, obs, [g for g in groups if len(g.cavs) >= 2][:1]
        acts = arb.act(obs, env.active_cav_vids())
        env.step(tuple(int(acts.get(v, FASTER)) for v in env.cav_vids()))
    raise AssertionError("no conflict group found")


def _fake_options(risks, fb_risk):
    def fake(obs, g, cfg, rng, include_fallback=True):
        out = [OptionResult(v, r, 0.0, np.zeros(1)) for v, r in zip(g.cavs, risks)]
        return out + [OptionResult(None, fb_risk, 0.0, np.zeros(1))]
    return fake


def test_high_risk_option_never_proceeds(monkeypatch):
    """Even if the fuzzy layer would let it go (long wait), an option with
    R_hat >= risk_cap is rejected: every CAV of the group yields."""
    arb, obs, groups = _scene_with_group()
    g = groups[0]
    monkeypatch.setattr(wf, "evaluate_options", _fake_options([0.9] * len(g.cavs), 0.0))
    monkeypatch.setattr(arb.fuzzy, "infer", lambda r, u, w: np.full(len(r), 0.9))
    monkeypatch.setattr(wf, "committed_cavs", lambda *a, **k: [])
    acts = arb.arbitrate(obs, groups)
    assert all(acts[v] == SLOWER for v in g.cavs)
    assert arb.log[-1]["mode"] == "fallback"


def test_cap_not_applied_without_fallback(monkeypatch):
    """The no-fallback ablation keeps its behaviour (the cap is part of the fallback)."""
    arb, obs, groups = _scene_with_group()
    arb.use_fallback = False
    g = groups[0]
    monkeypatch.setattr(wf, "evaluate_options", _fake_options([0.9] * len(g.cavs), 0.0))
    monkeypatch.setattr(arb.fuzzy, "infer", lambda r, u, w: np.full(len(r), 0.9))
    acts = arb.arbitrate(obs, groups)
    assert any(acts[v] == FASTER for v in g.cavs)


def test_priority_stays_with_committed_holder(monkeypatch):
    """A CAV that had priority and can no longer stop keeps it, even when the
    fuzzy layer now prefers another CAV."""
    arb, obs, groups = _scene_with_group()
    g = groups[0]
    holder, other = g.cavs[0], g.cavs[1]
    risks = [0.1 if v == holder else 0.0 for v in g.cavs]
    monkeypatch.setattr(wf, "evaluate_options", _fake_options(risks, 0.0))
    monkeypatch.setattr(arb.fuzzy, "infer",
                        lambda r, u, w: np.array([0.75 if x > 0 else 0.95 for x in r]))
    monkeypatch.setattr(wf, "committed_cavs", lambda *a, **k: [holder])
    monkeypatch.setattr(wf, "platoon_ahead", lambda *a, **k: [])
    arb.holders = {holder}
    acts = arb.arbitrate(obs, groups)
    assert acts[holder] == FASTER and acts[other] == SLOWER
    assert holder in arb.holders
