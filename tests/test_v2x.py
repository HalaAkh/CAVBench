import numpy as np
import pytest

from cavbench.arbiters import make_arbiter
from cavbench.envs import make_env
from cavbench.noise import V2X_LEVELS, V2XChannel
from cavbench.runner import run_episode


def _feed(ch, times, vid=0):
    """Record a vehicle whose position x equals the send time."""
    for t in times:
        ch.record(t, vid, [t, 0.0], 10.0)


def test_default_channel_is_ideal():
    assert V2XChannel().is_ideal
    assert V2XChannel.from_level("ideal").is_ideal
    assert not V2XChannel.from_level("typical").is_ideal
    assert V2X_LEVELS["degraded"] == (0.3, 0.3)


def test_invalid_values_raise():
    for kwargs in ({"delay": -0.1}, {"loss_prob": -0.1}, {"loss_prob": 1.0}):
        with pytest.raises(ValueError):
            V2XChannel(**kwargs)
    with pytest.raises(ValueError):
        V2XChannel.from_level("perfect")


def test_delay_delivers_old_state():
    ch = V2XChannel(delay=0.3)
    ch.reset()
    times = [k / 10 for k in range(11)]  # 0.0 .. 1.0 s
    _feed(ch, times)
    pos, spd, ages = ch.receive(1.0, [0], np.random.default_rng(0))
    assert pos[0, 0] == pytest.approx(0.7)
    assert spd[0] == 10.0
    assert ages[0] == pytest.approx(0.3)


def test_loss_makes_messages_stale():
    ch = V2XChannel(loss_prob=0.5)
    ch.reset()
    rng = np.random.default_rng(0)
    stale, max_age = 0, 0.0
    n = 2000
    for k in range(n):
        t = k / 15
        _feed(ch, [t])
        _, _, ages = ch.receive(t, [0], rng)
        stale += ages[0] > 1e-9
        max_age = max(max_age, ages[0])
    assert 0.4 < stale / n < 0.6
    assert max_age > 2 / 15  # consecutive losses make the age grow


def test_receive_does_not_consume_noise_samples():
    rng_a = np.random.default_rng([2, 2026])
    rng_b = np.random.default_rng([2, 2026])
    ch = V2XChannel(delay=0.1, loss_prob=0.3)
    ch.reset()
    _feed(ch, [0.0, 0.1, 0.2])
    for _ in range(5):
        ch.receive(0.2, [0], rng_a)
    assert np.array_equal(rng_a.normal(size=10), rng_b.normal(size=10))


def test_ideal_level_gives_identical_results():
    base = {"noise_level": "high", "duration": 15}
    a = run_episode(make_env("intersection", base), make_arbiter("fcfs"), seed=2)
    b = run_episode(make_env("intersection", {**base, "v2x_level": "ideal"}),
                    make_arbiter("fcfs"), seed=2)
    # Wall-clock timings (decision_ms_*) differ from run to run; everything else must match.
    keys = [k for k in a if not k.startswith("decision_ms")]
    assert set(a) == set(b)
    for k in keys:
        assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k  # (nan == nan)


def test_observation_ages_at_least_delay():
    env = make_env("intersection", {"v2x_delay": 0.2, "v2x_loss": 0.2, "duration": 5})
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        obs = env.observe(rng)
        ages = obs.extra["v2x_age"]
        assert set(ages) == set(obs.states)
        if env.sim_time >= 0.2:
            assert all(a >= 0.2 - 1e-9 for a in ages.values())
        _, _, terminated, truncated, _ = env.step(tuple(1 for _ in env.cav_vids()))
        if terminated or truncated:
            break


@pytest.mark.parametrize("level", ["typical", "degraded"])
def test_episode_runs_with_v2x(level):
    env = make_env("intersection", {"noise_level": "high", "v2x_level": level, "duration": 15})
    m = run_episode(env, make_arbiter("fcfs"), seed=0)
    assert "success" in m and m["episode_time"] <= 15.5
