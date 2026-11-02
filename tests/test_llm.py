"""LLM baseline plumbing, without calling the API."""
import json

from cavbench.arbiters import make_arbiter
from cavbench.arbiters import codrivingllm_wrapper as w
from cavbench.envs import make_env
from cavbench.runner import run_episode


def test_mock_reply_is_arrival_order():
    prompt = ("- vehicle 4 is 30.0 m from the conflict point (TTCP 3.0 s); vehicle 7 is 10.0 m "
              "from it (TTCP 1.2 s); dTTCP 1.8 s, serious.")
    assert json.loads(w.mock_reply(prompt))["order"] == [7, 4]


def test_cache_avoids_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "CACHE_DIR", tmp_path)
    calls = []

    class Fake(w.LLMClient):
        def _api(self, prompt):
            calls.append(prompt)
            return '{"order": [1]}'

    c = Fake("gpt-4o-mini")
    assert c("p") == c("p") == '{"order": [1]}'
    assert len(calls) == 1 and c.cache_hits == 1
    c2 = Fake("gpt-4o-mini")  # a new process reads the cache file
    c2("p")
    assert len(calls) == 1


def test_episode_reports_llm_usage():
    env = make_env("intersection", {"noise_level": "high"})
    m = run_episode(env, make_arbiter("codrivingllm", model="mock"), seed=0)
    assert m["llm_calls"] > 0 and m["llm_parse_failures"] == 0 and m["llm_cost_usd"] == 0
    assert m["success"] in (True, False)
