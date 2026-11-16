"""The CoDrivingLLM Table I metrics."""
import numpy as np

from cavbench.arbiters import make_arbiter
from cavbench.envs import make_env
from cavbench.runner import run_episode


def test_table_metrics_present_and_consistent():
    m = run_episode(make_env("intersection", {}), make_arbiter("fcfs"), seed=1000)
    assert m["collision_free"] == (not m["collision"])
    assert m["vel_n"] == 4  # one average speed per CAV
    assert m["vel_min"] <= m["vel_sum"] / m["vel_n"] <= m["vel_max"]
    if m["bpet_n"]:
        assert 0 <= m["bpet_min"] <= m["bpet_sum"] / m["bpet_n"] <= m["bpet_max"]


def test_benchmark_table_script(tmp_path):
    import pandas as pd
    from experiments.benchmark_table import table
    rows = []
    for arb, kind in [("UCRA", "whatif_fuzzy"), ("FCFS", "fcfs")]:
        m = run_episode(make_env("intersection", {}), make_arbiter(kind), seed=1001)
        rows.append({"scenario": "intersection", "noise": "none", "arbiter": arb, **m})
    t = table(pd.DataFrame(rows))
    assert list(t["Method"]) == ["UCRA", "FCFS"]
    assert np.isfinite(t["Velocity avg"]).all()
