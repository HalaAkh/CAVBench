import numpy as np

from cavbench.geometry import Path, meeting_points, route_relation


def straight(a, b, n=101, key=None):
    pts = np.linspace(a, b, n)
    return Path(pts, key=key or (tuple(a), tuple(b)))


def test_arc_length_and_projection():
    p = straight([0, 0], [10, 0])
    assert abs(p.length - 10) < 1e-9
    s, lat = p.project(np.array([4.0, 1.0]))
    assert abs(s - 4.0) < 1e-6 and abs(lat - 1.0) < 1e-6


def test_crossing_paths_meet_once():
    a = straight([-10, 0], [10, 0], key=("a",))
    b = straight([0, -10], [0, 10], key=("b",))
    mps = meeting_points(a, b, 1.0)
    assert len(mps) == 1
    assert abs(mps[0].s_i - 9.0) < 0.5 and abs(mps[0].s_j - 9.0) < 0.5
    assert not mps[0].shared


def test_merging_paths_share_a_lane():
    main = straight([0, 0], [20, 0], key=("main",))
    pts = np.vstack([np.linspace([0, 10], [10, 0], 101), np.linspace([10, 0], [20, 0], 101)[1:]])
    ramp = Path(pts, key=("ramp",))
    meet, shared = route_relation(ramp, main)
    assert meet and shared is not None and shared.length > 5


def test_parallel_lanes_do_not_meet():
    a = straight([0, 0], [20, 0], key=("l1",))
    b = straight([0, 4], [20, 4], key=("l2",))
    assert meeting_points(a, b, 1.0) == []
