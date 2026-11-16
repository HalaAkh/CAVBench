"""Episode metrics, aligned with the CoDrivingLLM benchmark.

* success: every CAV reached its destination with no CAV collision;
* collision: at least one CAV crashed;
* PET (post-encroachment time) for every pair of vehicles, at least one a CAV,
  whose routes cross or merge: the time between the first vehicle leaving
  the conflict area and the second entering it (0 if they overlap). A PET
  below 1.5 s is a severe conflict;
* average speed of the CAVs;
* decision time of the arbiter (reported by the runner);
* the metrics of CoDrivingLLM's Table I (Fang et al., 2025), so that results
  can be put next to that paper's table (see docs/benchmark_metrics.md):
    - collision_free: no CAV collision (the paper's "success": in its data,
      episodes that reach the 50 s limit without a crash count as successes);
    - benchmark PET: for each pair of vehicles (at least one a CAV) whose
      routes cross, the time between the moments the two vehicles pass the
      crossing point (closest approach to it), the paper's "Time Difference";
      stored per episode as sum, count, max and min so that the average,
      maximum and minimum over all pairs of all episodes can be computed;
    - travel velocity: the average speed of each CAV while it is in the scene,
      stored as sum, count, max and min over the episode's CAVs.

PET is computed from trajectories recorded at the 15 Hz simulation rate. The
conflict area is a disk of radius ``zone_radius`` around the point where the
two routes meet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry import meeting_points

TRAJ_COLUMNS = ["t", "vid", "x", "y", "speed", "is_cav", "crashed"]


def trajectory_frame(env) -> pd.DataFrame:
    return pd.DataFrame(env.trajectory, columns=TRAJ_COLUMNS)


def _occupancy(df_v: pd.DataFrame, point: np.ndarray, radius: float):
    d = np.hypot(df_v["x"].to_numpy() - point[0], df_v["y"].to_numpy() - point[1])
    inside = d < radius
    if not inside.any():
        return None
    t = df_v["t"].to_numpy()
    return float(t[inside].min()), float(t[inside].max())


def pet_values(env, df: pd.DataFrame | None = None, zone_radius: float = 3.0,
               meet_distance: float = 1.0) -> list[dict]:
    df = trajectory_frame(env) if df is None else df
    groups = {vid: g.sort_values("t") for vid, g in df.groupby("vid")}
    vids = sorted(groups)
    out = []
    for ia, a in enumerate(vids):
        for b in vids[ia + 1:]:
            if not (env.vehicle_is_cav.get(a) or env.vehicle_is_cav.get(b)):
                continue
            pa, pb = env.vehicle_paths[a], env.vehicle_paths[b]
            for mp in meeting_points(pa, pb, meet_distance):
                if mp.s_i < 1.0 or mp.s_j < 1.0:
                    continue  # one route starts on the shared stretch: car following
                point = pa.position(mp.s_i)
                oa = _occupancy(groups[a], point, zone_radius)
                ob = _occupancy(groups[b], point, zone_radius)
                if oa is None or ob is None:
                    continue
                first, second = (oa, ob) if oa[0] <= ob[0] else (ob, oa)
                pet = max(second[0] - first[1], 0.0)
                out.append({"a": a, "b": b, "pet": pet, "x": point[0], "y": point[1]})
                break
    return out


def benchmark_pet_values(env, df: pd.DataFrame | None = None, zone_radius: float = 3.0,
                         meet_distance: float = 1.0) -> list[float]:
    """CoDrivingLLM-style PET: for each pair of vehicles (at least one a CAV)
    whose routes cross (same pairs and crossing points as :func:`pet_values`),
    |t_a - t_b|, where t is the time at which each vehicle is closest to the
    crossing point (it must come within ``zone_radius`` of it)."""
    df = trajectory_frame(env) if df is None else df
    groups = {vid: g.sort_values("t") for vid, g in df.groupby("vid")}
    vids = sorted(groups)
    out = []
    for ia, a in enumerate(vids):
        for b in vids[ia + 1:]:
            if not (env.vehicle_is_cav.get(a) or env.vehicle_is_cav.get(b)):
                continue
            pa, pb = env.vehicle_paths[a], env.vehicle_paths[b]
            for mp in meeting_points(pa, pb, meet_distance):
                if mp.s_i < 1.0 or mp.s_j < 1.0:
                    continue  # car following, not a crossing
                point = pa.position(mp.s_i)
                times = []
                for g in (groups[a], groups[b]):
                    d = np.hypot(g["x"].to_numpy() - point[0], g["y"].to_numpy() - point[1])
                    k = int(np.argmin(d))
                    times.append(float(g["t"].to_numpy()[k]) if d[k] < zone_radius else None)
                if None not in times:
                    out.append(abs(times[0] - times[1]))
                break
    return out


def travel_velocities(df: pd.DataFrame) -> list[float]:
    """Average speed of each CAV over the frames it is in the scene."""
    cav = df[df["is_cav"]]
    return [float(g["speed"].mean()) for _, g in cav.groupby("vid")]


def episode_metrics(env, pet_threshold: float = 1.5) -> dict:
    df = trajectory_frame(env)
    cav_crash = any(v.crashed for v in env.controlled_vehicles)
    arrived = all(env.has_arrived(v) for v in env.controlled_vehicles)
    pets = pet_values(env, df)
    pet_arr = np.array([p["pet"] for p in pets]) if pets else np.array([])
    cav_rows = df[df["is_cav"]]
    return {
        "success": bool(arrived and not cav_crash),
        "collision": bool(cav_crash),
        "all_arrived": bool(arrived),
        "n_pet_pairs": int(len(pet_arr)),
        "severe_conflicts": int((pet_arr < pet_threshold).sum()),
        "any_severe_conflict": bool((pet_arr < pet_threshold).any()),
        "min_pet": float(pet_arr.min()) if len(pet_arr) else float("nan"),
        "avg_speed": float(cav_rows["speed"].mean()) if len(cav_rows) else float("nan"),
        "episode_time": float(df["t"].max()) if len(df) else 0.0,
        **_benchmark_table_metrics(env, df, cav_crash),
    }


def _benchmark_table_metrics(env, df: pd.DataFrame, cav_crash: bool) -> dict:
    bpet = np.array(benchmark_pet_values(env, df))
    vel = np.array(travel_velocities(df))
    nan = float("nan")
    return {
        "collision_free": bool(not cav_crash),
        "bpet_sum": float(bpet.sum()), "bpet_n": int(len(bpet)),
        "bpet_max": float(bpet.max()) if len(bpet) else nan,
        "bpet_min": float(bpet.min()) if len(bpet) else nan,
        "vel_sum": float(vel.sum()), "vel_n": int(len(vel)),
        "vel_max": float(vel.max()) if len(vel) else nan,
        "vel_min": float(vel.min()) if len(vel) else nan,
    }
