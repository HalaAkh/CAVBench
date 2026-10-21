"""Episode metrics, aligned with the CoDrivingLLM benchmark.

* success: every CAV reached its destination with no CAV collision;
* collision: at least one CAV crashed;
* PET (post-encroachment time) for every pair of vehicles, at least one a CAV,
  whose routes cross or merge: the time between the first vehicle leaving
  the conflict area and the second entering it (0 if they overlap). A PET
  below 1.5 s is a severe conflict;
* average speed of the CAVs;
* decision time of the arbiter (reported by the runner).

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
    }
