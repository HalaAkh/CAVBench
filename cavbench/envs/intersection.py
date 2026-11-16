"""Unsignalized four-way intersection with several CAVs and human-driven vehicles.

Built on HighwayEnv's ``MultiAgentIntersectionEnv`` (same road geometry and
human-driver model). The scenario parameters follow the CoDrivingLLM
benchmark (Fang et al., 2025): four CAVs, one per approach, about 65 m along
the 100 m approach lane (about 40-55 m from the centre); background vehicles
following IDM; a "challenger" vehicle going straight; actions SLOWER / IDLE / FASTER over target speeds 0, 4.5 and
9 m/s; a 5 Hz policy frequency and a 15 Hz simulation frequency.

Background (human-driven) vehicles, default ``initial_vehicle_count: 10``:
the set-up makes 10 spawn attempts (9 background vehicles with 60 %
probability each, skipped when closer than 15 m to another vehicle, then the
challenger), runs 3 s of warm-up, places the CAVs and removes every background
vehicle within 20 m of a CAV. About 1-4 human-driven vehicles remain (2.65 on
average on seeds 100000-100019). No vehicles are added during the episode
unless ``spawn_during_episode`` is True (default False).

Optional ``hdv_per_approach: k`` (default None = the set-up above, unchanged):
instead of the background vehicles, exactly k human-driven vehicles per
approach, placed ahead of that approach's first CAV (``hdv_ahead_gap``, 15-20 m
closer to the junction, further ones 15 m apart), with random destinations
and no random skipping or removal. Used for the paper's Case 2 check of the
MCTS baseline (4 AVs + 4 HDVs, configs/mcts_case2_check.yaml).
"""
from __future__ import annotations

import numpy as np
from highway_env import utils
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv
from highway_env.vehicle.behavior import IDMVehicle

from .base import ARBITRATION_DEFAULTS, ArbitrationEnvMixin


class IntersectionIDMVehicle(IDMVehicle):
    """Human-driven vehicle with HighwayEnv's intersection settings.

    HighwayEnv's intersection sets these values on the shared ``IDMVehicle``
    class itself, which changes the human drivers of every scenario created
    later in the same Python process (e.g. a merge episode run by the same
    experiment worker). A subclass keeps them local to this scenario.
    """

    DISTANCE_WANTED = 7.0  # [m]
    COMFORT_ACC_MAX = 6.0  # [m/s2]
    COMFORT_ACC_MIN = -3.0  # [m/s2]


class CAVIntersectionEnv(ArbitrationEnvMixin, MultiAgentIntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(ARBITRATION_DEFAULTS)
        config.update(
            {
                "controlled_vehicles": 4,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "duration": 50,  # [s] as in the benchmark
                "policy_frequency": 5,  # [Hz] benchmark: decisions every 0.2 s
                "simulation_frequency": 15,  # [Hz]
                "destination": None,
                "cav_start_mean": 65.0,  # [m] along the 100 m approach lane
                "cav_start_std": 5.0,
                "cav_wave_gap": 15.0,  # [m] between CAVs on the same approach
                "exit_distance": 15.0,  # [m] benchmark value
                "show_trajectories": False,
                "other_vehicles_type": "cavbench.envs.intersection.IntersectionIDMVehicle",
                # Optional fixed human-driven vehicles (see module docstring):
                "hdv_per_approach": None,  # None = random background vehicles
                "hdv_ahead_gap": [15.0, 20.0],  # [m] uniform range ahead of the CAV
            }
        )
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        self._start_episode()

    def step(self, action):
        out = self.arbitration_step(action)
        if self.config["spawn_during_episode"]:
            self._clear_vehicles()
            self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return out

    def has_arrived(self, vehicle, exit_distance: float | None = None) -> bool:
        d = self.config["exit_distance"] if exit_distance is None else exit_distance
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= d
        )

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        rng = self.np_random
        # Human drivers: IntersectionIDMVehicle (see "other_vehicles_type").

        fixed_hdvs = self.config.get("hdv_per_approach")
        if fixed_hdvs:  # Case 2 set-up: no random background vehicles
            n_vehicles = 0

        # Background (human-driven) vehicles, then 3 s of warm-up.
        for t in range(max(n_vehicles - 1, 0)):
            self._spawn_vehicle(np.linspace(0, 80, max(n_vehicles, 2))[t])
        for _ in range(3 * self.config["simulation_frequency"]):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
        if n_vehicles > 0:  # "challenger" going straight
            self._spawn_vehicle(60, spawn_probability=1.0, go_straight=True,
                                position_deviation=0.1, speed_deviation=0.0)

        # CAVs: one per approach; further waves start exactly `cav_wave_gap`
        # metres behind the previous CAV on the same approach.
        self.controlled_vehicles = []
        v0 = self.config["cav_initial_speed"]
        first_wave: dict[int, float] = {}
        for ego_id in range(self.config["controlled_vehicles"]):
            origin = ego_id % 4
            lane = self.road.network.get_lane((f"o{origin}", f"ir{origin}", 0))
            # Benchmark mode: as in the benchmark's code ("o" + randint(1, 4)),
            # a CAV never exits towards o0.
            exits = range(1, 4) if self.bm("exits") else range(4)
            dest = self.config["destination"] or f"o{rng.choice([d for d in exits if d != origin])}"
            if origin not in first_wave:
                first_wave[origin] = (self.config["cav_start_mean"]
                                      + self.config["cav_start_std"] * rng.normal())
            longitudinal = first_wave[origin] - self.config["cav_wave_gap"] * (ego_id // 4)
            ego = self.action_type.vehicle_class(
                self.road, lane.position(longitudinal, 0.0), speed=v0,
                heading=lane.heading_at(longitudinal),
            )
            ego.plan_route_to(dest)
            ego.speed_index = ego.speed_to_index(v0)
            ego.target_speed = ego.index_to_speed(ego.speed_index)
            self.road.vehicles.append(ego)
            self.controlled_vehicles.append(ego)
            for v in self.road.vehicles.copy():  # prevent collisions at spawn
                if v not in self.controlled_vehicles and np.linalg.norm(v.position - ego.position) < 20:
                    self.road.vehicles.remove(v)

        if fixed_hdvs:
            self._place_fixed_hdvs(int(fixed_hdvs), first_wave)

    def _place_fixed_hdvs(self, per_approach: int, first_wave: dict[int, float]) -> None:
        """Case 2 set-up: ``per_approach`` human-driven vehicles on every
        approach, ahead of its first CAV (closer to the junction), random
        destination, HighwayEnv's initial speed (8 +- 1 m/s) and driving style."""
        rng = self.np_random
        hdv_class = utils.class_from_path(self.config["other_vehicles_type"])
        lo, hi = self.config["hdv_ahead_gap"]
        for origin in range(4):
            lane = self.road.network.get_lane((f"o{origin}", f"ir{origin}", 0))
            base = first_wave.get(origin, self.config["cav_start_mean"])
            gap = rng.uniform(lo, hi)
            for i in range(per_approach):
                longitudinal = min(base + gap + 15.0 * i, lane.length - 1.0)
                vehicle = hdv_class.make_on_lane(
                    self.road, (f"o{origin}", f"ir{origin}", 0),
                    longitudinal=longitudinal, speed=8.0 + rng.normal(),
                )
                dest = rng.choice([d for d in range(4) if d != origin])
                vehicle.plan_route_to(f"o{dest}")
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
