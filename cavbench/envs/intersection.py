"""Unsignalized four-way intersection with several CAVs and human-driven vehicles.

Built on HighwayEnv's ``MultiAgentIntersectionEnv`` (same road geometry and
human-driver model). The scenario parameters follow the CoDrivingLLM
benchmark (Fang et al., 2025): four CAVs, one per approach, about 65 m from
the centre; ten background vehicles following IDM; a "challenger" vehicle
going straight; actions SLOWER / IDLE / FASTER over target speeds 0, 4.5 and
9 m/s; a 5 Hz policy frequency and a 15 Hz simulation frequency.
"""
from __future__ import annotations

import numpy as np
from highway_env import utils
from highway_env.envs.intersection_env import MultiAgentIntersectionEnv

from .base import ARBITRATION_DEFAULTS, ArbitrationEnvMixin


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
                "exit_distance": 15.0,  # [m] benchmark value
                "show_trajectories": False,
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
        hdv = utils.class_from_path(self.config["other_vehicles_type"])
        hdv.DISTANCE_WANTED = 7  # HighwayEnv intersection settings
        hdv.COMFORT_ACC_MAX = 6
        hdv.COMFORT_ACC_MIN = -3

        # Background (human-driven) vehicles, then 3 s of warm-up.
        for t in range(max(n_vehicles - 1, 0)):
            self._spawn_vehicle(np.linspace(0, 80, max(n_vehicles, 2))[t])
        for _ in range(3 * self.config["simulation_frequency"]):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
        if n_vehicles > 0:  # "challenger" going straight
            self._spawn_vehicle(60, spawn_probability=1.0, go_straight=True,
                                position_deviation=0.1, speed_deviation=0.0)

        # CAVs: one per approach (a second wave starts 15 m further back).
        self.controlled_vehicles = []
        v0 = self.config["cav_initial_speed"]
        for ego_id in range(self.config["controlled_vehicles"]):
            origin = ego_id % 4
            lane = self.road.network.get_lane((f"o{origin}", f"ir{origin}", 0))
            dest = self.config["destination"] or f"o{rng.choice([d for d in range(4) if d != origin])}"
            longitudinal = (self.config["cav_start_mean"]
                            + self.config["cav_start_std"] * rng.normal()
                            - 15.0 * (ego_id // 4))
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
