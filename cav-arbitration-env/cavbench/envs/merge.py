"""On-ramp merge: a ramp joins a single main lane at a merge point.

HighwayEnv's own merge scenario requires a lane change, which is a lateral
action. To keep arbitration purely longitudinal (who passes the merge point
first), the ramp here joins the main lane directly: vehicles from both roads
share the lane after the merge point. The road network is

    main:  a ----(200 m)---- b ----(150 m)---- c
    ramp:  j ----(~81 m)----/

The main road has priority over the ramp for human-driven vehicles, which
follow IDM and HighwayEnv's right-of-way rules. CAVs (two on each road by
default) are coordinated only by the arbiter under test.
"""
from __future__ import annotations

import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork

from .base import ARBITRATION_DEFAULTS, ArbitrationEnvMixin

MERGE_X = 200.0
EXIT_LENGTH = 150.0
RAMP_START = np.array([MERGE_X - 80.0, 15.0])


class CAVMergeEnv(ArbitrationEnvMixin, AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(ARBITRATION_DEFAULTS)
        config.update(
            {
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                        "target_speeds": [0, 6, 12],
                    },
                },
                "controlled_vehicles": 4,
                "n_hdv": 6,
                "duration": 40,
                "policy_frequency": 5,
                "simulation_frequency": 15,
                "speed_limit": 12.0,
                "cav_initial_speed": 8.0,
                "cav_distance_range": [25.0, 60.0],  # [m] to the merge point at start
                "exit_distance": 40.0,  # [m] past the merge point
                "screen_width": 900,
                "screen_height": 300,
                "centering_position": [0.6, 0.5],
                "scaling": 4.0,
                "show_trajectories": False,
            }
        )
        return config

    # -- road ---------------------------------------------------------------
    def _make_road(self) -> None:
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS, LineType.STRIPED, LineType.NONE
        vmax = self.config["speed_limit"]
        net.add_lane("a", "b", StraightLane([0, 0], [MERGE_X, 0], line_types=[c, c],
                                            priority=1, speed_limit=vmax))
        net.add_lane("b", "c", StraightLane([MERGE_X, 0], [MERGE_X + EXIT_LENGTH, 0],
                                            line_types=[c, c], priority=1, speed_limit=vmax))
        net.add_lane("j", "b", StraightLane(RAMP_START, [MERGE_X, 0], line_types=[c, c],
                                            priority=0, speed_limit=vmax))
        self.road = RegulatedRoad(network=net, np_random=self.np_random,
                                  record_history=self.config["show_trajectories"],
                                  neighbour_vehicles_connected_lanes=True)

    # -- vehicles -------------------------------------------------------------
    def _place(self, lane_index, distance_to_merge, speed, vehicle_cls):
        lane = self.road.network.get_lane(lane_index)
        longitudinal = lane.length - distance_to_merge
        return vehicle_cls(self.road, lane.position(longitudinal, 0.0), speed=speed,
                           heading=lane.heading_at(longitudinal))

    def _free(self, lane_index, d, min_gap=12.0) -> bool:
        lane = self.road.network.get_lane(lane_index)
        p = lane.position(lane.length - d, 0.0)
        return all(np.linalg.norm(v.position - p) >= min_gap for v in self.road.vehicles)

    def _make_vehicles(self) -> None:
        rng = self.np_random
        roads = [("a", "b", 0), ("j", "b", 0)]
        lo, hi = self.config["cav_distance_range"]

        self.controlled_vehicles = []
        v0 = self.config["cav_initial_speed"]
        for ego_id in range(self.config["controlled_vehicles"]):
            lane_index = roads[ego_id % 2]
            for _ in range(50):
                d = rng.uniform(lo, hi)
                if self._free(lane_index, d):
                    break
            ego = self._place(lane_index, d, v0, self.action_type.vehicle_class)
            ego.plan_route_to("c")
            ego.speed_index = ego.speed_to_index(v0)
            ego.target_speed = ego.index_to_speed(ego.speed_index)
            self.road.vehicles.append(ego)
            self.controlled_vehicles.append(ego)

        hdv_cls = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["n_hdv"]):
            lane_index = roads[0] if rng.uniform() < 0.7 else roads[1]
            max_d = 79.0 if lane_index == roads[1] else 190.0
            for _ in range(50):
                d = rng.uniform(10.0, max_d)
                if self._free(lane_index, d, min_gap=15.0):
                    v = self._place(lane_index, d, 8.0 + rng.normal(), hdv_cls)
                    v.plan_route_to("c")
                    v.randomize_behavior()
                    self.road.vehicles.append(v)
                    break

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self._start_episode()

    def step(self, action):
        return self.arbitration_step(action)

    def _reward(self, action) -> float:
        return 0.0  # CAVBench evaluates with its own metrics, not rewards

    def has_arrived(self, vehicle) -> bool:
        return (vehicle.lane_index[:2] == ("b", "c")
                and vehicle.lane.local_coordinates(vehicle.position)[0] >= self.config["exit_distance"])
