"""Functionality shared by every CAVBench scenario.

:class:`ArbitrationEnvMixin` adds, on top of a HighwayEnv environment:

* a vehicle class for CAVs whose acceleration is physically bounded;
* trajectory recording at the simulation frequency (15 Hz), used for PET;
* the noisy shared view of the scene (:meth:`observe`) given to arbiters;
* stable integer ids for vehicles, and the benchmark's termination rules.
"""
from __future__ import annotations

import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle

from ..geometry import PathCache, remaining_route
from ..noise import GaussianNoise
from ..state import SceneObservation, SharedState


class CAVehicle(MDPVehicle):
    """A connected autonomous vehicle with bounded longitudinal acceleration.

    HighwayEnv's default speed controller is a pure proportional controller,
    which can command unrealistic accelerations (e.g. -15 m/s^2 from 9 m/s).
    """

    ACC_LIMITS = (-5.0, 3.0)  # [m/s^2], overridden from the env config

    def speed_control(self, target_speed: float) -> float:
        acc = super().speed_control(target_speed)
        return float(np.clip(acc, *self.ACC_LIMITS))


ARBITRATION_DEFAULTS = {
    "noise_level": "none",  # one of NOISE_LEVELS, or set sigma_pos / sigma_vel
    "sigma_pos": None,
    "sigma_vel": None,
    "assumed_noise": None,  # noise the arbiter assumes (None = the true noise)
    "spawn_during_episode": False,
    "cav_acc_limits": [-5.0, 3.0],
    "cav_initial_speed": 5.0,
    "exit_distance": 15.0,
    "record_trajectories": True,
    "path_step": 0.5,
    "fast_observation": True,
}


class ArbitrationEnvMixin:
    """Mixin for HighwayEnv environments used with CAVBench arbiters."""

    # -- configuration -------------------------------------------------
    @property
    def noise(self) -> GaussianNoise:
        c = self.config
        if c.get("sigma_pos") is not None or c.get("sigma_vel") is not None:
            return GaussianNoise(c.get("sigma_pos") or 0.0, c.get("sigma_vel") or 0.0)
        return GaussianNoise.from_level(c["noise_level"])

    @property
    def assumed_noise(self) -> GaussianNoise:
        a = self.config.get("assumed_noise")
        if a is None:
            return self.noise
        return GaussianNoise.from_level(a) if isinstance(a, str) else GaussianNoise(*a)

    @property
    def cav_target_speeds(self) -> tuple[float, ...]:
        cfg = self.config["action"].get("action_config", self.config["action"])
        return tuple(float(v) for v in cfg.get("target_speeds", (0.0, 4.5, 9.0)))

    def define_spaces(self) -> None:
        """Arbiters read :meth:`observe`, not HighwayEnv's observation, which is
        expensive to build (pandas); it is skipped unless ``fast_observation``
        is False."""
        super().define_spaces()
        if self.config.get("fast_observation", True):
            self.observation_type.observe = lambda: np.zeros(1, dtype=np.float32)

    # -- episode bookkeeping ---------------------------------------------
    def _start_episode(self) -> None:
        CAVehicle.ACC_LIMITS = tuple(self.config["cav_acc_limits"])
        for v in self.controlled_vehicles:
            v.__class__ = CAVehicle
        self.path_cache = PathCache(self.road.network, self.config["path_step"])
        self._vid: dict[int, int] = {}
        self.vehicle_paths: dict = {}
        self.vehicle_is_cav: dict[int, bool] = {}
        self.trajectory: list[tuple] = []
        self.arrived_vids: set[int] = set()
        self.sim_time = 0.0
        self._record()

    def vid(self, vehicle) -> int:
        key = id(vehicle)
        if key not in self._vid:
            new = len(self._vid)
            self._vid[key] = new
            self.vehicle_paths[new] = self.path_cache.path_for(vehicle)
            self.vehicle_is_cav[new] = vehicle in self.controlled_vehicles
        return self._vid[key]

    def _record(self) -> None:
        if not self.config["record_trajectories"]:
            return
        for v in self.road.vehicles:
            self.trajectory.append(
                (self.sim_time, self.vid(v), float(v.position[0]), float(v.position[1]),
                 float(v.speed), v in self.controlled_vehicles, bool(v.crashed))
            )

    def _simulate(self, action=None) -> None:
        """HighwayEnv's simulation loop, plus trajectory recording every frame."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        dt = 1 / self.config["simulation_frequency"]
        for frame in range(frames):
            if action is not None and not self.config["manual_control"] and self.steps % frames == 0:
                self.action_type.act(action)
            self.road.act()
            self.road.step(dt)
            self.steps += 1
            self.sim_time += dt
            self._record()
            if frame < frames - 1:
                self._automatic_rendering()
        self.enable_auto_render = False

    def _remove_crashed_hdvs(self) -> None:
        """As in the benchmark: HDV-HDV crashes are cleared; CAV crashes end the episode."""
        if not any(v.crashed for v in self.controlled_vehicles):
            self.road.vehicles = [
                v for v in self.road.vehicles if v in self.controlled_vehicles or not v.crashed
            ]

    def _remove_arrived(self) -> None:
        """Arrived CAVs leave the scene (otherwise HighwayEnv loops them back
        into the network at the end of their route). Their objects stay in
        ``controlled_vehicles`` so the action tuple keeps its length."""
        for v in self.controlled_vehicles:
            if v in self.road.vehicles and not v.crashed and self.has_arrived(v):
                self.road.vehicles.remove(v)
                self.arrived_vids.add(self.vid(v))

    def _remove_exited_hdvs(self) -> None:
        """Human-driven vehicles leave when they reach the end of their route."""
        keep = []
        for v in self.road.vehicles:
            if v not in self.controlled_vehicles and len(remaining_route(v)) == 1:
                lane = self.road.network.get_lane(v.lane_index)
                if lane.local_coordinates(v.position)[0] >= lane.length - 4 * v.LENGTH:
                    continue
            keep.append(v)
        self.road.vehicles = keep

    def arbitration_step(self, action):
        obs, reward, terminated, truncated, info = AbstractEnv.step(self, action)
        self._remove_crashed_hdvs()
        self._remove_exited_hdvs()
        self._remove_arrived()
        info["cav_crashed"] = [bool(v.crashed) for v in self.controlled_vehicles]
        info["cav_arrived"] = [bool(self.has_arrived(v)) for v in self.controlled_vehicles]
        return obs, reward, terminated, truncated, info

    def _is_terminated(self) -> bool:
        return any(v.crashed for v in self.controlled_vehicles) or all(
            self.has_arrived(v) for v in self.controlled_vehicles
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    # -- the shared, noisy view ------------------------------------------
    def observe(self, rng: np.random.Generator) -> SceneObservation:
        """Noisy shared states of all vehicles (Step 1 of the method)."""
        vehicles = list(self.road.vehicles)
        pos = np.array([v.position for v in vehicles], dtype=float).reshape(-1, 2)
        spd = np.array([v.speed for v in vehicles], dtype=float)
        noisy_pos, noisy_spd = self.noise.perturb(pos, spd, rng)
        states = {}
        for v, p, sp in zip(vehicles, noisy_pos, noisy_spd):
            path = self.path_cache.path_for(v)
            s, _ = path.project(p)
            is_cav = v in self.controlled_vehicles
            states[self.vid(v)] = SharedState(
                vid=self.vid(v), is_cav=is_cav, position=p, speed=float(sp), path=path, s=s,
                target_speed=float(v.target_speed) if is_cav else None,
                arrived=bool(is_cav and self.has_arrived(v)),
            )
        assumed = self.assumed_noise
        return SceneObservation(
            time=float(self.time), states=states,
            sigma_pos=assumed.sigma_pos, sigma_vel=assumed.sigma_vel,
            cav_target_speeds=self.cav_target_speeds,
            policy_dt=1.0 / self.config["policy_frequency"],
        )

    def cav_vids(self) -> list[int]:
        """All CAVs, in action-tuple order (including those that have left)."""
        return [self.vid(v) for v in self.controlled_vehicles]

    def active_cav_vids(self) -> list[int]:
        """CAVs still in the scene."""
        return [self.vid(v) for v in self.controlled_vehicles if v in self.road.vehicles]
