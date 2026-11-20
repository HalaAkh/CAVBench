"""CoDrivingLLM reproduced from its released code, inside CAVBench.

CoDrivingLLM: Fang et al., "Towards interactive and learnable cooperative
driving automation: a large language model-driven decision-making framework",
IEEE Trans. Veh. Technol. 74(8), 2025. Code: https://github.com/FanGShiYuu/CoDrivingLLM
(files Run_multi_CAV_LLM.py, llm_controller/llm_agent_negotiation_system.py,
llm_controller/llm_agent_action.py, llm_controller/prompt_llm.py).

This module re-implements that code step by step, so that the method runs in
CAVBench exactly as released. Every function below names the benchmark
function it mirrors. The prompt texts and the safety-tool messages are the
benchmark's own, reproduced verbatim (the repository has no licence: they are
used here only to reproduce the published method, with attribution).

What one decision step (0.2 s) does, as in Run_multi_CAV_LLM.py:

1. negotiation (LlmAgent_negotiation_module): find conflicting CAV pairs
   (routes within 1 m of each other, both vehicles within 50 m of the conflict
   point and before it), describe them to the LLM as a "traffic police
   officer", which answers who passes first. Called every step, even when
   there is no conflict.
2. decision (LlmAgent_action_module), for each CAV in turn: cap its speed at
   5 m/s (the benchmark does this to the vehicle itself), build the scenario
   text with the benchmark's tools (available actions, vehicle ahead/behind in
   the same lane, TTC-based safety messages, dTTCP-based conflict safety for
   the pairs where it should pass second), and ask the LLM for IDLE, FASTER
   or SLOWER.
3. the CAVs are MDPVehicles with target speeds 0, 5, 10, 15, 20 m/s, initial
   target 5 m/s, no acceleration limit and no extra safety layer
   (safety_guarantee False in the benchmark).

As released: gpt-4o-mini, prompt sent as a single system message, temperature
not set (API default), memory not used (``past_memory = ''`` in
llm_agent_action.py; the memory database in the repository is empty).

Memory module (option ``memory``, off by default). The released code contains
the paper's memory module but switches it off with two commented lines whose
comments say how to switch it on: ``memory_update`` (line 46, "active this line
to restore new memory during interaction") builds the database, and
``relative_memory`` (line 159, "with this line to active memory retrival, active
line46 to build your own database before you output past memory") inserts the
two most similar past decisions into the decision prompt (top_k=2, the "2-shot"
of the paper's Table I). ``memory={"mode": "build"}`` does what line 46 does,
``memory={"mode": "use"}`` what line 159 does, with the functions below
reproduced from llm_agent_action.py, memory.py and prompt_llm.generate_comment.
The vector store (Chroma with langchain's OpenAIEmbeddings, i.e.
text-embedding-ada-002 in the pinned langchain 0.0.335) is replaced by a list of
the same embeddings searched by squared L2 distance, Chroma's default; ties
(identical texts) are broken by insertion order.

The only differences that remain are listed in docs/codrivingllm_repro.md
(CAVBench's own road/traffic generation, noisy shared states when an
experiment adds noise, arrived CAVs leaving the scene, and an unusable LLM
answer becoming IDLE instead of crashing the run).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import ACTION_NAMES, FASTER, IDLE, SLOWER, Arbiter
from .codrivingllm_wrapper import CACHE_DIR, PRICES, require_api_key

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------
# DiscreteMetaAction.ACTIONS_LONGI of the benchmark's highway_env fork
BENCH_ACTIONS = {1: "IDLE", 3: "FASTER", 4: "SLOWER"}
TO_CAVBENCH = {1: IDLE, 3: FASTER, 4: SLOWER}
# prompt_llm.py
ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
ACTIONS_DESCRIPTION = {
    0: "change lane to the left of the current lane,",
    1: "remain in the current lane with current speed",
    2: "change lane to the right of the current lane",
    3: "accelerate the vehicle",
    4: "decelerate the vehicle",
}
# MDPVehicle of the fork: SPEED_COUNT 5, SPEED_MIN 0, SPEED_MAX 20
CAV_TARGET_SPEEDS = (0.0, 5.0, 10.0, 15.0, 20.0)
BENCH_LANE_SPEED_LIMIT = 5.0  # speed_limit of every lane of the benchmark intersection

SCENES = {
    # speed cap of llm_controller_run, reaction range of is_conflict, which
    # vehicles negotiate (detect_conflicts), speed limit used by cal_ttcp
    "intersection": {"speed_cap": 5.0, "reaction_range": 50.0, "cav_only": True,
                     "ttcp_speed_limit": BENCH_LANE_SPEED_LIMIT},
    "merge": {"speed_cap": 20.0, "reaction_range": 120.0, "cav_only": False,
              "ttcp_speed_limit": None},  # None: the speed limit of the ego's lane
}

# ---------------------------------------------------------------------------
# Prompt texts (prompt_llm.py, PRE_DEF_PROMPT; intersection versions)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE_PREFIX_intersection = """
    You are ChatGPT, a large language model trained by OpenAI.
    You are now act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
    This environment you need to face is the intersection.
    The information in 'current scenario' :

    """

TRAFFIC_RULES_intersection = """
    1. Try to keep a safe distance to the car in front of you.
    2. Remeber you can't change your lane. In this intersection environment, you just can choose three kinds of actions to control the longitudinal movement of the vehicle: IDLE, FASTER, SLOWER.
    """

DECISION_CAUTIONS_intersection = """
    1. You must output a decision when you finish this task. Your final output decision must be unique and not ambiguous. For example you cannot say "I can either keep lane or accelerate at current time".
    2. You need to always remember your current lane ID, your available actions and available lanes before you make any decision.
    3. Once you have a decision, you should check the safety with all the vehicles affected by your decision.
    4. If you verify a decision is unsafe, you should start a new one and verify its safety again from scratch.
    5. The most important information is: Your final decision have to be one of the following action: IDLE, FASTER, SLOWER (you can not output other decisions like ACCELERATE, DECELERATE...)
    """

SAFETY_STEPS = """\nTo check decision safety you should follow steps:
      Step 1: Get the vehicles in this lane that you may affect. Acceleration, deceleration and idle action affect the Current lane, while left and right lane changes affect the Adjacent lane.
      Step 2: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
      Step 3: If you find There is no car driving on your "current lane" and you have no conflict with any other vehicle, you can drive faster ! but not too fast to follow the traffic rules.
      Step 4: If you want to make lane change consider :"Safety Assessment for Lane Changes:" Safe means it is safe to change ,If you want to do IDLE, FASTER, SLOWER, you should consider "Safety Assessment in Current Lane:"
      """


def negotiation_prompt(conflicting_info: str) -> str:
    """LlmAgent_negotiation_module.send_to_chatgpt"""
    return (
        "You are simulating as a traffic police officer overseeing traffic conflicts. Here are the current conflict scenarios:\n"
        f"{conflicting_info}\n\n"
        "For each conflict, decide which vehicle should pass first and which should pass later to ensure safety:\n"
        "Output your answer in the following format:\n"
        "```\n"
        "Final Answer: \n"
        "    \"decisions\": [\n"
        "        {\"first_vehicle\": \"<vehicle_i>\", \"second_vehicle\": \"<vehicle_j>\"},\n"
        "        {\"first_vehicle\": \"<vehicle_i>\", \"second_vehicle\": \"<vehicle_j>\"},\n"
        "        ...\n"
        "    ]\n"
        "```\n"
    )


def decision_prompt(current_scenario: str, negotiation_results: str, past_memory: str = "") -> str:
    """LlmAgent_action_module.send_to_chatgpt (intersection branch)"""
    return (f"{SYSTEM_MESSAGE_PREFIX_intersection}"
            f"You, the 'ego' car, are now driving. You have already driven for some seconds.\n"
            "Here is the current scenario:\n"
            f"{current_scenario}\n\n"
            "There are several rules you need to follow when you drive:\n"
            f"{TRAFFIC_RULES_intersection}\n\n"
            "Here are your attention points:\n"
            f"{DECISION_CAUTIONS_intersection}\n\n"
            "Here is your action when scenarios are similar to the current scenario in the past, you should learn from past memory try not to take the cation that cause more danger:\n"
            f"{past_memory}\n\n"
            "Based on the planning trajectory, you have the following conflicts with other vehicles.\n"
            "Here are the conflicts and the suggested passing orders (when you are suggested to passes second, better to slow down): \n"
            f"{negotiation_results}\n\n"
            "Once you make a final decision, output it in the following format:\n"
            "```\n"
            "Final Answer: \n"
            "    \"decision\": {\"<ego car's decision, ONE of the available actions (decision have to be one of the following action!!!:  LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER)>\"},\n"
            "```\n")


# ---------------------------------------------------------------------------
# The vehicles as the benchmark's code sees them
# ---------------------------------------------------------------------------
class BenchVehicle:
    """What the benchmark functions read from a vehicle. Position and speed come
    from CAVBench's shared (possibly noisy) observation; lane, route, destination
    and speed index are shared exactly. ``str()`` gives the benchmark's vehicle
    name, e.g. ``MDPVehicle #3: [ 92.3  -2. ]``."""

    __slots__ = ("vid", "is_cav", "position", "speed", "lane_index", "route_key", "route_pts",
                 "destination", "speed_index", "speed_count")

    def __init__(self, vid, is_cav, position, speed, lane_index, route_key, route_pts,
                 destination, speed_index=None, speed_count=None):
        self.vid, self.is_cav = vid, is_cav
        self.position = np.asarray(position, dtype=float)
        self.speed = speed
        self.lane_index = lane_index
        self.route_key, self.route_pts, self.destination = route_key, route_pts, destination
        self.speed_index, self.speed_count = speed_index, speed_count

    @property
    def name(self) -> str:
        return f"{'MDPVehicle' if self.is_cav else 'IDMVehicle'} #{self.vid}"

    def __str__(self) -> str:  # Vehicle.__str__ of the fork
        return f"{self.name}: {self.position}"

    __repr__ = __str__


def _dist(a, b) -> float:
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _div(a, b):
    """Division with numpy semantics (inf / nan instead of an exception), as the
    benchmark divides numpy floats."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.float64(a) / np.float64(b)


# ---------------------------------------------------------------------------
# Negotiation module (llm_agent_negotiation_system.py)
# ---------------------------------------------------------------------------
class RouteGeometry:
    """record_route_xy / get_lane_xy: every lane of the planned route sampled
    with 1000 points; plus, per pair of routes, the squared distance from each
    point of one route to the other route (cached: routes do not change)."""

    def __init__(self, network) -> None:
        self.network = network
        self._pts: dict[tuple, np.ndarray] = {}
        self._pair: dict[tuple, np.ndarray] = {}

    def points(self, route_key: tuple) -> np.ndarray:
        if route_key not in self._pts:
            from highway_env.road.lane import CircularLane, StraightLane

            xs, ys = [], []
            for a, b in route_key:
                lane = self.network.get_lane((a, b, 0))  # lane id set to 0, as get_lane_xy
                if isinstance(lane, StraightLane):
                    x = np.linspace(lane.start[0], lane.end[0], 1000)
                    y = np.linspace(lane.start[1], lane.end[1], 1000)
                elif isinstance(lane, CircularLane):
                    theta = np.linspace(lane.start_phase, lane.end_phase, 1000)
                    x = lane.center[0] + lane.radius * np.cos(theta)
                    y = lane.center[1] + lane.radius * np.sin(theta)
                else:  # (the benchmark has only these two lane types)
                    s = np.linspace(0.0, lane.length, 1000)
                    x, y = np.array([lane.position(si, 0.0) for si in s]).T
                xs.append(x)
                ys.append(y)
            self._pts[route_key] = (np.column_stack((np.hstack(xs), np.hstack(ys)))
                                    if xs else np.zeros((0, 2)))
        return self._pts[route_key]

    def min_sq_dist(self, key_i: tuple, key_j: tuple) -> np.ndarray:
        """points_distance_between_routes: for each point of route i, the min
        squared distance to route j (same arithmetic as the benchmark)."""
        k = (key_i, key_j)
        if k not in self._pair:
            pi, pj = self.points(key_i), self.points(key_j)
            out = np.empty(len(pi))
            for start in range(0, len(pi), 500):
                c = pi[start:start + 500]
                d = (c[:, 0:1] - pj[None, :, 0]) ** 2 + (c[:, 1:2] - pj[None, :, 1]) ** 2
                out[start:start + 500] = d.min(axis=1)
            self._pair[k] = out
        return self._pair[k]


def is_conflict(geo: RouteGeometry, vi: BenchVehicle, vj: BenchVehicle, reaction_range: float):
    """LlmAgent_negotiation_module.is_conflict"""
    route_i, route_j = vi.route_pts, vj.route_pts
    if len(route_i) != 0 and len(route_j) != 0 and vi.lane_index != vj.lane_index:
        points_distance = geo.min_sq_dist(vi.route_key, vj.route_key)
        if np.min(points_distance) < 1:
            cp_index = np.where(points_distance < 1)[0][0]  # first element < 1
            cp = route_i[cp_index]
            distance_to_conflict_point = _dist(vi.position, cp)
            vi_to_dest = _dist(vi.position, vi.destination)
            vi_cp_to_dest = _dist(cp, vi.destination)
            vj_to_dest = _dist(vj.position, vj.destination)
            vj_cp_to_dest = _dist(cp, vj.destination)
            j_points_distance = geo.min_sq_dist(vj.route_key, vi.route_key)
            j_cp = route_j[np.where(j_points_distance < 1)[0][0]]
            vj_to_conflict_point = _dist(vj.position, j_cp)
            if (distance_to_conflict_point < reaction_range and vj_to_conflict_point < reaction_range) \
                    and vi_to_dest > vi_cp_to_dest and vj_to_dest > vj_cp_to_dest:
                return True, distance_to_conflict_point
    return False, None


def detect_conflicts(geo, vehicle_list, reaction_range):
    """LlmAgent_negotiation_module.detect_conflicts"""
    out = {}
    for i, vi in enumerate(vehicle_list):
        found = {}
        for j, vj in enumerate(vehicle_list):
            if i != j:
                ok, d = is_conflict(geo, vi, vj, reaction_range)
                if ok:
                    found[vj] = d
        out[vi] = found
    return out


def conflicting_pairs(vehicle_list, conflict) -> list[dict]:
    """The pair list built in LlmAgent_negotiation_module.send_to_chatgpt."""
    info = []
    for i in range(len(vehicle_list)):
        vi = vehicle_list[i]
        if vi in conflict:
            for j in range(i):
                vj = vehicle_list[j]
                if vj in conflict[vi] and vi in conflict[vj]:
                    info.append({"vehicle_i": vi, "vehicle_i speed": vi.speed,
                                 "vehicle_i distance to conflict": conflict[vi][vj],
                                 "vehicle_j": vj, "vehicle_j speed": vj.speed,
                                 "vehicle_j distance to conflict": conflict[vj][vi]})
    return info


def format_conflicts(info: list[dict]) -> str:
    return "\n".join(
        [f"- Vehicle i: {g['vehicle_i']}, Vehicle i Speed: {g['vehicle_i speed']} m/s, "
         f"Vehicle i distance to conflict point: {g['vehicle_i distance to conflict']} meters, "
         f"- Vehicle j: {g['vehicle_j']}, Vehicle j Speed: {g['vehicle_j speed']} m/s, "
         f"Vehicle j distance to conflict point: {g['vehicle_j distance to conflict']} meters, "
         for g in info])


# ---------------------------------------------------------------------------
# Decision module (llm_agent_action.py and prompt_llm.py)
# ---------------------------------------------------------------------------
_PAIR_RE = re.compile(r'"first_vehicle": "(MDPVehicle #[0-9]+|IDMVehicle #[0-9]+)", '
                      r'"second_vehicle": "(MDPVehicle #[0-9]+|IDMVehicle #[0-9]+)"')
_SECOND_RE = re.compile(r"- You have conflict with (MDPVehicle #[0-9]+|IDMVehicle #[0-9]+). "
                        r"It is suggested that you should passes second.")


def negotiation_results_for(ego: BenchVehicle, negotiation_reply: str) -> str:
    """extract_vehicle_conflicts + transfer_negotiation_prompts_to_results"""
    vid = str(ego).split(":")[0].strip()
    out = ""
    for first, second in _PAIR_RE.findall(negotiation_reply):
        if first == vid:
            out += f"- You have conflict with {second}. It is suggested that you should passes first.\n"
        elif second == vid:
            out += f"- You have conflict with {first}. It is suggested that you should passes second.\n"
    return out


def available_actions(ego: BenchVehicle, network) -> list[int]:
    """getAvailableActions._get_available_actions1, called (as in the benchmark)
    with is_intersection=False: lane changes from side lanes, then speeds."""
    actions = [1]
    if network is not None:
        try:
            side = network.side_lanes(ego.lane_index)
        except Exception:
            side = []
        for l_index in side:
            if l_index[2] < ego.lane_index[2] and network.get_lane(l_index).is_reachable_from(ego.position):
                actions.append(0)
            if l_index[2] > ego.lane_index[2] and network.get_lane(l_index).is_reachable_from(ego.position):
                actions.append(2)
    if ego.speed_index < ego.speed_count - 1:
        actions.append(3)
    if ego.speed_index > 0:
        actions.append(4)
    return actions


def available_actions_text(actions: list[int]) -> str:
    """getAvailableActions.inference (non-intersection branch, as called)"""
    out = "You can ONLY use one of the following actions: \n "
    for a in actions:
        out += ACTIONS_ALL[a] + "--" + ACTIONS_DESCRIPTION[a] + "; \n"
    if 1 in actions:
        out += "You should check idle action as FIRST priority. "
    if 0 in actions or 2 in actions:
        out += "For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. "
    if 3 in actions:
        out += "Consider acceleration action carefully. "
    if 4 in actions:
        out += "The deceleration action is LAST priority. "
    out += SAFETY_STEPS
    return out


def lane_involved_cars(lane_id: str, ego: BenchVehicle, vehicles: list[BenchVehicle]):
    """getLaneInvolvedCar.inference -> (leadingCar, rearingCar, message)"""
    lane_vehicles = [v for v in vehicles if v is not ego and v.lane_index == ego.lane_index]
    dest = ego.destination
    ego_d = _dist(ego.position, dest)
    distance = [_dist(v.position, dest) - ego_d for v in lane_vehicles]
    rearing_distance, rearing_idx = min(((d, i) for i, d in enumerate(distance) if d > 0),
                                        default=(float("inf"), -1), key=lambda x: x[0])
    leading_distance, leading_idx = max(((d, i) for i, d in enumerate(distance) if d < 0),
                                        default=(-float("inf"), -1), key=lambda x: x[0])
    if leading_idx == -1 and rearing_idx == -1:
        return None, None, f"There is no car driving on {lane_id}. This lane is safe."
    if leading_idx == -1:
        r = lane_vehicles[rearing_idx]
        return None, r, (f"{r} is driving on {lane_id} which is the same lane as you are driving, and it's "
                         f"driving behind the ego car. You need to make sure that your actions do not "
                         f"conflict with each of the vehicles mentioned.")
    if rearing_idx == -1:
        f_ = lane_vehicles[leading_idx]
        return f_, None, (f"{f_} is driving at {round(f_.speed, 1)}m/s on {lane_id} which is the same lane "
                          f"as you are driving, and it's driving in front of the ego car for "
                          f"{round(distance[leading_idx], 2)} meters. You need to make sure that your "
                          f"actions do not conflict with each of the vehicles mentioned.")
    r, f_ = lane_vehicles[rearing_idx], lane_vehicles[leading_idx]
    return f_, r, (f"{f_} and {r} are driving on {lane_id} which is the same lane as you are driving, and "
                   f"{f_} is driving at {round(f_.speed, 1)}m/s in front of the ego car for "
                   f"{round(distance[leading_idx], 2)} meters, while {r} is driving behind the ego car. "
                   f"You need to make sure that your actions do not conflict with each of the vehicles "
                   f"mentioned.")


def acceleration_check(veh: BenchVehicle, ego: BenchVehicle) -> str:
    """isAccelerationConflictWithCar.inference (acceleration 4, length 5)"""
    vid = str(veh)
    if veh.lane_index == ego.lane_index:
        dest = ego.destination
        ego_d, veh_d = _dist(ego.position, dest), _dist(veh.position, dest)
        if ego_d >= veh_d:
            ttc = _div(ego_d - veh_d - 5.0 * 2, ego.speed + 4.0 - veh.speed)
            if ttc > 10 or ttc < -2:
                return f"acceleration is safe with {vid}"
            elif 10 >= ttc > 5:
                return f"acceleration may not safe with {vid}, should be careful if you want to accelerate"
            elif 5 >= ttc > 3:
                return "acceleration will cause danger, you can not accelerate"
            else:
                return "acceleration will cause serious danger, must decelerate."
        return f"acceleration is safe with {vid}"
    return f"acceleration is safe with {vid}"


def keep_speed_check(veh: BenchVehicle, ego: BenchVehicle) -> str:
    """isKeepSpeedConflictWithCar.inference (length 5)"""
    vid = str(veh)
    if veh.lane_index != ego.lane_index:
        return f"{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input."
    dest = ego.destination
    ego_d, veh_d = _dist(ego.position, dest), _dist(veh.position, dest)
    if ego_d >= veh_d:
        ttc = _div(ego_d - veh_d - 5.0 * 2, ego.speed - veh.speed)
        if ttc > 20 or ttc < -3:
            return f"keep lane with current speed is safe with {vid}"
        elif -3 < ttc <= 0:
            return f"keep lane with current speed may not safe with {vid}, should consider decelerate"
        elif 20 >= ttc > 5:
            return f"keep lane with current speed may not safe with {vid}, should consider decelerate"
        elif 5 >= ttc > 3:
            return "keep lane with current speed will cause danger, you should consider decelerate"
        else:
            return "keep lane with current speed will cause serious danger, must decelerate."
    return f"keep lane with current speed is safe with {vid}"


def deceleration_check(veh: BenchVehicle, ego: BenchVehicle) -> str:
    """isDecelerationSafe.inference (deceleration 6, length 5)"""
    vid = str(veh)
    if veh.lane_index == ego.lane_index:
        dest = ego.destination
        ego_d, veh_d = _dist(ego.position, dest), _dist(veh.position, dest)
        if veh_d >= ego_d:  # veh is the rearing vehicle of ego
            ttc = _div(veh_d - ego_d - 5.0, veh.speed - (ego.speed - 6.0))
            if ttc > 20 or ttc < -3:
                return f"deceleration with current speed is safe with {vid}"
            elif -3 < ttc <= 0:
                return f"deceleration with current speed may not safe with {vid}"
            elif 20 >= ttc > 5:
                return f"deceleration with current speed may not safe with {vid}"
            elif 5 >= ttc > 1:
                return ("deceleratione with current speed will cause danger, if you have no other choice, "
                        "try not to decelerate so fast as much as possible")
            else:
                return f"deceleration with current speed may be conflict with {vid}, you should maintain speed or accelerate"
        return f"deceleration with current speed is safe with {vid}"
    return f"deceleration with current speed is safe with {vid}"


def safety_in_current_lane(lane_cars: dict, ego: BenchVehicle) -> dict:
    """check_safety_in_current_lane: only the LAST non-empty of leadingCar /
    rearingCar is checked (as in the benchmark)."""
    out = {"acceleration_conflict": None, "keep_speed_conflict": None, "deceleration_conflict": None}
    car, which = None, None
    for key, value in lane_cars.items():
        if value is not None:
            car, which = value, key
    if car:
        if which == "leadingCar":
            out["acceleration_conflict"] = acceleration_check(car, ego)
        if which in ("rearingCar", "leadingCar"):
            out["keep_speed_conflict"] = keep_speed_check(car, ego)
        if which == "rearingCar":
            out["deceleration_conflict"] = deceleration_check(car, ego)
    return out


def cal_ttcp(speed_limit, veh_dis2cp, veh_v):
    """cal_ttcp (MAX_ACCELERATION 2)"""
    a = 2
    t_acc2max = (speed_limit - veh_v) / a
    dis_acc2max = veh_v * t_acc2max + 0.5 * a * t_acc2max ** 2
    if dis_acc2max < veh_dis2cp:
        return t_acc2max + (veh_dis2cp - dis_acc2max) / speed_limit
    v = np.sqrt(veh_v ** 2 + 2 * veh_dis2cp * a)
    return (v - veh_v) / a


def safety_with_conflict_vehicles(ego, negotiation_results, conflicting_info, speed_limit):
    """check_safety_with_conflict_vehicles"""
    safety = {"acceleration_conflict": None, "keep_speed_conflict": None, "deceleration_conflict": None}
    info = {"delta ttcp": None, "distance to conflict": None, "speed": None,
            "distance to conflict (others)": None, "speed (others)": None}
    level_max = 0
    for vehicle in _SECOND_RE.findall(negotiation_results):
        for g in conflicting_info:
            if (g["vehicle_i"] is ego and str(g["vehicle_j"]).split(":")[0].strip() == vehicle) or \
                    (g["vehicle_j"] is ego and str(g["vehicle_i"]).split(":")[0].strip() == vehicle):
                ttcp_i = cal_ttcp(speed_limit, g["vehicle_i distance to conflict"], g["vehicle_i speed"])
                ttcp_j = cal_ttcp(speed_limit, g["vehicle_j distance to conflict"], g["vehicle_j speed"])
                ttcp = ttcp_j - ttcp_i
                if abs(ttcp) > 8:
                    level = 0
                elif 8 >= abs(ttcp) > 5:
                    level = 1
                elif 5 >= abs(ttcp) > 2:
                    level = 2
                else:
                    level = 3
                if level > level_max:
                    level_max = level
                    if g["vehicle_i"] is ego:
                        info.update({"delta ttcp": ttcp_j - ttcp_i,
                                     "distance to conflict": g["vehicle_i distance to conflict"],
                                     "speed": g["vehicle_i speed"],
                                     "distance to conflict (others)": g["vehicle_i distance to conflict"],
                                     "speed (others)": g["vehicle_j speed"],
                                     "delta speed": round(g["vehicle_i speed"] - g["vehicle_j speed"], 1)})
                    else:
                        info.update({"delta ttcp": ttcp_i - ttcp_j,
                                     "distance to conflict": g["vehicle_j distance to conflict"],
                                     "speed": g["vehicle_j speed"],
                                     "distance to conflict (others)": g["vehicle_i distance to conflict"],
                                     "speed (others)": g["vehicle_i speed"],
                                     "delta speed": round(g["vehicle_j speed"] - g["vehicle_i speed"], 1)})
    safety["acceleration_conflict"] = {
        0: "acceleration is safe",
        1: "acceleration may not safe, should be careful if you want to accelerate",
        2: "acceleration will cause danger, you can not accelerate",
        3: "acceleration will cause serious danger, must decelerate.",
    }[level_max]
    return safety, info


def format_training_info(available_msg, lanes_adjacent, lane_cars_msg, lane_change_safety,
                         current_lane_safety, conflict_safety, most_dangerous) -> str:
    """format_training_info"""
    m = "Available Actions:\n"
    m += f"- {available_msg}\n"
    m += "\nLane Information:\n"
    m += f"- Current Lane ID: {lanes_adjacent['currentLaneID']}\n"
    m += f"- Left Adjacent Lane ID: {lanes_adjacent['leftLane'] or 'None'}\n"
    m += f"- Right Adjacent Lane ID: {lanes_adjacent['rightLane'] or 'None'}\n"
    m += "- Adjacent Lane ID is None stands for adjacent Lane not exist, you can only drive on current lane\n"
    m += "\nVehicles in the same the Lane with you:\n"
    for lane_id, car_info in lane_cars_msg.items():
        m += f"- Lane ID {lane_id}: {car_info}\n"
    m += "\nSafety Assessment for Lane Changes:\n"
    if isinstance(lane_change_safety, dict):
        m += f"- Left Lane Change: {'Safe' if lane_change_safety['left_lane_change_safe'] else 'Not Safe'}\n"
        m += f"- Right Lane Change: {'Safe' if lane_change_safety['right_lane_change_safe'] else 'Not Safe'}\n"
    else:
        m += lane_change_safety
    m += "\nSafety Assessment in Current Lane:\n"
    for action, s in current_lane_safety.items():
        m += f"- {action.capitalize().replace('_', ' ')}: {s}\n"
    m += "\nSafety Assessment with Vehicle Not in Current Lane but Have Conflict with Ego Vehicle:\n"
    for action, s in conflict_safety.items():
        m += f"- {action.capitalize().replace('_', ' ')}: {s}\n"
    if most_dangerous["delta ttcp"] is not None:
        d = most_dangerous["delta ttcp"]
        if d >= 5:
            relation = "much close"
        elif 5 > d >= 2:
            relation = "slight close"
        elif 2 > d >= -2:
            relation = "equal close"
        elif -2 > d >= -5:
            relation = "slight further"
        else:
            relation = "much further"
        m += "\n Currently, the most dangerous collision information with the ego vehicle is as follows:"
        m += (f"Ego is _{relation}_ to conflict point than other vehicles considering current speed and "
              f"distance, Ego vehicle speed minus other vehicle is {most_dangerous['delta speed']}")
    else:
        m += "\n Currently, ego vehicle do not have conflict \n"
        m += "\n Conflict info is empty \n"
    return m


def prompt_engineer(ego, vehicles, network, negotiation_results, conflicting_info, speed_limit) -> str:
    """LlmAgent_action_module.prompt_engineer"""
    msg0 = available_actions_text(available_actions(ego, network))
    lane_id = str(ego.lane_index)
    lanes_adjacent = {"currentLaneID": lane_id, "leftLane": "", "rightLane": ""}  # getAvailableLanes
    leading, rearing, info = lane_involved_cars(lane_id, ego, vehicles)
    lane_cars_msg = {lane_id: info}
    lane_cars = {"leadingCar": leading, "rearingCar": rearing}
    lane_change = "There is no need to assess lane change safety."
    current = safety_in_current_lane(lane_cars, ego)
    conflict_safety, most_dangerous = safety_with_conflict_vehicles(
        ego, negotiation_results, conflicting_info, speed_limit)
    return format_training_info(msg0, lanes_adjacent, lane_cars_msg, lane_change, current,
                                conflict_safety, most_dangerous)


def extract_decision(response_content: str):
    """LlmAgent_action_module.extract_decision (intersection branch)"""
    try:
        start = response_content.find('"decision": {') + len('"decision": {')
        end = response_content.find("}", start)
        decision = response_content[start:end].strip('"')
        if "FASTER" in decision:
            decision = "FASTER"
        elif "SLOWER" in decision:
            decision = "SLOWER"
        elif "IDLE" in decision:
            decision = "IDLE"
        return decision
    except Exception:
        return None


def action_id(name) -> int:
    """get_action_id_from_name: -1 when not found (the benchmark's env then
    fails on the unknown action)"""
    if name is None:
        return -1
    for i, n in BENCH_ACTIONS.items():
        if n == name or n == name.upper():
            return i
    return -1


# ---------------------------------------------------------------------------
# Memory module (switched off in the released code; see the module docstring)
# ---------------------------------------------------------------------------
MEMORY_DIR = CACHE_DIR.parent / "llm_memory"
EMBED_MODEL = "text-embedding-ada-002"  # default of OpenAIEmbeddings in langchain 0.0.335
EMBED_PRICE = 0.10  # USD per million tokens (cost report only)


def generate_comment(relation, llm_action):
    """prompt_llm.generate_comment"""
    if relation == 'much close' or relation == 'slight close':
        if llm_action == 1 or llm_action == 3:
            return 'safe and good for efficiency'
        else:
            return 'not efficiency, consider FASTER next time'
    elif relation == 'equal close':
        return 'might not safe, you should be careful and check'
    else:
        if llm_action == 1:
            return 'it is not safety to accelerate, don not accelerate next time'
        elif llm_action == 3:
            return 'may not safe if your speed is high'
        else:
            return 'it is safe, good for you'


def relative_memory(memory, prompt_info):
    """LlmAgent_action_module.relative_memory"""
    experience = ""
    extract_prompt = prompt_info.strip().split('\n')
    query_scenario = '\n'.join(extract_prompt[-2:])  # only save the last two line of prompt_info which store the most dangerous conflict as memory page_content
    past_decisions = memory.retrieveMemory(query_scenario, top_k=2)
    for past_decision in past_decisions:
        experience += f"- Last time {past_decision['negotiation_result']}, you choose to {past_decision['final_action']}, it is {past_decision['comments']}\n"
    experience += f"Above messages are some examples of how you make a decision in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario."
    return experience


def memory_query(prompt_info: str) -> str:
    """The text relative_memory searches with (last two lines of the scenario)."""
    return '\n'.join(prompt_info.strip().split('\n')[-2:])


def memory_update(memory, prompt_info, llm_action):
    """LlmAgent_action_module.memory_update (llm_action = np.array([action id]))"""
    human_question = str(None)
    negotiation = str(None)  # memory store the dangerous info (which negotiation is ego to yield) for ego vehicle
    action = str(llm_action)
    saved_info = prompt_info.strip().split('\n')[-1]
    if saved_info != ' Conflict info is empty':
        relation = re.findall(r'_(.*?)_', saved_info)
        comments = generate_comment(relation[0], llm_action[0])
    else:
        comments = 'recommended to FASTER'
    memory.addMemory(str(saved_info), human_question, negotiation, action, comments)


class ReproMemory:
    """memory.DrivingMemory: a store of (page_content, metadata) searched by the
    embedding of page_content.

    Building (``write=True``): items are kept per episode and written when the
    episode ends (``flush``), one file per process, so that an interrupted
    episode leaves nothing behind and parallel workers never share a file.
    Using: all files ``repro_<name>.*.jsonl`` are loaded; an episode seed found
    in two files (an episode re-run) is kept once. Embeddings of the stored
    texts and of the queries are cached on disk.
    """

    def __init__(self, name: str = "intersection", embedding: str = "openai",
                 write: bool = False, load: bool = True) -> None:
        if embedding not in ("openai", "local"):
            raise ValueError("embedding must be 'openai' or 'local'")
        self.name, self.embedding, self.write = name, embedding, write
        self.items: list[tuple[str, dict]] = []
        self._pending: list[dict] = []
        self._episode = None
        self._vec_cache: dict[str, np.ndarray] = {}
        self._vectors = None
        self._client = None
        self.embed_calls = self.embed_tokens = self.hits = 0
        self._path = MEMORY_DIR / f"repro_{name}.{os.getpid()}.jsonl"
        self._emb_path = MEMORY_DIR / f"embeddings_{EMBED_MODEL}.{os.getpid()}.jsonl"
        if load:
            self._load()

    # -- storage -------------------------------------------------------------
    def _load(self) -> None:
        seen: dict[int, str] = {}
        for path in sorted(MEMORY_DIR.glob(f"repro_{self.name}.*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    ep = e.get("episode")
                    if seen.setdefault(ep, path.name) != path.name:
                        continue  # the same episode run again in another file
                    self.items.append((e["content"], e["metadata"]))
        if self.embedding == "openai":
            import base64
            for path in sorted(MEMORY_DIR.glob(f"embeddings_{EMBED_MODEL}.*.jsonl")):
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        try:
                            e = json.loads(line)
                            self._vec_cache[e["t"]] = np.frombuffer(base64.b64decode(e["v"]), dtype=np.float32)
                        except (ValueError, KeyError):
                            continue

    def __len__(self) -> int:
        return len(self.items)

    def start_episode(self, episode) -> None:
        self._pending, self._episode = [], episode

    def addMemory(self, sce_descrip, human_question, negotiation, action, comments):
        """DrivingMemory.addMemory"""
        meta = {"human_question": human_question, 'negotiation_result': negotiation,
                'final_action': action, 'comments': comments}
        self._pending.append({"episode": self._episode, "content": sce_descrip, "metadata": meta})

    def flush(self) -> None:
        if not (self.write and self._pending):
            self._pending = []
            return
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            for e in self._pending:
                f.write(json.dumps(e) + "\n")
        self._pending = []

    # -- embeddings ----------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        if self.embedding == "local":
            from .llm_memory import local_embedding
            return [local_embedding(t).astype(np.float32) for t in texts]
        todo = [t for t in dict.fromkeys(texts) if t not in self._vec_cache]
        if todo:
            import base64
            if self._client is None:
                require_api_key()
                from openai import OpenAI
                self._client = OpenAI(timeout=60.0, max_retries=6)
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(todo), 1000):
                batch = todo[i:i + 1000]
                r = self._client.embeddings.create(model=EMBED_MODEL, input=batch)
                self.embed_calls += 1
                if getattr(r, "usage", None) is not None:
                    self.embed_tokens += r.usage.total_tokens
                with self._emb_path.open("a", encoding="utf-8") as f:
                    for t, d in zip(batch, r.data):
                        v = np.asarray(d.embedding, dtype=np.float32)
                        self._vec_cache[t] = v
                        f.write(json.dumps({"t": t, "v": base64.b64encode(v.tobytes()).decode()}) + "\n")
        return [self._vec_cache[t] for t in texts]

    def prefetch(self, queries: list[str]) -> None:
        """Embed the stored texts (once) and the given queries in batches."""
        if self._vectors is None and self.items:
            self._vectors = np.stack(self._embed([c for c, _ in self.items]))
        if queries and self.embedding == "openai":
            self._embed(queries)

    def retrieveMemory(self, query_scenario, top_k=5):
        """DrivingMemory.retrieveMemory: the top_k nearest stored items
        (squared L2 distance, Chroma's default; ties in insertion order)."""
        if not self.items:
            return []
        self.prefetch([])
        q = self._embed([query_scenario])[0]
        d = ((self._vectors - q[None, :]) ** 2).sum(axis=1)
        idx = np.argsort(d, kind="stable")[:top_k]
        self.hits += len(idx)
        return [self.items[i][1] for i in idx]

    @property
    def cost_usd(self) -> float:
        return self.embed_tokens * EMBED_PRICE / 1e6


# ---------------------------------------------------------------------------
# LLM client: one system message, temperature not sent (as released)
# ---------------------------------------------------------------------------
class ReproLLM:
    """Calls the chat API exactly as the benchmark (``messages=[{"role": "system",
    "content": prompt}]``, no temperature unless one is given). Replies are
    cached on disk keyed by (episode seed, step, call, prompt), so an
    interrupted experiment resumes without paying twice, while identical
    prompts at different steps still get independent (sampled) answers."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float | None = None,
                 cache: bool = True, max_retries: int = 6, timeout: float = 60.0) -> None:
        self.model, self.temperature = model, temperature
        self.max_retries, self.timeout = max_retries, timeout
        self.calls = self.cache_hits = self.tokens_in = self.tokens_out = 0
        self._client = None
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        # One cache file per process: several experiment workers appending long
        # lines to one shared file can interleave them (corrupted lines). All
        # files codrivingllm_repro_<model>*.jsonl are read back.
        self._path = None
        if cache and model != "mock":
            self._path = CACHE_DIR / f"codrivingllm_repro_{model}.{os.getpid()}.jsonl"
            for path in sorted(CACHE_DIR.glob(f"codrivingllm_repro_{model}*.jsonl")):
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        try:
                            e = json.loads(line)
                            self._cache[e["key"]] = e["reply"]
                        except (ValueError, KeyError):
                            continue

    def _api(self, prompt: str) -> str:
        if self.model == "mock":
            return mock_reply(prompt)
        if self._client is None:
            require_api_key()
            from openai import OpenAI  # pip install openai

            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        kwargs = {"model": self.model, "messages": [{"role": "system", "content": prompt}]}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.chat.completions.create(**kwargs)
                if r.usage is not None:
                    with self._lock:
                        self.tokens_in += r.usage.prompt_tokens
                        self.tokens_out += r.usage.completion_tokens
                return r.choices[0].message.content or ""
            except Exception as exc:
                name = type(exc).__name__
                if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError",
                            "BadRequestError"):
                    raise RuntimeError(f"OpenAI API {name}: {exc}") from exc
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        return ""

    def __call__(self, prompt: str, context: str = "") -> str:
        key = hashlib.sha256(f"{self.model}|{self.temperature}|{context}|{prompt}".encode()).hexdigest()
        with self._lock:
            self.calls += 1
            if key in self._cache:
                self.cache_hits += 1
                return self._cache[key]
        reply = self._api(prompt)
        with self._lock:
            self._cache[key] = reply
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "context": context, "prompt": prompt,
                                        "reply": reply}) + "\n")
        return reply

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (self.tokens_in * pin + self.tokens_out * pout) / 1e6


_LINE_RE = re.compile(r"- Vehicle i: (\w+ #\d+): .*?Vehicle i Speed: ([-\d.e]+) m/s, Vehicle i distance "
                      r"to conflict point: ([-\d.e]+) meters, - Vehicle j: (\w+ #\d+): .*?Vehicle j "
                      r"Speed: ([-\d.e]+) m/s, Vehicle j distance to conflict point: ([-\d.e]+) meters")


def mock_reply(prompt: str) -> str:
    """Free stand-in for testing the pipeline (it is NOT the method): the
    vehicle that would reach the conflict point first passes first; the ego
    slows down when told to pass second or to decelerate, otherwise speeds up."""
    if prompt.startswith("You are simulating as a traffic police officer"):
        rows = []
        for m in _LINE_RE.finditer(prompt):
            vi, si, di, vj, sj, dj = m.groups()
            ti = float(di) / max(float(si), 0.1)
            tj = float(dj) / max(float(sj), 0.1)
            first, second = (vi, vj) if ti <= tj else (vj, vi)
            rows.append(f'        {{"first_vehicle": "{first}", "second_vehicle": "{second}"}}')
        return 'Final Answer: \n    "decisions": [\n' + ",\n".join(rows) + "\n    ]\n"
    if "must decelerate" in prompt or "you should passes second" in prompt:
        d = "SLOWER"
    else:
        d = "FASTER"
    return f'Final Answer: \n    "decision": {{"{d}"}},\n'


# ---------------------------------------------------------------------------
# The arbiter
# ---------------------------------------------------------------------------
class CoDrivingLLMReproArbiter(Arbiter):
    """CoDrivingLLM as released (see the module docstring).

    Options (defaults = as released):
      model="gpt-4o-mini"; temperature=None (not sent: API default);
      vehicle="benchmark" (CAV target speeds 0/5/10/15/20, initial target 5 m/s,
      no acceleration limit) or "cavbench" (CAVBench's CAVs, as for the other
      methods); speed_cap=True (the benchmark's 5 m/s cap at the intersection);
      safety=False (no extra safety layer; "follow" adds CAVBench's shared one);
      parallel=True (the per-CAV decision calls are independent, so they are
      sent together; the answers are the same as sending them one by one);
      memory=None (as released). {"mode": "build", "name": ...} stores every
      decision as line 46 of llm_agent_action.py would (no retrieval);
      {"mode": "use", "name": ...} adds the two most similar stored decisions
      to each decision prompt as line 159 would (memory frozen). Optional keys:
      "embedding" ("openai" or "local", for tests), "top_k" (2).
    """

    name = "codrivingllm_repro"

    def __init__(self, model: str = "gpt-4o-mini", llm_fn=None, temperature: float | None = None,
                 vehicle: str = "benchmark", speed_cap: bool = True, safety: bool | str = False,
                 parallel: bool = True, scene: str | None = None, memory: dict | None = None,
                 **kw) -> None:
        super().__init__(safety=safety, **kw)
        if vehicle not in ("benchmark", "cavbench"):
            raise ValueError("vehicle must be 'benchmark' or 'cavbench'")
        self.model, self.vehicle, self.speed_cap, self.parallel = model, vehicle, speed_cap, parallel
        self.scene = scene
        self._llm = llm_fn if llm_fn is not None else ReproLLM(model, temperature)
        self.env = None
        self._seed = 0
        self._step = 0
        self.n_negotiation = self.n_decision = self.unusable = self.llm_failures = 0
        self.memory_mode = None
        self.memory = None
        if memory:
            mode = memory.get("mode")
            if mode not in ("build", "use"):
                raise ValueError("memory mode must be 'build' or 'use'")
            if int(memory.get("top_k", 2)) != 2:
                raise ValueError("the released code retrieves top_k=2")
            self.memory_mode = mode
            self.memory = ReproMemory(memory.get("name", "intersection"),
                                      memory.get("embedding", "openai"),
                                      write=(mode == "build"), load=(mode == "use"))
            if mode == "use" and len(self.memory) == 0:
                raise RuntimeError(f"memory '{self.memory.name}' is empty: run the build step first "
                                   f"(configs/e1_codrivingllm_memory_build.yaml)")

    # -- episode setup ---------------------------------------------------
    def reset(self, seed=None) -> None:
        super().reset(seed)
        self._seed = 0 if seed is None else int(seed)
        self._step = 0
        self.n_negotiation = self.n_decision = self.unusable = self.llm_failures = 0
        c = self._llm
        self._usage0 = ((c.calls, c.cache_hits, c.tokens_in, c.tokens_out, c.cost_usd)
                        if isinstance(c, ReproLLM) else None)
        if self.memory is not None:
            self.memory.start_episode(self._seed)
            self._mem0 = (self.memory.embed_calls, self.memory.hits, self.memory.cost_usd)

    def attach(self, env) -> None:
        """Called by the runner after env.reset(): keeps the environment (to read
        planned routes and lanes, and to apply the benchmark's speed cap) and
        sets the CAVs up as the benchmark's MDPVehicles."""
        self.env = env
        scene = self.scene or ("merge" if "merge" in type(env).__name__.lower() else "intersection")
        self.cfg = SCENES[scene]
        self.geo = RouteGeometry(env.road.network)
        self._routes: dict[int, tuple] = {}
        for v in env.road.vehicles:  # full planned route (the fork never shortens it)
            self._route_of(v)
        if self.vehicle == "benchmark":
            for v in env.controlled_vehicles:
                v.target_speeds = np.array(CAV_TARGET_SPEEDS)
                v.speed_index = int(v.speed_to_index(BENCH_LANE_SPEED_LIMIT))
                v.target_speed = float(v.index_to_speed(v.speed_index))
                v.ACC_LIMITS = (-np.inf, np.inf)  # instance attribute: no acceleration clip

    def _route_of(self, v) -> tuple:
        vid = self.env.vid(v)
        if vid not in self._routes:
            route = list(getattr(v, "route", None) or [v.lane_index])
            self._routes[vid] = tuple((a, b) for a, b, *_ in route)
        return self._routes[vid]

    # -- one decision step -------------------------------------------------
    def _bench_vehicles(self, obs) -> dict[int, BenchVehicle]:
        env = self.env
        out = {}
        for v in env.road.vehicles:
            vid = env.vid(v)
            st = obs.states.get(vid)
            if st is None:
                continue
            key = self._route_of(v)
            pts = self.geo.points(key)
            last = env.road.network.get_lane((key[-1][0], key[-1][1], 0))
            is_cav = v in env.controlled_vehicles
            out[vid] = BenchVehicle(
                vid, is_cav, np.array(st.position, dtype=float), float(st.speed), tuple(v.lane_index),
                key, pts, last.position(last.length, 0),
                speed_index=int(getattr(v, "speed_index", 0)) if is_cav else None,
                speed_count=len(getattr(v, "target_speeds", CAV_TARGET_SPEEDS)) if is_cav else None)
        return out

    def _ask(self, prompt: str, context: str) -> str:
        try:
            if isinstance(self._llm, ReproLLM):
                return self._llm(prompt, context)
            return self._llm(prompt)
        except RuntimeError:
            raise
        except Exception as exc:  # API still failing after the retries
            self.llm_failures += 1
            return f"<LLM error: {type(exc).__name__}>"

    def act(self, obs, all_cavs):
        if self.env is None:
            raise RuntimeError("codrivingllm_repro needs attach(env) after env.reset(); "
                               "cavbench.runner.run_episode does this.")
        t0 = time.perf_counter()
        env, cfg = self.env, self.cfg
        self._step += 1
        ctx = f"{self._seed}|{self._step}"
        bv = self._bench_vehicles(obs)
        vmap = {env.vid(v): v for v in env.road.vehicles}
        cav_order = [vid for vid in env.cav_vids() if vid in bv]
        negotiators = [bv[v] for v in cav_order] if cfg["cav_only"] else list(bv.values())

        # 1) negotiation, every step
        conflict = detect_conflicts(self.geo, negotiators, cfg["reaction_range"])
        conflicting_info = conflicting_pairs(negotiators, conflict)
        self.n_negotiation += 1
        neg_reply = self._ask(negotiation_prompt(format_conflicts(conflicting_info)), ctx + "|negotiation")
        self.log.append({"time": obs.time, "mode": "negotiation",
                         "group": sorted({g[k].vid for g in conflicting_info for k in ("vehicle_i", "vehicle_j")}),
                         "decisions": [(f, s) for f, s in _PAIR_RE.findall(neg_reply)],
                         "reply": neg_reply[:300]})

        # 2) one prompt per CAV, in order; the speed cap is applied to each CAV
        #    just before its prompt is built (llm_controller_run)
        vehicles = list(bv.values())
        prompts = {}
        infos = {}
        for vid in cav_order:
            if vid not in all_cavs:
                continue
            ego = bv[vid]
            cap = cfg["speed_cap"]
            if self.speed_cap and vmap[vid].speed > cap:
                vmap[vid].speed = cap  # the benchmark sets the vehicle's speed
            if self.speed_cap and ego.speed > cap:
                ego.speed = cap
            limit = cfg["ttcp_speed_limit"] or float(getattr(vmap[vid].lane, "speed_limit", 10.0) or 10.0)
            neg_results = negotiation_results_for(ego, neg_reply)
            info = prompt_engineer(ego, vehicles, env.road.network, neg_results, conflicting_info, limit)
            infos[vid] = (info, neg_results)
        if self.memory_mode == "use":  # embed all queries of this step in one request
            self.memory.prefetch([memory_query(info) for info, _ in infos.values()])
        for vid, (info, neg_results) in infos.items():
            past = relative_memory(self.memory, info) if self.memory_mode == "use" else ""
            prompts[vid] = decision_prompt(info, neg_results, past)

        # 3) the decision calls
        vids = list(prompts)
        if self.parallel and len(vids) > 1:
            with ThreadPoolExecutor(max_workers=len(vids)) as ex:
                replies = list(ex.map(lambda v: self._ask(prompts[v], f"{ctx}|decision|{v}"), vids))
        else:
            replies = [self._ask(prompts[v], f"{ctx}|decision|{v}") for v in vids]
        actions = {vid: IDLE for vid in all_cavs}
        for vid, reply in zip(vids, replies):
            self.n_decision += 1
            name = extract_decision(reply)
            aid = action_id(name)
            if self.memory_mode == "build":  # line 46 of llm_agent_action.py
                memory_update(self.memory, infos[vid][0], np.array([aid]))
            if aid == -1:  # the benchmark would stop with an error here
                self.unusable += 1
                actions[vid] = IDLE
            else:
                actions[vid] = TO_CAVBENCH[aid]
            self.log.append({"time": obs.time, "mode": "decision", "vid": vid,
                             "action": ACTION_NAMES[actions[vid]], "reply": reply[:300]})
        self.decision_times.append((time.perf_counter() - t0) * 1000.0)
        if self.safety is not None:
            actions = self.safety.apply(obs, actions)
        return actions

    # -- bookkeeping ---------------------------------------------------------
    def extra_metrics(self) -> dict:
        m = {"llm_parse_failures": self.unusable, "llm_failures": self.llm_failures,
             "llm_calls_negotiation": self.n_negotiation, "llm_calls_decision": self.n_decision}
        c = self._llm
        if isinstance(c, ReproLLM):
            base = getattr(self, "_usage0", None) or (0, 0, 0, 0, 0.0)
            now = (c.calls, c.cache_hits, c.tokens_in, c.tokens_out, c.cost_usd)
            d = [b - a for a, b in zip(base, now)]
            m.update({"llm_calls": d[0], "llm_cache_hits": d[1], "llm_tokens_in": d[2],
                      "llm_tokens_out": d[3], "llm_cost_usd": round(d[4], 6)})
        if self.memory is not None:
            mem = self.memory
            m0 = getattr(self, "_mem0", (0, 0, 0.0))
            self.memory.flush()  # build mode: this episode's decisions are stored now
            m.update({"llm_calls_embedding": mem.embed_calls - m0[0], "memory_items": len(mem),
                      "memory_hits": mem.hits - m0[1]})
            if "llm_cost_usd" in m:
                m["llm_cost_usd"] = round(m["llm_cost_usd"] + mem.cost_usd - m0[2], 6)
        return m
