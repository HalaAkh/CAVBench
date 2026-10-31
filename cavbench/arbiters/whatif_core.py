"""What-if (counterfactual) evaluation of candidate joint maneuvers (Steps 3-5).

For a conflict group C = {CAV_1, ..., CAV_K}, option m_k means "CAV k goes,
every other CAV in C yields". Each option is simulated N times over a horizon
T. Rollout n starts from the observed (noisy) shared state plus an imagined
error eta^(n) drawn from the assumed noise model:

    x^(n) = s_hat + eta^(n),   eta^(n) ~ N(0, sigma^2)

Vehicles move along their planned routes (1-D, arc length). In each rollout:

* the vehicle that goes accelerates towards the top target speed (<= a_go);
* yielding CAVs brake towards a stop (a_yield);
* other CAVs keep their current target speed;
* human-driven vehicles follow IDM (Treiber et al., 2000);
* every vehicle also follows (IDM interaction term) a vehicle that is already
  on a shared stretch of lane ahead of it, with the same rule as the
  low-level safety layer in the simulator; nobody brakes for crossing
  traffic, so who gives way is decided by the option being tested.

For every pair of vehicles whose routes meet (crossing, merging or same lane)
and that includes a member of C, the clearance between the two vehicle
footprints is tracked; each 5 m x 2 m car is approximated by three circles of
radius 1 m along its heading (a standard motion-planning approximation, which
unlike centre distance does not flag cars passing in adjacent lanes). The
smallest clearance d_min^(n) over the horizon gives

    R_hat_k = (1/N) sum_n 1[d_min^(n) < d_safe],   U_k = std(d_min) / 5 m,

with d_safe = 2 m (about one car width of clearance).

All K options and N rollouts are simulated at once as NumPy arrays. The same
imagined errors are reused across options (common random numbers), which
makes the comparison between options fairer; each option's N rollouts remain
independent, which is what Theorem 1 requires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conflicts import ConflictGroup
from ..geometry import route_relation
from .commitment import platoon_ahead
from ..state import SceneObservation

GO, YIELD, FREE_CAV, HDV = 0, 1, 2, 3


@dataclass
class RolloutConfig:
    n_rollouts: int = 100  # N, from Theorem 1 (N >= 98 for K=4, gamma=0.15, delta=0.1)
    horizon: float = 4.0  # T [s]
    dt: float = 0.1  # [s] -> L = 40 steps
    a_go: float = 2.0  # [m/s^2] acceleration of the vehicle that goes
    a_yield: float = -3.0  # [m/s^2] braking of yielding vehicles
    a_min: float = -6.0  # [m/s^2] physical braking limit
    kp: float = 1 / 0.6  # speed-tracking gain (HighwayEnv's KP_A)
    d_safe: float = 2.0  # [m] clearance between footprints (about one car width)
    u_scale: float = 5.0  # [m] U = std(d_min) / u_scale (one car length)
    vehicle_length: float = 5.0
    circle_offsets: tuple = (-1.5, 0.0, 1.5)  # [m] 3 circles along the heading ...
    circle_radius: float = 1.0  # [m] ... of radius 1 m cover a 5 m x 2 m car
    idm_a: float = 3.0
    idm_b: float = 5.0
    idm_T: float = 1.5
    idm_s0: float = 2.0
    idm_delta: float = 4.0
    hdv_desired_speed: float = 8.0
    leader_range: float = 30.0
    context_radius: float = 60.0  # vehicles this close to a member are simulated too
    common_random_numbers: bool = True

    @property
    def steps(self) -> int:
        return int(round(self.horizon / self.dt))


@dataclass
class OptionResult:
    goer: int | None  # CAV that goes first in this option (None: everyone yields)
    risk: float  # R_hat_k
    uncertainty: float  # U_k
    d_min: np.ndarray  # (N,) minimum distance per rollout


GRID_DS = 0.25  # [m] resolution of the route lookup tables
_TABLE_CACHE: dict[tuple, np.ndarray] = {}


def _route_table(paths) -> np.ndarray:
    """Stack, for each route, positions and tangents every GRID_DS metres.

    Shape (M, G, 4) with columns x, y, tx, ty; shorter routes are padded with
    their last row, which reproduces np.interp's clamping at the route end.
    """
    rows = []
    for p in paths:
        if p.key not in _TABLE_CACHE or len(_TABLE_CACHE[p.key]) != int(np.ceil(p.length / GRID_DS)) + 1:
            sg = np.arange(int(np.ceil(p.length / GRID_DS)) + 1) * GRID_DS
            xy = p.position(sg)
            t = np.stack([np.interp(sg, p.s, p.tangents[:, 0]),
                          np.interp(sg, p.s, p.tangents[:, 1])], axis=-1)
            _TABLE_CACHE[p.key] = np.concatenate([xy, t], axis=-1)
        rows.append(_TABLE_CACHE[p.key])
    g = max(len(r) for r in rows) + 1
    out = np.empty((len(rows), g, 4))
    for m, r in enumerate(rows):
        out[m, :len(r)] = r
        out[m, len(r):] = r[-1]
    return out


def _involved(obs: SceneObservation, group: ConflictGroup, radius: float) -> list[int]:
    ids = list(group.cavs) + [v for v in group.others if v not in group.cavs]
    centres = np.array([obs.states[v].position for v in group.cavs])
    for vid, st in obs.states.items():
        if vid in ids:
            continue
        if np.min(np.linalg.norm(centres - st.position[None, :], axis=1)) < radius:
            ids.append(vid)
    return ids


def evaluate_options(
    obs: SceneObservation, group: ConflictGroup, cfg: RolloutConfig, rng: np.random.Generator,
    include_fallback: bool = False,
) -> list[OptionResult]:
    """Simulate every option m_k (k goes, the others yield).

    With ``include_fallback`` an extra last option is added in which every
    member yields (the fallback); its ``goer`` is None.
    """
    ids = _involved(obs, group, cfg.context_radius)
    states = [obs.states[v] for v in ids]
    n_opt = len(group.cavs)
    K, N, M = n_opt + int(include_fallback), cfg.n_rollouts, len(ids)
    paths = [st.path for st in states]
    path_len = np.array([p.length for p in paths])

    # Roles of each vehicle under each option.
    roles = np.full((K, M), HDV)
    v_target = np.zeros((K, M))
    member = np.zeros(M, dtype=bool)
    for m, st in enumerate(states):
        if st.is_cav and st.vid in group.cavs:
            member[m] = True
            roles[:, m] = YIELD
        elif st.is_cav:
            roles[:, m] = FREE_CAV
            v_target[:, m] = st.target_speed if st.target_speed is not None else st.speed
        else:
            v_target[:, m] = cfg.hdv_desired_speed
    for k, cav in enumerate(group.cavs):
        roles[k, ids.index(cav)] = GO
        for ahead in platoon_ahead(obs, group, cav):  # vehicles queued ahead go too
            roles[k, ids.index(ahead)] = GO

    # Initial states with imagined errors eta.
    shape = (1 if cfg.common_random_numbers else K, N, M)
    eta_s = rng.normal(0.0, obs.sigma_pos, size=shape)
    eta_v = rng.normal(0.0, obs.sigma_vel, size=shape)
    s0 = np.array([st.s for st in states])
    v0 = np.array([st.speed for st in states])
    s = np.broadcast_to(s0 + eta_s, (K, N, M)).copy()
    v = np.maximum(np.broadcast_to(v0 + eta_v, (K, N, M)), 0.0).copy()

    # Only vehicles whose routes meet (crossing, merging or following on the
    # same lane) can collide; parallel lanes 4 m apart must not count.
    # Leaders are only vehicles on a shared stretch of route: like HighwayEnv's
    # drivers, nobody in a rollout brakes for crossing traffic; who gives way
    # at a crossing is decided by the option being tested.
    relevant = np.zeros((M, M), dtype=bool)
    shared = np.zeros((M, M), dtype=bool)
    sh_start_i = np.zeros((M, M))  # start of the shared stretch on i's route
    sh_start_j = np.zeros((M, M))  # ... and on j's route
    sh_len = np.zeros((M, M))
    for a in range(M):
        for b in range(M):
            if a == b:
                continue
            meet, sh = route_relation(paths[a], paths[b])
            relevant[a, b] = meet
            if sh is not None:
                shared[a, b] = True
                sh_start_i[a, b], sh_start_j[a, b], sh_len[a, b] = sh.s_i, sh.s_j, sh.length
    relevant = relevant | relevant.T
    pair_mask = np.triu((member[:, None] | member[None, :]) & relevant)
    ia, ib = np.nonzero(pair_mask)
    offsets = np.asarray(cfg.circle_offsets)
    sqrt_ab = 2.0 * np.sqrt(cfg.idm_a * cfg.idm_b)
    d_min = np.full((K, N), np.inf)
    roles_b = roles[:, None, :]
    vt_b = v_target[:, None, :]
    v_max = obs.v_max

    # Directed pairs (follower fi, candidate leader lj) on a shared stretch,
    # grouped by follower so that per-follower minima use np.minimum.reduceat.
    fi, lj = np.nonzero(shared)
    order = np.argsort(fi, kind="stable")
    fi, lj = fi[order], lj[order]
    p_si, p_sj, p_len = sh_start_i[fi, lj], sh_start_j[fi, lj], sh_len[fi, lj]
    seg = np.flatnonzero(np.r_[True, fi[1:] != fi[:-1]]) if len(fi) else np.array([], int)
    followers = fi[seg]

    # Route lookup tables: position and tangent every GRID_DS metres.
    table = _route_table(paths)  # (M, G, 4): x, y, tx, ty
    g_max = table.shape[1] - 1
    m_idx = np.arange(M)[None, None, :]
    interaction = np.zeros((K, N, M))
    has_leader = np.zeros((K, N, M), dtype=bool)

    for step in range(cfg.steps + 1):
        u = np.clip(s / GRID_DS, 0.0, g_max - 1e-9)
        i0 = u.astype(np.int64)
        w = (u - i0)[..., None]
        xyt = table[m_idx, i0] * (1.0 - w) + table[m_idx, i0 + 1] * w
        x, y, tx, ty = xyt[..., 0], xyt[..., 1], xyt[..., 2], xyt[..., 3]
        active = s < path_len  # vehicles leave the scene at the end of their route

        if len(ia):  # footprint clearance of the relevant pairs (three-circle model)
            cx_a = x[..., ia, None] + tx[..., ia, None] * offsets
            cy_a = y[..., ia, None] + ty[..., ia, None] * offsets
            cx_b = x[..., ib, None] + tx[..., ib, None] * offsets
            cy_b = y[..., ib, None] + ty[..., ib, None] * offsets
            cd = np.hypot(cx_a[..., :, None] - cx_b[..., None, :],
                          cy_a[..., :, None] - cy_b[..., None, :])
            clearance = cd.min(axis=(-1, -2)) - 2.0 * cfg.circle_radius
            clearance = np.where(active[..., ia] & active[..., ib], clearance, np.inf)
            d_min = np.minimum(d_min, clearance.min(axis=-1))
        if step == cfg.steps:
            break

        # Leaders: vehicles already on a shared stretch of lane and further
        # along it than the follower (see geometry.is_leader); IDM interaction.
        if len(fi):
            dp = np.hypot(x[..., lj] - x[..., fi], y[..., lj] - y[..., fi])
            pos_i = s[..., fi] - p_si
            pos_j = s[..., lj] - p_sj
            lead = (pos_j >= 0) & (pos_j <= p_len) & (pos_j > pos_i)
            lead &= (dp < cfg.leader_range) & active[..., fi] & active[..., lj]
            gap = np.maximum(dp - cfg.vehicle_length, 0.1)
            vi = v[..., fi]
            v_j_along = v[..., lj] * (tx[..., fi] * tx[..., lj] + ty[..., fi] * ty[..., lj])
            s_star = cfg.idm_s0 + np.maximum(0.0, vi * cfg.idm_T + vi * (vi - v_j_along) / sqrt_ab)
            term = np.where(lead, -cfg.idm_a * (s_star / gap) ** 2, 0.0)
            interaction[..., followers] = np.minimum.reduceat(term, seg, axis=-1)
            has_leader[..., followers] = np.logical_or.reduceat(lead, seg, axis=-1)

        # Commanded acceleration by role.
        a_go = np.minimum(cfg.kp * (v_max - v), cfg.a_go)
        a_yield = np.maximum(cfg.kp * (0.0 - v), cfg.a_yield)
        a_free = np.clip(cfg.kp * (vt_b - v), cfg.a_yield, cfg.a_go)
        a_hdv = cfg.idm_a * (1.0 - (v / np.maximum(vt_b, 0.1)) ** cfg.idm_delta) + interaction
        a_cmd = np.select(
            [roles_b == GO, roles_b == YIELD, roles_b == FREE_CAV], [a_go, a_yield, a_free], a_hdv
        )
        a_follow = np.where(has_leader, cfg.idm_a + interaction, np.inf)
        a = np.where(roles_b == HDV, a_cmd, np.minimum(a_cmd, a_follow))
        a = np.maximum(a, cfg.a_min)
        v = np.maximum(v + a * cfg.dt, 0.0)
        s = s + v * cfg.dt

    d_min = np.where(np.isfinite(d_min), d_min, 100.0)
    out = []
    goers = list(group.cavs) + ([None] if include_fallback else [])
    for k, cav in enumerate(goers):
        dk = d_min[k]
        out.append(OptionResult(cav, float(np.mean(dk < cfg.d_safe)),
                                float(np.std(dk) / cfg.u_scale), dk))
    return out
