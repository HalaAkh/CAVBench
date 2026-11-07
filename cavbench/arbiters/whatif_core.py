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
from ..geometry import meeting_points, route_relation
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
    # Uncertainty floor (change 1)
    sigma_pos_min: float = 0.5  # [m]
    sigma_vel_min: float = 0.3  # [m/s]
    hdv_speed_sd: float = 1.0  # [m/s] spread of human drivers' desired speed
    # Danger = clearance OR time gap at the conflict area (change 2)
    pet_safe: float = 1.5  # [s]; 0 disables the PET criterion
    zone_radius: float = 3.0  # [m] conflict area, as in the PET metric
    # Human drivers who give way (change 3)
    p_hdv_yield: float = 0.5  # probability that a human driver follows the priority rule
    yield_check: float = 0.5  # [s] how often the rule is re-checked
    yield_horizon: float = 3.0  # [s] constant-speed prediction horizon of the rule
    yield_distance: float = 5.0  # [m] predicted centre distance counted as a conflict
    # Speed-up (change 4): skip pairs that cannot meet within the horizon
    prune_pairs: bool = True
    prune_margin: float = 8.0  # [m]
    exact_range: float = 12.0  # [m] exact footprint clearance below this centre distance
    # V2X message age: older messages are more uncertain
    age_aware: bool = True
    age_acc: float = 3.0  # [m/s^2] unknown acceleration since the message was sent

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

    Shape (M, G, 5) with columns x, y, tx, ty, lane priority; shorter routes are
    padded with their last row, which reproduces np.interp's clamping at the
    route end.
    """
    rows = []
    for p in paths:
        if p.key not in _TABLE_CACHE or len(_TABLE_CACHE[p.key]) != int(np.ceil(p.length / GRID_DS)) + 1:
            sg = np.arange(int(np.ceil(p.length / GRID_DS)) + 1) * GRID_DS
            xy = p.position(sg)
            t = np.stack([np.interp(sg, p.s, p.tangents[:, 0]),
                          np.interp(sg, p.s, p.tangents[:, 1])], axis=-1)
            _TABLE_CACHE[p.key] = np.concatenate([xy, t, p.priority_at(sg)[:, None]], axis=-1)
        rows.append(_TABLE_CACHE[p.key])
    g = max(len(r) for r in rows) + 1
    out = np.empty((len(rows), g, 5), dtype=np.float32)
    for m, r in enumerate(rows):
        out[m, :len(r)] = r
        out[m, len(r):] = r[-1]
    return out


_ZONE_CACHE: dict[tuple, tuple | None] = {}


def _conflict_zone(pa, pb, radius: float):
    """Arc-length intervals [enter, exit] on routes a and b of the conflict
    area: a disk of ``radius`` around the first point where the routes meet
    (the same area as the PET metric). None if they do not meet, or if they
    start on the same lane (car following, not a conflict)."""
    key = (pa.key, pb.key, radius)
    if key not in _ZONE_CACHE:
        out = None
        if pa.key[:1] != pb.key[:1]:
            mps = meeting_points(pa, pb, 1.0)
            if mps:
                point = pa.position(mps[0].s_i)
                out = (_interval(pa, point, radius), _interval(pb, point, radius))
                if out[0] is None or out[1] is None:
                    out = None
        _ZONE_CACHE[key] = out
    return _ZONE_CACHE[key]


def _interval(path, point, radius):
    d = np.hypot(*(path.points - point[None, :]).T)
    inside = d < radius
    if not inside.any():
        return None
    i = int(np.argmin(d))
    a = i
    while a > 0 and inside[a - 1]:
        a -= 1
    b = i
    while b < len(d) - 1 and inside[b + 1]:
        b += 1
    return float(path.s[a]), float(path.s[b])


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
    sig_p = max(obs.sigma_pos, cfg.sigma_pos_min)
    sig_v = max(obs.sigma_vel, cfg.sigma_vel_min)
    # Per-vehicle spread: a message that is `age` seconds old (V2X delay or
    # packet loss) was moved forward by dead reckoning, which assumes a
    # constant speed; the speed error and an unknown acceleration add
    #   sigma_s(age)^2 = sigma^2 + (sigma_v age)^2 + (a age^2 / 2)^2,
    #   sigma_v(age)^2 = sigma_v^2 + (a age)^2.
    age = np.zeros(M)
    if cfg.age_aware:
        ages = obs.extra.get("v2x_age", {}) if obs.extra else {}
        age = np.array([float(ages.get(st.vid, 0.0)) for st in states])
    sig_p_m = np.sqrt(sig_p ** 2 + (sig_v * age) ** 2 + (0.5 * cfg.age_acc * age ** 2) ** 2)
    sig_v_m = np.sqrt(sig_v ** 2 + (cfg.age_acc * age) ** 2)
    eta_s = rng.normal(0.0, 1.0, size=shape) * sig_p_m
    eta_v = rng.normal(0.0, 1.0, size=shape) * sig_v_m
    is_hdv = roles[0] == HDV
    # Human drivers: own desired speed per rollout; do they follow the priority rule?
    hdv_speed = v_target[0][None, None, :] + cfg.hdv_speed_sd * rng.normal(size=(1, N, M)) * is_hdv
    v_target_b = np.where(is_hdv[None, None, :], np.maximum(hdv_speed, 2.0),
                          v_target[:, None, :])  # (K or 1, N, M)
    complies = (rng.random(size=(1, N, M)) < cfg.p_hdv_yield) & is_hdv
    s0 = np.array([st.s for st in states])
    v0 = np.array([st.speed for st in states])
    s = np.broadcast_to(s0 + eta_s, (K, N, M)).astype(np.float32)
    v = np.maximum(np.broadcast_to(v0 + eta_v, (K, N, M)), 0.0).astype(np.float32)

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
    # Speed-up: drop crossing pairs that cannot meet within the horizon (one of
    # them is too far from the conflict area to reach it, or left it long ago).
    if len(ia) and cfg.prune_pairs:
        v_lim = max(obs.v_max, cfg.hdv_desired_speed + 3 * cfg.hdv_speed_sd)
        reach = np.maximum(v0, v_lim) * cfg.horizon + 3 * sig_p_m + cfg.prune_margin
        past = np.maximum(v0, 1.0) * max(cfg.pet_safe, 1.0) + 3 * sig_p_m + cfg.prune_margin
        keep = np.ones(len(ia), dtype=bool)
        for n, (a, b) in enumerate(zip(ia, ib)):
            if shared[a, b] or shared[b, a]:
                continue  # same lane / merging: can meet anywhere along it
            z = _conflict_zone(paths[a], paths[b], cfg.zone_radius)
            if z is None:
                continue
            (ea, xa), (eb, xb) = z
            if (ea - s0[a] > reach[a] or eb - s0[b] > reach[b]
                    or s0[a] - xa > past[a] or s0[b] - xb > past[b]):
                keep[n] = False
        ia, ib = ia[keep], ib[keep]
    offsets = np.asarray(cfg.circle_offsets, dtype=np.float32)
    sqrt_ab = 2.0 * np.sqrt(cfg.idm_a * cfg.idm_b)
    d_min = np.full((K, N), np.inf)
    roles_b = roles[:, None, :]
    is_go, is_yield, is_free = roles_b == GO, roles_b == YIELD, roles_b == FREE_CAV
    vt_b = np.broadcast_to(v_target_b, (K, N, M))
    v_max = obs.v_max

    # Conflict areas of the pairs, for the PET criterion.
    use_pet = cfg.pet_safe > 0 and len(ia) > 0
    if use_pet:
        zones = [_conflict_zone(paths[a], paths[b], cfg.zone_radius) for a, b in zip(ia, ib)]
        zp = np.array([z is not None for z in zones])
        za, zb = ia[zp], ib[zp]
        use_pet = len(za) > 0
    if use_pet:
        zones = [z for z in zones if z is not None]
        enter_a = np.array([z[0][0] for z in zones]); exit_a = np.array([z[0][1] for z in zones])
        enter_b = np.array([z[1][0] for z in zones]); exit_b = np.array([z[1][1] for z in zones])

        def crossing_times(s_now, v_now, enter, exit_):
            """Initial (possibly past, negative) entry and exit times."""
            vv = np.maximum(v_now, 1.0)
            t_in = np.where(s_now >= enter, -(s_now - enter) / vv, np.nan)
            t_out = np.where(s_now >= exit_, -(s_now - exit_) / vv, np.nan)
            return t_in, t_out

        tin_a, tout_a = crossing_times(s[..., za], v[..., za], enter_a, exit_a)
        tin_b, tout_b = crossing_times(s[..., zb], v[..., zb], enter_b, exit_b)

    # Pairs for the human drivers' give-way rule: a human driver h and any
    # vehicle o whose route meets h's.
    use_yield = cfg.p_hdv_yield > 0 and is_hdv.any()
    if use_yield:
        yh, yo = np.nonzero(is_hdv[:, None] & relevant & ~np.eye(M, dtype=bool))
        use_yield = len(yh) > 0
        check_every = max(int(round(cfg.yield_check / cfg.dt)), 1)
        t_pred = np.arange(0.25, cfg.yield_horizon, 0.25)
        yielding = np.zeros((K, N, M), dtype=bool)

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
            # Centre distance first; the exact three-circle clearance is only
            # computed for pairs that are within `exact_range` in some rollout
            # (farther apart, clearance >= centre distance - 5 m > d_safe, and
            # the middle-circle distance is used).
            both = active[..., ia] & active[..., ib]
            dcx, dcy = x[..., ia] - x[..., ib], y[..., ia] - y[..., ib]
            dc = np.sqrt(dcx * dcx + dcy * dcy)
            clearance = dc - 2.0 * cfg.circle_radius
            near = ((dc < cfg.exact_range) & both).any(axis=(0, 1))
            if near.any():
                na, nb = ia[near], ib[near]
                cx_a = x[..., na, None] + tx[..., na, None] * offsets
                cy_a = y[..., na, None] + ty[..., na, None] * offsets
                cx_b = x[..., nb, None] + tx[..., nb, None] * offsets
                cy_b = y[..., nb, None] + ty[..., nb, None] * offsets
                ddx = cx_a[..., :, None] - cx_b[..., None, :]
                ddy = cy_a[..., :, None] - cy_b[..., None, :]
                cd2 = (ddx * ddx + ddy * ddy).reshape(ddx.shape[:-2] + (-1,)).min(axis=-1)
                clearance[..., near] = np.sqrt(cd2) - 2.0 * cfg.circle_radius
            clearance = np.where(both, clearance, np.inf)
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

        # Human drivers' give-way rule (re-checked every yield_check seconds).
        if use_yield and step % check_every == 0:
            sp = s[..., :, None] + v[..., :, None] * t_pred  # (K, N, M, P) constant speed
            up = np.clip(sp / GRID_DS, 0.0, g_max - 1e-9).astype(np.int64)
            px = table[np.arange(M)[None, None, :, None], up, 0]
            py = table[np.arange(M)[None, None, :, None], up, 1]
            close = (np.hypot(px[..., yh, :] - px[..., yo, :], py[..., yh, :] - py[..., yo, :])
                     < cfg.yield_distance).any(axis=-1)
            prio = table[m_idx, i0, 4]
            ph, po = prio[..., yh], prio[..., yo]
            # equal priority: the one behind (in the other's frame) gives way
            dx, dy = x[..., yo] - x[..., yh], y[..., yo] - y[..., yh]
            ahead_of_h = dx * tx[..., yh] + dy * ty[..., yh]  # o's distance in front of h
            ahead_of_o = -(dx * tx[..., yo] + dy * ty[..., yo])
            h_gives_way = (ph < po) | ((ph == po) & (ahead_of_h > ahead_of_o))
            must = close & h_gives_way & active[..., yh] & active[..., yo]
            yielding = np.zeros((K, N, M), dtype=bool)
            np.logical_or.at(yielding, (slice(None), slice(None), yh), must)
            yielding &= complies

        # Commanded acceleration by role.
        a_go = np.minimum(cfg.kp * (v_max - v), cfg.a_go)
        a_yield = np.maximum(cfg.kp * (0.0 - v), cfg.a_yield)
        a_free = np.clip(cfg.kp * (vt_b - v), cfg.a_yield, cfg.a_go)
        vt_h = np.where(yielding, 0.01, vt_b) if use_yield else vt_b
        a_hdv = cfg.idm_a * (1.0 - (v / np.maximum(vt_h, 0.01)) ** cfg.idm_delta) + interaction
        a_cmd = np.where(is_go, a_go, np.where(is_yield, a_yield, np.where(is_free, a_free, a_hdv)))
        a_follow = np.where(has_leader, cfg.idm_a + interaction, np.inf)
        a = np.where(roles_b == HDV, a_cmd, np.minimum(a_cmd, a_follow))
        a = np.maximum(a, cfg.a_min)
        v = np.maximum(v + a * cfg.dt, 0.0)
        s = s + v * cfg.dt
        if use_pet:
            t_now = (step + 1) * cfg.dt
            sa, sb = s[..., za], s[..., zb]
            tin_a = np.where(np.isnan(tin_a) & (sa >= enter_a), t_now, tin_a)
            tout_a = np.where(np.isnan(tout_a) & (sa >= exit_a), t_now, tout_a)
            tin_b = np.where(np.isnan(tin_b) & (sb >= enter_b), t_now, tin_b)
            tout_b = np.where(np.isnan(tout_b) & (sb >= exit_b), t_now, tout_b)

    d_min = np.where(np.isfinite(d_min), d_min, 100.0)
    danger = d_min < cfg.d_safe
    if use_pet:
        # PET when both vehicles enter the area and the second enters in the
        # future (past events cannot be changed by the decision).
        a_first = np.where(np.isnan(tin_b), True, tin_a <= tin_b)
        t_in2 = np.where(a_first, tin_b, tin_a)
        t_out1 = np.where(a_first, tout_a, tout_b)
        both = ~np.isnan(tin_a) & ~np.isnan(tin_b) & (t_in2 >= 0)
        pet = np.where(np.isnan(t_out1), 0.0, t_in2 - t_out1)  # first still inside: overlap
        danger |= (both & (pet < cfg.pet_safe)).any(axis=-1)
    out = []
    goers = list(group.cavs) + ([None] if include_fallback else [])
    for k, cav in enumerate(goers):
        dk = d_min[k]
        out.append(OptionResult(cav, float(np.mean(danger[k])),
                                float(np.std(dk) / cfg.u_scale), dk))
    return out
