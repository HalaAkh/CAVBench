"""Safety-critical multi-agent MCTS baseline (Lin et al., 2025), re-implemented.

Z. Lin, J. Lan, C. Anagnostopoulos, Z. Tian, D. Flynn, "Safety-Critical
Multi-Agent MCTS for Mixed Traffic Coordination at Unsignalized Intersections",
IEEE Trans. Intelligent Transportation Systems, 26(11):18992-19005, 2025.

The authors' MATLAB code is not public, so this file follows the paper
(equation numbers in the comments). Every value or adaptation that the paper
does not fix is marked "our choice" and listed in docs/mcts_baseline.md.

For each conflict group found by the shared ConflictDetector, one centralised
tree search runs over the joint actions of the group's CAVs (one of SLOWER /
IDLE / FASTER each: 3^K joint actions):

* tree step 0.5 s, maximum depth 8 (4 s), 1000 iterations (Algorithm 1);
* children are created all at once and marked safe "o" / unsafe "x" by the
  V2V / V2H / V2R checks (Eq. 22, 24); only "o" children are selected (UCB,
  Eq. 25);
* rollouts continue to depth d_max with random safe joint actions and return
  the discounted multi-objective reward (Eq. 27, 29-35);
* values are backed up with Eq. 28; the action is the root child that
  maximises N + eps * Q (Eq. 41).

Vehicles inside the tree move 1-D along their planned routes (arc length):
the group's CAVs track the target speed set by their action, other CAVs keep
their current target speed, and human-driven vehicles (HDVs) follow IDM with
the paper's driving-style scaling (Eq. 14). The observed (noisy) states are
taken as exact, as in the paper.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, fields

import numpy as np

from ..conflicts import ConflictGroup
from ..geometry import meeting_points, route_relation
from ..state import SceneObservation
from .base import FASTER, IDLE, SLOWER, Arbiter
from .whatif_core import GRID_DS, RolloutConfig, _involved, _route_table


@dataclass
class MCTSConfig:
    # -- tree search (Algorithm 1, Table I) --
    n_iterations: int = 1000  # K (Table I)
    d_max: int = 8  # maximum search depth (Table I)
    dt_tree: float = 0.5  # [s] duration of one tree step (our choice: not in the paper)
    dt_sim: float = 0.1  # [s] integration step inside a tree step (our choice)
    c_exp: float = 1.0  # UCB exploration constant (Table I)
    gamma_r: float = 0.95  # reward discount (Table I)
    beta_dec: float = 0.75  # value-update decay (Table I)
    eps: float = 0.5  # best-child weight, Eq. 41 (Table I's "policy balance factor" eta = 0.5)
    alpha_model: float = 0.0  # weight of the learned rollout policy, Eq. 26 (our choice: no predictor)
    reuse_tree: bool = False  # Eq. 43 tree reuse (our choice: rebuilt every 0.2 s)
    rollout_first: int = 1  # first batch of joint actions safety-checked in a rollout step (speed only)
    rollout_chunk: int = 64  # largest batch (speed only)
    # -- motion model inside the tree (our choices, from the simulator) --
    kp: float = 1 / 0.6  # speed-tracking gain (HighwayEnv's KP_A)
    cav_acc_limits: tuple = (-5.0, 3.0)  # [m/s^2] CAV acceleration bounds (CAVBench env)
    context_radius: float = 60.0  # [m] vehicles this close to a group CAV are simulated
    leader_range: float = 30.0  # [m] same leader rule as the what-if rollouts
    # -- dynamic safe distance, Eq. 7-8 --
    d_base: float = 1.5  # [m] (Table I)
    kappa_v: float = 0.3  # (Table I)
    beta_1: float = 0.2  # (Table I)
    beta_2: float = 0.2  # (Table I)
    v_ref: float | None = None  # [m/s] (our choice: None = top CAV target speed)
    zone_center: tuple | None = None  # conflict-zone centre (our choice: from the scenario)
    zone_r_ex: float | None = None  # [m] R_EX, half-length of the zone (our choice: 11 m)
    # -- risk, Eq. 9-12 and 18-19 --
    lambda_v: float = 0.3  # (Table I)
    tp_steps: int = 5  # T_p, prediction horizon 5.0 (Table I)
    tp_dt: float = 1.0  # [s] spacing of the T_p prediction points (our choice)
    w1_v2v: float = 0.6  # (Table I)
    w2_v2v: float = 0.4  # (Table I)
    q_th_v2v: float = 0.8  # (Table I)
    w1_v2h: float = 0.4  # (Table I)
    w2_v2h: float = 0.3  # (Table I)
    w3_v2h: float = 0.3  # (Table I)
    q_th_v2h: float = 0.6  # (Table I)
    n_samples: int = 20  # Monte Carlo samples for C_ih, Eq. 18 (our choice)
    # -- HDV prediction, Eq. 13-17 (Table I) --
    eta_h: float = 0.5  # driving style of every HDV (our choice: not observable; Beta(2,2) mean)
    sigma_x: float = 0.1  # (Table I)
    sigma_y: float = 0.1  # (Table I)
    eps_x: float = 0.05  # (Table I)
    eps_y: float = 0.05  # (Table I)
    sigma_v: float = 0.2  # (Table I)
    sigma_theta: float = 0.1  # (Table I)
    rho_xv: float = 0.3  # (Table I)
    rho_ytheta: float = 0.3  # (Table I)
    beta_x: float = 0.5  # (Table I)
    beta_y: float = 0.5  # (Table I)
    beta_v: float = 0.6  # (Table I)
    beta_theta: float = 0.4  # (Table I)
    alpha_a: float = 0.3  # IDM max-acceleration style scaling (Table I)
    alpha_T: float = 0.4  # IDM time-headway style scaling (Table I)
    alpha_d: float = 0.3  # safety-distance style scaling (Table I)
    # -- reward, Eq. 29-35 --
    w_v2v: float = 1.0  # (our choice: not in the paper)
    w_v2h: float = 1.0  # (our choice: not in the paper)
    w_v2r: float = 0.0  # (our choice: V2R always satisfied on lane centres)
    lambda_T: float = 0.4  # (Table I)
    lambda_c: float = 0.3  # (Table I)
    risk_as_penalty: bool = True  # (our choice: lambda_T R_Tp and lambda_c C_ih lower the reward)
    mu_s: float = 1.0  # scenario scale of phi_v2h (our choice: not in the paper)
    k1: float = 1.0  # psi_1 constants kappa_1..4 (our choice: not in the paper)
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    e1: float = 0.2  # psi_2 constants eta_1, eta_2 (our choice: not in the paper)
    e2: float = 0.2
    v_tol: float = 2.0  # (Table I)
    alpha_v: float = 1.5  # (Table I)
    beta_v_pen: float = 1.0  # underspeeding penalty beta_v (Table I)
    w_vel: float = 0.4  # (Table I)
    w_acc: float = 0.3  # (Table I)
    w_jerk: float = 0.1  # (Table I)
    w_self_safety: float = 0.5  # w1 (Table I)
    w_self_eff: float = 0.3  # w2 (Table I)
    w_self_comfort: float = 0.2  # w3 (Table I)
    lambda_base: float = 0.5  # (Table I)
    alpha_wait: float = 0.1  # [1/s] (our choice: not in the paper)
    v_thres: float = 1.0  # [m/s] (our choice: not in the paper)


MCTS_FIELDS = {f.name for f in fields(MCTSConfig)}
INTERSECTION_R_EX = 11.0  # HighwayEnv junction: outer_distance = 9 m + 4 m / 2


# -- risk functions (Section III-B) -------------------------------------------
def dynamic_safe_distance(dv, dtheta, in_app, in_int, cfg: MCTSConfig, v_ref: float):
    """Eq. 7-8: d_safe = max(d_base, kappa_v |dv|) * alpha_1 * alpha_2 * alpha_3."""
    dv = np.abs(dv)
    a1 = 1.0 + cfg.beta_1 * dv / v_ref
    a2 = 1.0 + cfg.beta_2 * np.abs(dtheta) / np.pi
    a3 = 1.0 + np.asarray(in_app, dtype=float) + np.asarray(in_int, dtype=float)
    return np.maximum(cfg.d_base, cfg.kappa_v * dv) * a1 * a2 * a3


def instantaneous_risk(d, d_safe, dv, cfg: MCTSConfig, v_max: float):
    """Eq. 9: r = exp(-d / d_safe) * (1 + lambda_v |dv| / v_max)."""
    return np.exp(-d / d_safe) * (1.0 + cfg.lambda_v * np.abs(dv) / v_max)


def rho(d, d_safe):
    """Eq. 11: 0 at d >= d_safe, else (1 - d / d_safe)^2."""
    return np.where(d >= d_safe, 0.0, (1.0 - d / d_safe) ** 2)


def temporal_risk(d_pred, d_safe, tp_steps: int):
    """Eq. 10: R_Tp = (1/T_p) sum_{t=1}^{T_p} 1/(1+t) rho(d(t), d_safe).

    ``d_pred``: (..., T_p, P) predicted distances; ``d_safe``: (..., P)."""
    w = 1.0 / (1.0 + np.arange(1, tp_steps + 1))
    return np.einsum("t,...tp->...p", w, rho(d_pred, np.asarray(d_safe)[..., None, :])) / tp_steps


def phi_v2v(d, d_safe):
    """Eq. 30: -1 inside the safe distance, else 0."""
    return np.where(d <= d_safe, -1.0, 0.0)


def phi_v2h(d, d_safe, dv, dtheta, cfg: MCTSConfig, v_ref: float):
    """V2H penalty (Section IV-B): -mu_s psi_1 below d_th = d_safe/2,
    -mu_s psi_2 between d_th and d_safe, 0 beyond."""
    d_th = 0.5 * d_safe
    d_f = (d_safe - d) / np.maximum(d_safe - d_th, 1e-9)
    dv, dtheta = np.abs(dv), np.abs(dtheta)
    psi1 = cfg.k1 + cfg.k2 * d_f + cfg.k3 * dv + cfg.k4 * dtheta
    psi2 = d_f * (1.0 + cfg.e1 * dv / v_ref + cfg.e2 * dtheta / np.pi)
    return -cfg.mu_s * np.where(d <= d_th, psi1, np.where(d <= d_safe, psi2, 0.0))


def hdv_covariance(t: float, cfg: MCTSConfig) -> np.ndarray:
    """Eq. 15-16: covariance of (x, y, v, theta) of an HDV predicted t seconds ahead
    (x along its route, y lateral)."""
    e = cfg.eta_h
    sx, sy = cfg.sigma_x * (1 + cfg.beta_x * e), cfg.sigma_y * (1 + cfg.beta_y * e)
    sv, sth = cfg.sigma_v * (1 + cfg.beta_v * e), cfg.sigma_theta * (1 + cfg.beta_theta * e)
    cov = np.zeros((4, 4))
    cov[0, 0] = sx ** 2 * t + cfg.eps_x ** 2 * t ** 2
    cov[1, 1] = sy ** 2 * t + cfg.eps_y ** 2 * t ** 2
    cov[2, 2] = sv ** 2
    cov[3, 3] = sth ** 2
    cov[0, 2] = cov[2, 0] = cfg.rho_xv * sx * sv * t
    cov[1, 3] = cov[3, 1] = cfg.rho_ytheta * sy * sth * t
    return cov


def footprint_distance(xa, ya, txa, tya, xb, yb, txb, tyb,
                       offsets=(-1.5, 0.0, 1.5), radius: float = 1.0):
    """Clearance between two 5 m x 2 m cars, each covered by three circles of
    radius 1 m along its heading (as in whatif_core); 0 when they overlap."""
    o = np.asarray(offsets)
    ax, ay = xa[..., None] + txa[..., None] * o, ya[..., None] + tya[..., None] * o
    bx, by = xb[..., None] + txb[..., None] * o, yb[..., None] + tyb[..., None] * o
    dx = ax[..., :, None] - bx[..., None, :]
    dy = ay[..., :, None] - by[..., None, :]
    d2 = (dx * dx + dy * dy).reshape(dx.shape[:-2] + (-1,)).min(axis=-1)
    return np.maximum(np.sqrt(d2) - 2.0 * radius, 0.0)


def _angle_between(txa, tya, txb, tyb):
    na = np.maximum(np.hypot(txa, tya), 1e-9)
    nb = np.maximum(np.hypot(txb, tyb), 1e-9)
    return np.arccos(np.clip((txa * txb + tya * tyb) / (na * nb), -1.0, 1.0))


# -- the scene model used inside the tree --------------------------------------
@dataclass
class TreeState:
    s: np.ndarray  # (B, M) arc length on each route
    v: np.ndarray  # (B, M) speed
    lvl: np.ndarray  # (B, K) target-speed level of each group CAV
    acc: np.ndarray  # (B, M) mean acceleration over the last tree step
    jerk: np.ndarray  # (B, M) change of that acceleration per second

    def row(self, i: int) -> "TreeState":
        return TreeState(self.s[i:i + 1], self.v[i:i + 1], self.lvl[i:i + 1],
                         self.acc[i:i + 1], self.jerk[i:i + 1])


@dataclass
class Evaluation:
    safe: np.ndarray  # (B,) bool, Eq. 22
    risk: np.ndarray  # (B,) largest Q_risk / Q_th over the pairs (> 1: unsafe)
    reward: np.ndarray  # (B,) R_total of this step, Eq. 35 (one step, undiscounted)


class SceneModel:
    """Everything the tree needs about one conflict group, fixed for one decision."""

    def __init__(self, obs: SceneObservation, group: ConflictGroup, cfg: MCTSConfig,
                 rng: np.random.Generator, t_wait: dict[int, float],
                 zone: tuple[np.ndarray, float]) -> None:
        self.cfg = cfg
        rc = RolloutConfig()
        ids = _involved(obs, group, cfg.context_radius)
        self.ids = ids
        states = [obs.states[v] for v in ids]
        M = len(ids)
        self.M, self.K = M, len(group.cavs)
        self.mem_idx = np.array([ids.index(v) for v in group.cavs])
        self.is_cav = np.array([st.is_cav for st in states])
        self.is_hdv = ~self.is_cav
        self.levels = np.array(sorted(obs.cav_target_speeds), dtype=float)
        self.v_max = obs.v_max
        self.v_ref = cfg.v_ref or obs.v_max
        self.zone_c, self.zone_r = np.asarray(zone[0], dtype=float), float(zone[1])

        paths = [st.path for st in states]
        self.path_len = np.array([p.length for p in paths])
        self.table = _route_table(paths).astype(np.float64)
        self.g_max = self.table.shape[1] - 1
        self.m_idx = np.arange(M)

        # Fixed targets: other CAVs keep their current target speed.
        self.vt_fixed = np.array([st.target_speed if st.is_cav and st.target_speed is not None
                                  else st.speed for st in states], dtype=float)
        # HDV IDM with the paper's style scaling (Eq. 14), base values as in the rollouts.
        e = cfg.eta_h
        self.idm_a = rc.idm_a * (1 + cfg.alpha_a * e)
        self.idm_T = rc.idm_T * (1 - cfg.alpha_T * e)
        self.idm_s0 = rc.idm_s0 * (1 - cfg.alpha_d * e)
        self.idm_b, self.idm_delta = rc.idm_b, rc.idm_delta
        self.hdv_v0 = rc.hdv_desired_speed
        self.vehicle_length = rc.vehicle_length

        # Route relations: which pairs can meet, and who can lead whom.
        relevant = np.zeros((M, M), dtype=bool)
        shared = np.zeros((M, M), dtype=bool)
        sh = np.zeros((M, M, 3))
        for a in range(M):
            for b in range(M):
                if a != b:
                    meet, st = route_relation(paths[a], paths[b])
                    relevant[a, b] = meet
                    if st is not None:
                        shared[a, b] = True
                        sh[a, b] = (st.s_i, st.s_j, st.length)
        relevant |= relevant.T
        # Leaders for the HDVs (followers): vehicles ahead on a shared stretch.
        self._shared, self._sh = shared, sh
        cand = shared & self.is_hdv[:, None]  # cand[h, l]: l can lead HDV h
        # Vehicles whose motion depends on the joint action: the group CAVs and,
        # transitively, the HDVs that can follow one of them. All other vehicles
        # move the same way in every branch, so they are simulated once per decision.
        dep = np.zeros(M, dtype=bool)
        dep[self.mem_idx] = True
        while True:
            new = dep | (cand & dep[None, :]).any(axis=1)
            if (new == dep).all():
                break
            dep = new
        self.ind_cols = np.flatnonzero(~dep)
        self.dep_cav = np.flatnonzero(dep & self.is_cav)
        self.dep_hdv = np.flatnonzero(dep & self.is_hdv)
        self.mem_pos = np.searchsorted(self.dep_cav, self.mem_idx)  # members within dep_cav
        self.dep_lead = self._leaders(dep & self.is_hdv, self.dep_hdv)
        self.n_inner = max(int(round(cfg.dt_tree / cfg.dt_sim)), 1)

        # Pairs checked and rewarded: a group CAV and a vehicle whose route meets its own.
        member = np.zeros(M, dtype=bool)
        member[self.mem_idx] = True
        pa, pb = np.nonzero(np.triu((member[:, None] | member[None, :]) & relevant, k=1))
        self.pa, self.pb = pa, pb
        self.p_cols = np.unique(np.r_[pa, pb]).astype(np.int64)  # vehicles in some pair
        self.p_a = np.searchsorted(self.p_cols, pa)
        self.p_b = np.searchsorted(self.p_cols, pb)
        self.v2v = self.is_cav[pa] & self.is_cav[pb]
        self.thr = np.where(self.v2v, cfg.q_th_v2v, cfg.q_th_v2h)
        # V2H pairs: index of the HDV and of the CAV.
        hp = np.flatnonzero(~self.v2v)
        self.hp = hp
        h_idx = np.where(self.is_hdv[pa[hp]], pa[hp], pb[hp])
        c_idx = np.where(self.is_hdv[pa[hp]], pb[hp], pa[hp])
        self.hh = np.searchsorted(self.p_cols, h_idx)  # as columns of p_cols
        self.hc = np.searchsorted(self.p_cols, c_idx)
        inc = np.zeros((len(pa), M))  # pair -> vehicle incidence, for per-vehicle sums
        inc[np.arange(len(pa)), pa] = 1.0
        inc[np.arange(len(pa)), pb] = 1.0
        self.inc = inc

        # Cooperation coefficients (Eq. 33) and pair weights w_ij (Eq. 32).
        self.lam = np.array([cfg.lambda_base * math.exp(-cfg.alpha_wait * t_wait.get(v, 0.0))
                             for v in group.cavs])
        self.w_other = 1.0 / max(M - 1, 1)

        # Standard-normal samples for C_ih (common to all nodes of this decision).
        self.z = rng.standard_normal((cfg.n_samples, 4))
        self._samples: dict[int, tuple] = {}
        self.tp_times = cfg.tp_dt * np.arange(1, cfg.tp_steps + 1)

        # Initial state.
        s0 = np.array([st.s for st in states], dtype=float)
        v0 = np.array([st.speed for st in states], dtype=float)
        lvl0 = np.array([int(np.argmin(np.abs(self.levels - (obs.states[v].target_speed
                                                             if obs.states[v].target_speed
                                                             is not None else obs.states[v].speed))))
                         for v in group.cavs])
        z = np.zeros((1, M))
        self.root_state = TreeState(s0[None], v0[None], lvl0[None], z.copy(), z.copy())
        self._precompute_independent(s0, v0)
        self._prepare_dependent()

    # -- geometry ------------------------------------------------------------
    def lookup(self, s, cols=None):
        """x, y, tx, ty at arc lengths ``s`` (..., M), or (..., len(cols)) for those vehicles."""
        u = np.minimum(np.maximum(s * (1.0 / GRID_DS), 0.0), self.g_max - 1e-9)
        i0 = u.astype(np.int64)
        w = (u - i0)[..., None]
        m = self.m_idx if cols is None else cols
        a = self.table[m, i0]
        xyt = a + (self.table[m, i0 + 1] - a) * w
        return xyt[..., 0], xyt[..., 1], xyt[..., 2], xyt[..., 3]

    def zones(self, x, y):
        """Eq. 5: Omega_int (|x|, |y| <= R_EX) and Omega_app (|x| <= 2 R_EX OR |y| <= 2 R_EX)."""
        ax, ay = np.abs(x - self.zone_c[0]), np.abs(y - self.zone_c[1])
        r = self.zone_r
        return (ax <= 2 * r) | (ay <= 2 * r), (ax <= r) & (ay <= r)

    # -- motion ----------------------------------------------------------------
    def _leaders(self, follower_mask: np.ndarray, hdv_cols: np.ndarray):
        """Follower-leader pairs (shared stretch of lane) for the HDVs in ``follower_mask``;
        None if there are none."""
        fi, lj = np.nonzero(self._shared & follower_mask[:, None])
        if not len(fi):
            return None
        order = np.argsort(fi, kind="stable")
        fi, lj = fi[order], lj[order]
        seg = np.flatnonzero(np.r_[True, fi[1:] != fi[:-1]])
        cols = np.unique(np.r_[fi, lj]).astype(np.int64)  # positions needed for these only
        return {
            "fi": fi, "lj": lj, "si": self._sh[fi, lj, 0], "sj": self._sh[fi, lj, 1],
            "len": self._sh[fi, lj, 2], "seg": seg,
            "fpos": np.searchsorted(hdv_cols, fi[seg]),  # followers within hdv_cols
            "cols": cols, "lf": np.searchsorted(cols, fi), "ll": np.searchsorted(cols, lj),
            "len_fi": self.path_len[fi], "len_lj": self.path_len[lj],
        }

    def _integrate(self, s, v, vt_c, cav_cols, hdv_cols, lead, n_steps: int, record: list):
        """Advance columns ``cav_cols`` and ``hdv_cols`` of the (1, M) arrays ``s``, ``v``
        (in place) by ``n_steps`` steps of dt_sim, recording every step. Used once per
        decision for the vehicles that do not depend on the joint action."""
        cfg = self.cfg
        lo, hi = cfg.cav_acc_limits
        dt = cfg.dt_tree / self.n_inner
        sqrt_ab = 2.0 * math.sqrt(self.idm_a * self.idm_b)
        cc, hc = cav_cols, hdv_cols
        for _ in range(n_steps):
            # CAVs: track the target speed (HighwayEnv's controller, bounded acceleration)
            a_c = np.minimum(np.maximum(cfg.kp * (vt_c - v[:, cc]), lo), hi)
            # HDVs: IDM (free road + leader on a shared stretch of lane, as in whatif_core)
            vh = v[:, hc] * (1.0 / self.hdv_v0)
            vh2 = vh * vh
            a_h = self.idm_a * (1.0 - vh2 * vh2)
            if lead is not None:
                fi, lj, lf, ll = lead["fi"], lead["lj"], lead["lf"], lead["ll"]
                x, y, tx, ty = self.lookup(s[:, lead["cols"]], lead["cols"])
                dp = np.hypot(x[:, ll] - x[:, lf], y[:, ll] - y[:, lf])
                sfi, slj = s[:, fi], s[:, lj]
                pos_i, pos_j = sfi - lead["si"], slj - lead["sj"]
                on = ((pos_j >= 0) & (pos_j <= lead["len"]) & (pos_j > pos_i)
                      & (dp < cfg.leader_range) & (sfi < lead["len_fi"]) & (slj < lead["len_lj"]))
                gap = np.maximum(dp - self.vehicle_length, 0.1)
                vi = v[:, fi]
                vj = v[:, lj] * (tx[:, lf] * tx[:, ll] + ty[:, lf] * ty[:, ll])
                s_star = self.idm_s0 + np.maximum(0.0, vi * self.idm_T + vi * (vi - vj) / sqrt_ab)
                q = s_star / gap
                term = np.where(on, -self.idm_a * q * q, 0.0)
                a_h[:, lead["fpos"]] += np.minimum.reduceat(term, lead["seg"], axis=1)
            vcn = np.minimum(np.maximum(v[:, cc] + a_c * dt, 0.0), self.v_max)
            vhn = np.maximum(v[:, hc] + a_h * dt, 0.0)
            v[:, cc] = vcn
            v[:, hc] = vhn
            s[:, cc] += vcn * dt
            s[:, hc] += vhn * dt
            record.append((s.copy(), v.copy()))

    def _precompute_independent(self, s0: np.ndarray, v0: np.ndarray) -> None:
        """Trajectories of the vehicles that do not depend on the joint action, at every
        dt_sim over the whole tree horizon (d_max tree steps)."""
        ind = np.zeros(self.M, dtype=bool)
        ind[self.ind_cols] = True
        cav, hdv = np.flatnonzero(ind & self.is_cav), np.flatnonzero(ind & self.is_hdv)
        s, v = s0[None].copy(), v0[None].copy()
        rec = [(s.copy(), v.copy())]
        n = self.cfg.d_max * self.n_inner
        self._integrate(s, v, self.vt_fixed[cav][None], cav, hdv,
                        self._leaders(ind & self.is_hdv, hdv), n, record=rec)
        ic = self.ind_cols
        self.s_ind = np.array([r[0][0, ic] for r in rec])  # (n + 1, |ind|)
        self.v_ind = np.array([r[1][0, ic] for r in rec])
        vd = self.v_ind[::self.n_inner]  # at every tree depth
        acc = np.zeros_like(vd)
        acc[1:] = (vd[1:] - vd[:-1]) / self.cfg.dt_tree
        self.acc_ind = acc  # (d_max + 1, |ind|), acc at depth 0 = 0 as in the root state
        self.jerk_ind = np.zeros_like(acc)
        self.jerk_ind[1:] = (acc[1:] - acc[:-1]) / self.cfg.dt_tree

    def _prepare_dependent(self) -> None:
        """Index tables for simulating only the action-dependent vehicles, in compact
        arrays ordered [dependent CAVs, dependent HDVs]. Leaders of dependent HDVs
        that are independent vehicles are read from the precomputed trajectories."""
        self.dep_order = np.r_[self.dep_cav, self.dep_hdv].astype(np.int64)
        self.n_dc = len(self.dep_cav)
        lead = self.dep_lead
        self.dl = None
        if lead is None:
            return
        fi, lj = lead["fi"], lead["lj"]
        pos_in_dep = {int(c): k for k, c in enumerate(self.dep_order)}
        pos_in_ind = {int(c): k for k, c in enumerate(self.ind_cols)}
        f_loc = np.array([pos_in_dep[int(f)] for f in fi])
        in_dep = np.array([int(l) in pos_in_dep for l in lj])
        l_loc = np.array([pos_in_dep.get(int(l), -1) for l in lj])
        l_ind = np.array([pos_in_ind.get(int(l), -1) for l in lj])
        LD = np.unique(np.r_[f_loc, l_loc[in_dep]]).astype(np.int64)  # dependent columns used
        LI = np.unique(l_ind[~in_dep]).astype(np.int64)  # independent leaders used
        f_c = np.searchsorted(LD, f_loc)
        l_c = np.where(in_dep, np.searchsorted(LD, np.where(in_dep, l_loc, LD[0])),
                       len(LD) + np.searchsorted(LI, np.where(in_dep, LI[0] if len(LI) else 0, l_ind)))
        ind_data = np.zeros((len(self.s_ind), len(LI), 6))  # s, v, x, y, tx, ty at every dt_sim
        if len(LI):
            cols = self.ind_cols[LI]
            x, y, tx, ty = self.lookup(self.s_ind[:, LI], cols)
            ind_data = np.stack([self.s_ind[:, LI], self.v_ind[:, LI], x, y, tx, ty], axis=-1)
        self.dl = {"LD": LD, "LD_full": self.dep_order[LD], "f_c": f_c, "l_c": l_c,
                   "ind_data": ind_data, "fpos": np.searchsorted(self.dep_hdv, fi[lead["seg"]])}

    def _integrate_dep(self, sD, vD, vt_c, n_steps: int, k0: int) -> None:
        """Advance the compact dependent arrays sD, vD (B, n_dep) in place; same
        equations as :meth:`_integrate`."""
        cfg = self.cfg
        lo, hi = cfg.cav_acc_limits
        dt = cfg.dt_tree / self.n_inner
        nc = self.n_dc
        dl, lead = self.dl, self.dep_lead
        if dl is not None:
            sqrt_ab = 2.0 * math.sqrt(self.idm_a * self.idm_b)
            B = len(sD)
            LD, f_c, l_c = dl["LD"], dl["f_c"], dl["l_c"]
            n_li = dl["ind_data"].shape[1]
        for j in range(n_steps):
            vc, vh_ = vD[:, :nc], vD[:, nc:]
            a_c = np.minimum(np.maximum(cfg.kp * (vt_c - vc), lo), hi)
            if dl is not None:
                vh = vh_ * (1.0 / self.hdv_v0)
                vh2 = vh * vh
                a_h = self.idm_a * (1.0 - vh2 * vh2)
                u = np.minimum(np.maximum(sD[:, LD] * (1.0 / GRID_DS), 0.0), self.g_max - 1e-9)
                i0 = u.astype(np.int64)
                ta = self.table[dl["LD_full"], i0, :4]
                xyt = ta + (self.table[dl["LD_full"], i0 + 1, :4] - ta) * (u - i0)[..., None]
                part = np.concatenate([sD[:, LD, None], vD[:, LD, None], xyt], axis=-1)
                if n_li:
                    part = np.concatenate(
                        [part, np.broadcast_to(dl["ind_data"][k0 + j], (B, n_li, 6))], axis=1)
                F, L = part[:, f_c], part[:, l_c]
                sfi, vi, xf, yf, txf, tyf = (F[..., c] for c in range(6))
                slj, vl, xl, yl, txl, tyl = (L[..., c] for c in range(6))
                dp = np.hypot(xl - xf, yl - yf)
                pos_i, pos_j = sfi - lead["si"], slj - lead["sj"]
                on = ((pos_j >= 0) & (pos_j <= lead["len"]) & (pos_j > pos_i)
                      & (dp < cfg.leader_range) & (sfi < lead["len_fi"]) & (slj < lead["len_lj"]))
                gap = np.maximum(dp - self.vehicle_length, 0.1)
                vj = vl * (txf * txl + tyf * tyl)
                s_star = self.idm_s0 + np.maximum(0.0, vi * self.idm_T + vi * (vi - vj) / sqrt_ab)
                q = s_star / gap
                term = np.where(on, -self.idm_a * q * q, 0.0)
                a_h[:, dl["fpos"]] += np.minimum.reduceat(term, lead["seg"], axis=1)
                vD[:, nc:] = np.maximum(vh_ + a_h * dt, 0.0)
            vD[:, :nc] = np.minimum(np.maximum(vc + a_c * dt, 0.0), self.v_max)
            sD += vD * dt

    def step(self, state: TreeState, joint: np.ndarray, depth: int) -> TreeState:
        """Apply joint actions (B, K) for one tree step from ``depth`` to ``depth + 1``
        (``state``: 1 or B rows)."""
        B = len(joint)
        n = self.n_inner
        k0 = depth * n
        lvl = np.clip(np.broadcast_to(state.lvl, joint.shape) + (joint - IDLE), 0,
                      len(self.levels) - 1)
        do = self.dep_order
        sD = np.broadcast_to(state.s[:, do], (B, len(do))).copy()
        vD = np.broadcast_to(state.v[:, do], (B, len(do))).copy()
        vt_c = np.broadcast_to(self.vt_fixed[self.dep_cav], (B, self.n_dc)).copy()
        vt_c[:, self.mem_pos] = self.levels[lvl]
        self._integrate_dep(sD, vD, vt_c, n, k0)
        s = np.empty((B, self.M))
        v = np.empty((B, self.M))
        ic = self.ind_cols
        s[:, do], v[:, do] = sD, vD
        s[:, ic] = self.s_ind[k0 + n]
        v[:, ic] = self.v_ind[k0 + n]
        v_start = state.v
        acc = (v - v_start) / self.cfg.dt_tree
        jerk = (acc - state.acc) / self.cfg.dt_tree
        acc[:, ic] = self.acc_ind[depth + 1]
        jerk[:, ic] = self.jerk_ind[depth + 1]
        return TreeState(s, v, lvl, acc, jerk)

    # -- safety check and reward -------------------------------------------------
    def _hdv_samples(self, depth: int):
        """Correlated (dx, dy, dv, dtheta) samples at t = depth * dt_tree (Eq. 15-17)."""
        if depth not in self._samples:
            cfg = self.cfg
            t = depth * cfg.dt_tree
            cov = hdv_covariance(t, cfg)
            sd = np.sqrt(np.diag(cov))
            corr = np.divide(cov, np.outer(sd, sd), out=np.zeros_like(cov),
                             where=np.outer(sd, sd) > 0)
            corr = np.clip(corr, -0.99, 0.99)
            np.fill_diagonal(corr, 1.0)
            smp = (self.z @ np.linalg.cholesky(corr).T) * sd
            # Eq. 17: bounded displacement and speed
            sv = cfg.sigma_v * (1 + cfg.beta_v * cfg.eta_h)
            disp = np.hypot(smp[:, 0], smp[:, 1])
            lim = (self.v_max + sv) * t
            scale = np.where(disp > lim, lim / np.maximum(disp, 1e-12), 1.0)
            smp[:, 0] *= scale
            smp[:, 1] *= scale
            self._samples[depth] = (smp, sv)
        return self._samples[depth]

    def evaluate(self, state: TreeState, depth: int) -> Evaluation:
        cfg = self.cfg
        s, v = state.s, state.v
        B = len(s)
        active = s < self.path_len
        pa, pb = self.pa, self.pb
        P = len(pa)
        if P:
            # Now (index 0) and the T_p predicted points (constant speed along the route),
            # for the vehicles that are in some pair.
            pc, ia, ib = self.p_cols, self.p_a, self.p_b
            sc, vc_ = s[:, pc], v[:, pc]
            sp = np.concatenate([sc[:, None, :], sc[:, None, :] + vc_[:, None, :]
                                 * self.tp_times[None, :, None]], axis=1)  # (B, 1+T, C)
            X, Y, TX, TY = self.lookup(sp, pc)
            d_all = footprint_distance(X[..., ia], Y[..., ia], TX[..., ia], TY[..., ia],
                                       X[..., ib], Y[..., ib], TX[..., ib], TY[..., ib])
            x, y, tx, ty = X[:, 0], Y[:, 0], TX[:, 0], TY[:, 0]  # (B, C)
            d = d_all[:, 0]
            rvx = vc_[:, ia] * tx[:, ia] - vc_[:, ib] * tx[:, ib]  # relative velocity
            rvy = vc_[:, ia] * ty[:, ia] - vc_[:, ib] * ty[:, ib]
            dv = np.hypot(rvx, rvy)
            dth = _angle_between(tx[:, ia], ty[:, ia], tx[:, ib], ty[:, ib])
            app, inn = self.zones(x, y)
            in_app = app[:, ia] | app[:, ib]
            in_int = inn[:, ia] | inn[:, ib]
            d_safe = dynamic_safe_distance(dv, dth, in_app, in_int, cfg, self.v_ref)
            r = instantaneous_risk(d, d_safe, dv, cfg, self.v_max)

            # Temporal risk, Eq. 10 (a vehicle past the end of its route has left).
            spp = sp[:, 1:]
            gone = (spp[..., ia] >= self.path_len[pa]) | (spp[..., ib] >= self.path_len[pb])
            rtp = temporal_risk(np.where(gone, np.inf, d_all[:, 1:]), d_safe, cfg.tp_steps)

            # Collision probability with the uncertain HDV, Eq. 18.
            c_ih = np.zeros((B, P))
            if len(self.hp) and depth > 0:
                smp, sv = self._hdv_samples(depth)
                h, c = self.hh, self.hc  # columns of the HDV and of the CAV in (B, C)
                dx, dy, dvs, dths = (smp[:, k] for k in range(4))
                xh, yh, txh, tyh = (a[:, h, None] for a in (x, y, tx, ty))
                nh = np.maximum(np.hypot(txh, tyh), 1e-9)
                txh, tyh = txh / nh, tyh / nh
                hx = xh + txh * dx - tyh * dy
                hy = yh + tyh * dx + txh * dy
                cs, sn = np.cos(dths), np.sin(dths)
                htx, hty = txh * cs - tyh * sn, tyh * cs + txh * sn
                hv = np.clip(vc_[:, h, None] + dvs, 0.0, self.v_max + 2 * sv)
                xc, yc, txc, tyc = (a[:, c, None] for a in (x, y, tx, ty))
                ds = footprint_distance(xc, yc, txc, tyc, hx, hy, htx, hty)
                vc = vc_[:, c, None]
                dvh = np.hypot(vc * txc - hv * htx, vc * tyc - hv * hty)
                dthh = _angle_between(txc, tyc, htx, hty)
                hp = self.hp
                dsafe_h = dynamic_safe_distance(dvh, dthh, in_app[:, hp, None], in_int[:, hp, None],
                                                cfg, self.v_ref) * (1 - cfg.alpha_d * cfg.eta_h)
                c_ih[:, hp] = (ds < dsafe_h).mean(axis=-1)

            q = np.where(self.v2v, cfg.w1_v2v * r + cfg.w2_v2v * rtp,
                         cfg.w1_v2h * r + cfg.w2_v2h * rtp + cfg.w3_v2h * c_ih)
            pair_on = active[:, pa] & active[:, pb]
            ratio = np.where(pair_on, q / self.thr, 0.0)
            risk = ratio.max(axis=1)
            safe = risk <= 1.0  # V2R is always satisfied (lane centres)

            # Safety reward per pair, shared by both vehicles (Eq. 29-30).
            sign = -1.0 if cfg.risk_as_penalty else 1.0
            c_v2v = cfg.w_v2v * (phi_v2v(d, d_safe) + sign * cfg.lambda_T * rtp)
            c_v2h = cfg.w_v2h * (phi_v2h(d, d_safe, dv, dth, cfg, self.v_ref)
                                 + sign * cfg.lambda_T * rtp + sign * cfg.lambda_c * c_ih)
            contrib = np.where(pair_on, np.where(self.v2v, c_v2v, c_v2h), 0.0)
            q_safety = contrib @ self.inc  # (B, M)
        else:
            risk = np.zeros(B)
            safe = np.ones(B, dtype=bool)
            q_safety = np.zeros((B, self.M))

        # Efficiency and comfort per vehicle.
        dev = v - self.v_max
        q_vel = np.where(np.abs(dev) <= cfg.v_tol, 0.0,
                         np.where(dev > cfg.v_tol, -cfg.alpha_v * (dev - cfg.v_tol),
                                  -cfg.beta_v_pen * np.abs(dev)))
        q_eff = cfg.w_vel * q_vel - cfg.w_acc * state.acc ** 2
        q_eff = np.where(active, q_eff, 0.0)
        q_comfort = np.where(active, -cfg.w_jerk * state.jerk ** 2, 0.0)
        q_safety = np.where(active, q_safety, 0.0)

        mi = self.mem_idx
        q_self = (cfg.w_self_safety * q_safety[:, mi] + cfg.w_self_eff * q_eff[:, mi]
                  + cfg.w_self_comfort * q_comfort[:, mi])
        both = q_safety + q_eff
        q_other = (both.sum(axis=1, keepdims=True) - both[:, mi]) * self.w_other
        n = self.K
        r_i = (q_self + self.lam * q_other) / (1.0 + self.lam * (n - 1))
        return Evaluation(safe, risk, r_i.sum(axis=1))


# -- the tree -----------------------------------------------------------------
class Node:
    """Tree node (Eq. 21). Statistics of children are kept in arrays on the parent."""

    __slots__ = ("depth", "parent", "idx", "state", "G", "children", "ch_state", "ch_G",
                 "ch_safe", "ch_risk", "cN", "cQ", "N", "Q")

    def __init__(self, depth, parent, idx, state, G):
        self.depth, self.parent, self.idx, self.state, self.G = depth, parent, idx, state, G
        self.children = None  # dict idx -> Node, created when first selected
        self.ch_state = self.ch_G = self.ch_safe = self.ch_risk = self.cN = self.cQ = None
        self.N, self.Q = 0, 0.0  # used for the root only

    @property
    def visits(self) -> int:
        return self.N if self.parent is None else int(self.parent.cN[self.idx])

    @property
    def expanded(self) -> bool:
        return self.ch_safe is not None


class MCTSPlanner:
    """Algorithm 1 for one conflict group."""

    def __init__(self, model: SceneModel, cfg: MCTSConfig, rng: np.random.Generator) -> None:
        self.model, self.cfg, self.rng = model, cfg, rng
        self.joint = np.array(list(itertools.product((SLOWER, IDLE, FASTER), repeat=model.K)),
                              dtype=np.int64)  # (3^K, K)
        self.root = Node(0, None, -1, model.root_state, 0.0)
        self.n_nodes = 1
        self.n_unsafe = 0

    # Expand(n): all children, with their safety status (Eq. 22, 24)
    def expand(self, n: Node) -> None:
        A = len(self.joint)
        chunk = 729  # bound the memory of very large groups
        parts = []
        for a0 in range(0, A, chunk):
            st = self.model.step(n.state, self.joint[a0:a0 + chunk], n.depth)
            parts.append((st, self.model.evaluate(st, n.depth + 1)))
        cat = lambda f: np.concatenate([getattr(p[0], f) for p in parts])  # noqa: E731
        n.ch_state = TreeState(cat("s"), cat("v"), cat("lvl"), cat("acc"), cat("jerk"))
        n.ch_safe = np.concatenate([e.safe for _, e in parts])
        n.ch_risk = np.concatenate([e.risk for _, e in parts])
        reward = np.concatenate([e.reward for _, e in parts])
        n.ch_G = n.G + self.cfg.gamma_r ** n.depth * reward
        n.cN = np.zeros(A, dtype=np.int64)
        n.cQ = np.zeros(A)
        n.children = {}
        self.n_nodes += A
        self.n_unsafe += int((~n.ch_safe).sum())

    def child(self, n: Node, i: int) -> Node:
        if i not in n.children:
            n.children[i] = Node(n.depth + 1, n, i, n.ch_state.row(i), float(n.ch_G[i]))
        return n.children[i]

    # SelectChild(n): UCB over the safe children (Eq. 25)
    def select(self, n: Node) -> int | None:
        safe = n.ch_safe
        if not safe.any():
            return None
        N = n.cN
        with np.errstate(divide="ignore", invalid="ignore"):
            ucb = n.cQ + self.cfg.c_exp * np.sqrt(math.log(max(n.visits, 1)) / N)
        ucb = np.where(N == 0, np.inf, ucb)
        ucb = np.where(safe, ucb, -np.inf)
        return int(np.argmax(ucb))

    # Rollout(n): random safe joint actions to depth d_max (Eq. 26-27, alpha = 0)
    def rollout(self, n: Node) -> float:
        cfg, model = self.cfg, self.model
        state, G = n.state, n.G
        A = len(self.joint)
        for t in range(n.depth, cfg.d_max):
            perm = self.rng.permutation(A)
            first = None
            chosen = None
            # The first safe action of a random order is uniform over the safe set.
            # Actions are checked in batches of 1, 2, 4, ... (speed only: same result).
            a0, size = 0, min(A, cfg.rollout_first)
            while a0 < A:
                idx = perm[a0:a0 + size]
                st = model.step(state, self.joint[idx], t)
                ev = model.evaluate(st, t + 1)
                if first is None:
                    first = (st, ev)
                ok = np.flatnonzero(ev.safe)
                if len(ok):
                    chosen = (st, ev, int(ok[0]))
                    break
                a0 += size
                size = min(2 * size, cfg.rollout_chunk)
            if chosen is None:  # no safe joint action: uniform over all
                st, ev = first
                chosen = (st, ev, 0)
            st, ev, k = chosen
            G += cfg.gamma_r ** t * float(ev.reward[k])
            state = st.row(k)
        return G

    # Backpropagate(n, Delta), Eq. 28
    def backpropagate(self, n: Node, delta: float) -> None:
        b = self.cfg.beta_dec
        while n is not None:
            p = n.parent
            if p is None:
                n.N += 1
                n.Q += (delta - n.Q) / n.N ** b
            else:
                p.cN[n.idx] += 1
                p.cQ[n.idx] += (delta - p.cQ[n.idx]) / p.cN[n.idx] ** b
            n = p

    def iterate(self) -> None:
        n = self.root
        while True:
            if n.depth >= self.cfg.d_max:  # max depth reached (see docs: not specified)
                self.backpropagate(n, n.G)
                return
            if n.visits == 0:
                self.backpropagate(n, self.rollout(n))
                return
            if not n.expanded:
                self.expand(n)
                self.backpropagate(n, self.rollout(n))
                return
            i = self.select(n)
            if i is None:  # every child unsafe (see docs: not specified)
                self.backpropagate(n, self.rollout(n))
                return
            n = self.child(n, i)

    def run(self, n_iterations: int) -> None:
        for _ in range(n_iterations):
            self.iterate()
        if not self.root.expanded:
            self.expand(self.root)

    def best(self) -> tuple[int, bool]:
        """Eq. 41 over the safe root children; (index, False) if none is safe,
        then the least unsafe child (lowest maximum risk)."""
        r = self.root
        if r.ch_safe.any():
            score = np.where(r.ch_safe, r.cN + self.cfg.eps * r.cQ, -np.inf)
            return int(np.argmax(score)), True
        return int(np.argmin(r.ch_risk)), False


class MCTSArbiter(Arbiter):
    """Safety-critical multi-agent MCTS (Lin et al., T-ITS 2025), per conflict group.

    Parameters: any field of :class:`MCTSConfig` (e.g. ``n_iterations: 200`` in
    a YAML config), plus the usual Arbiter arguments."""

    name = "mcts"

    def __init__(self, config: dict | MCTSConfig | None = None, **kw) -> None:
        mcts_kw = {k: kw.pop(k) for k in list(kw) if k in MCTS_FIELDS}
        super().__init__(**kw)
        if isinstance(config, MCTSConfig):
            base = config
        else:
            base = MCTSConfig(**(config or {}))
        self.cfg = MCTSConfig(**{**base.__dict__, **mcts_kw})
        if self.cfg.reuse_tree:
            raise NotImplementedError(
                "reuse_tree=True (Eq. 43) is not implemented: the scene is re-observed with "
                "noise every 0.2 s, so the tree is rebuilt from the new observation.")
        if self.cfg.alpha_model != 0.0:
            raise NotImplementedError("alpha_model must be 0: there is no learned value predictor.")
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.t_wait: dict[int, float] = {}
        self.n_iterations = self.n_nodes = self.n_unsafe = 0
        self.no_safe_root = 0
        self.max_group = 0

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self._reset_counters()

    def act(self, obs, all_cavs):
        # T_wait (Eq. 33): time each CAV has spent below v_thres in this episode.
        for v in all_cavs:
            st = obs.states.get(v)
            if st is not None and st.speed < self.cfg.v_thres:
                self.t_wait[v] = self.t_wait.get(v, 0.0) + obs.policy_dt
        return super().act(obs, all_cavs)

    def zone(self, obs: SceneObservation, group: ConflictGroup) -> tuple[np.ndarray, float]:
        """Centre and R_EX of the conflict zone (Eq. 5) for this scenario."""
        cfg = self.cfg
        r_ex = cfg.zone_r_ex or INTERSECTION_R_EX
        if cfg.zone_center is not None:
            return np.asarray(cfg.zone_center, dtype=float), r_ex
        nodes = {n for st in obs.states.values() for lane in st.path.key for n in lane}
        if any(str(n).startswith(("ir", "il")) for n in nodes):
            return np.zeros(2), r_ex  # HighwayEnv intersection: centre at the origin
        if nodes and nodes <= {"a", "b", "c", "j"}:
            from ..envs.merge import MERGE_X  # CAVBench merge: the ramp joins at (MERGE_X, 0)
            return np.array([MERGE_X, 0.0]), r_ex
        # Unknown scenario: the first meeting point of the group's first conflict.
        if group.conflicts:
            c = group.conflicts[0]
            a, b = obs.states[c.i], obs.states[c.j]
            mps = meeting_points(a.path, b.path, 1.0)
            if mps:
                return a.path.position(mps[0].s_i), r_ex
        return np.asarray(obs.states[group.cavs[0]].position, dtype=float), r_ex

    def plan(self, obs: SceneObservation, group: ConflictGroup) -> tuple[dict[int, int], MCTSPlanner, bool]:
        model = SceneModel(obs, group, self.cfg, self.rng, self.t_wait, self.zone(obs, group))
        planner = MCTSPlanner(model, self.cfg, self.rng)
        planner.run(self.cfg.n_iterations)
        k, ok = planner.best()
        acts = {v: int(a) for v, a in zip(group.cavs, planner.joint[k])}
        return acts, planner, ok

    def arbitrate(self, obs, groups):
        actions = {}
        for g in groups:
            acts, planner, ok = self.plan(obs, g)
            actions.update(acts)
            self.n_iterations += self.cfg.n_iterations
            self.n_nodes += planner.n_nodes
            self.n_unsafe += planner.n_unsafe
            self.max_group = max(self.max_group, len(g.cavs))
            if not ok:
                self.no_safe_root += 1
            root = planner.root
            self.log.append({
                "time": obs.time, "mode": "mcts" if ok else "mcts_no_safe_root",
                "group": list(g.cavs), "actions": acts,
                "safe_children": int(root.ch_safe.sum()), "children": int(len(root.ch_safe)),
            })
        return actions

    def extra_metrics(self) -> dict:
        return {
            "mcts_iterations": self.n_iterations,
            "mcts_nodes": self.n_nodes,
            "mcts_pruned_fraction": self.n_unsafe / self.n_nodes if self.n_nodes else 0.0,
            "mcts_no_safe_root": self.no_safe_root,
            "mcts_max_group_size": self.max_group,
        }
