"""Watch one CAVBench episode: roads, cars and the arbiter's decisions.

The experiments run without graphics (much faster). Use this script to see
what happens in a single episode, or to save it as a video:

    python experiments/watch.py                                  # UCRA, intersection
    python experiments/watch.py --arbiter fcfs --noise high --seed 3
    python experiments/watch.py --scenario intersection_dense --seed 7
    python experiments/watch.py --scenario merge
    python experiments/watch.py --seed 3 --save videos/ucra_seed3.mp4   # or .gif

The episode is exactly the one the experiments run for the same scenario,
arbiter, noise level and seed (same traffic, same noise samples).

What you see:
  * grey roads; blue cars are human-driven (IDM); numbered cars are CAVs;
  * CAV colour = the arbiter's command: green = go, yellow = cautious /
    keep speed, orange = yield (slow down); red = crashed;
  * small rings = where the other vehicles *appear* to be in the noisy
    shared data (only with noise);
  * magenta lines = CAVs in conflict (Delta TTCP < 8 s) being arbitrated;
  * top panel = the latest UCRA decision: who goes, risk, uncertainty and
    fuzzy acceptability A (accept > 0.7, cautious 0.4-0.7, fallback <= 0.4).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cavbench.arbiters import FASTER, IDLE, SLOWER, make_arbiter  # noqa: E402
from cavbench.envs import make_env  # noqa: E402
from cavbench.metrics import episode_metrics  # noqa: E402

# Same scenario presets as configs/*.yaml
SCENARIOS = {
    "intersection": ("intersection", {}),
    "intersection_dense": ("intersection", {"controlled_vehicles": 8, "initial_vehicle_count": 14}),
    "merge": ("merge", {}),
}
ARBITERS = {  # short name -> (kind, params, label)
    "ucra": ("whatif_fuzzy", {}, "UCRA (ours)"),
    "ucra_nou": ("whatif_fuzzy", {"use_uncertainty": False}, "UCRA without U"),
    "ucra_nofallback": ("whatif_fuzzy", {"use_fallback": False}, "UCRA without fallback"),
    "fcfs": ("fcfs", {}, "First come, first served"),
    "nocoord": ("no_coordination", {}, "No coordination"),
    "llm": ("codrivingllm", {"model": "gpt-4o-mini"}, "CoDrivingLLM-style (gpt-4o-mini)"),
}
VIEW = {  # screen size [px], scaling [px/m], camera centre [m]
    "intersection": {"screen_width": 820, "screen_height": 820, "scaling": 5.2,
                     "centering_position": [0.5, 0.5], "camera": (0.0, 0.0)},
    "merge": {"screen_width": 1150, "screen_height": 330, "scaling": 6.0,
              "centering_position": [0.5, 0.5], "camera": (215.0, 6.0)},
}
CMD_COLOURS = {FASTER: (40, 190, 60), IDLE: (235, 200, 30), SLOWER: (255, 130, 20)}
PANEL_H = 64


class _Camera:
    """A fixed point for HighwayEnv's viewer to centre on."""

    def __init__(self, xy):
        self.position = np.array(xy, dtype=float)


class Watcher:
    """Draws every simulation frame (15 Hz) with the decision overlay."""

    def __init__(self, env, arbiter, title: str, noise: str, seed: int,
                 save: str | None, speed: float) -> None:
        import pygame

        self.pg = pygame
        self.env, self.arbiter = env, arbiter
        self.title, self.noise, self.seed = title, noise, seed
        self.save, self.speed = save, speed
        self.frames: list[np.ndarray] = []
        self.obs, self.actions, self.groups = None, {}, []
        self.label: dict[int, int] = {}
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("arial", 15)
        self.small = pygame.font.SysFont("arial", 13)
        self.big = pygame.font.SysFont("arial", 40, bold=True)
        self.window = None
        self.clock = pygame.time.Clock()
        self.closed = False

    # HighwayEnv calls this for the intermediate frames of each step.
    def _capture_frame(self) -> None:
        self.draw()

    def setup(self, camera) -> None:
        self.env.render()  # creates the viewer
        self.env.viewer.observer_vehicle = _Camera(camera)
        self.env._record_video_wrapper = self  # receive intermediate frames
        self.label = {self.env.vid(v): i + 1 for i, v in enumerate(self.env.controlled_vehicles)}

    def colour_cavs(self) -> None:
        cautious = set()
        if self.obs is not None:
            for e in self.arbiter.log:
                if abs(e.get("time", -1) - self.obs.time) < 1e-6 and e.get("mode") == "cautious":
                    cautious.update(e.get("group", []))
        for v in self.env.controlled_vehicles:
            vid = self.env.vid(v)
            a = self.actions.get(vid, FASTER)
            if a != SLOWER and vid in cautious:
                a = IDLE
            v.color = None if v.crashed else CMD_COLOURS[a]

    def _world(self, xy):
        return self.env.viewer.sim_surface.pos2pix(float(xy[0]), float(xy[1]))

    def draw(self, banner: str | None = None) -> None:
        if self.closed:
            return
        pg = self.pg
        img = self.env.render()  # (H, W, 3)
        surf = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        h, w = img.shape[0], img.shape[1]
        canvas = pg.Surface((w, h + PANEL_H))
        canvas.fill((250, 250, 250))
        canvas.blit(surf, (0, PANEL_H))

        def at(xy):
            x, y = self._world(xy)
            return int(x), int(y) + PANEL_H

        road = self.env.road
        if self.obs is not None:
            # conflict links
            for g in self.groups:
                pts = [self.obs.states[c].position for c in g.cavs if c in self.obs.states]
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        pg.draw.line(canvas, (200, 0, 200), at(pts[i]), at(pts[j]), 2)
            # noisy shared positions
            if self.noise != "none":
                for st in self.obs.states.values():
                    pg.draw.circle(canvas, (20, 20, 20), at(st.position), 5, 1)
        # CAV numbers
        for v in self.env.controlled_vehicles:
            if v in road.vehicles:
                n = self.label.get(self.env.vid(v), 0)
                t = self.small.render(str(n), True, (0, 0, 0), (255, 255, 255))
                x, y = at(v.position)
                canvas.blit(t, (x + 9, y - 18))

        # top panel
        t = self.env.time
        sigma = getattr(self.env.noise, "sigma_pos", 0.0)
        head = (f"{self.title}   |   noise: {self.noise} (sigma = {sigma:g} m)   |   "
                f"seed {self.seed}   |   t = {t:4.1f} s")
        canvas.blit(self.font.render(head, True, (20, 20, 20)), (10, 6))
        canvas.blit(self.small.render(self._decision_text(), True, (60, 60, 60)), (10, 26))
        x = 10
        for name, col in (("go", CMD_COLOURS[FASTER]), ("cautious / keep speed", CMD_COLOURS[IDLE]),
                          ("yield", CMD_COLOURS[SLOWER]), ("human driver", (100, 200, 255)),
                          ("crashed", (255, 100, 100))):
            pg.draw.rect(canvas, col, (x, 47, 12, 10))
            lab = self.small.render(name, True, (60, 60, 60))
            canvas.blit(lab, (x + 16, 44))
            x += 16 + lab.get_width() + 18
        if banner:
            col = (30, 150, 50) if banner.startswith("SUCCESS") else (200, 30, 30)
            b = self.big.render(banner, True, col, (255, 255, 255))
            canvas.blit(b, ((w - b.get_width()) // 2, PANEL_H + 20))  # top, not over the cars

        frame = pg.surfarray.array3d(canvas).swapaxes(0, 1)
        if self.save:
            self.frames.append(frame)
        else:
            if self.window is None:
                self.window = pg.display.set_mode(canvas.get_size())
                pg.display.set_caption("CAVBench")
            self.window.blit(canvas, (0, 0))
            pg.display.flip()
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    self.closed = True
            self.clock.tick(15 * self.speed)

    def _decision_text(self) -> str:
        if not self.groups:
            return "No conflict: CAVs drive normally."
        log = [e for e in self.arbiter.log if self.obs is not None
               and abs(e.get("time", -1) - self.obs.time) < 1e-6 and "acceptability" in e]
        if not log:
            ids = ", ".join(str(self.label.get(c, c)) for g in self.groups for c in g.cavs)
            return f"Conflict between CAVs {ids} (Delta TTCP < 8 s)."
        parts = []
        for e in log:
            grp = ",".join(str(self.label.get(c, c)) for c in e["group"])
            if e["goer"] is None:
                parts.append(f"CAVs {grp}: fallback, all yield (best A = {max(e['acceptability']):.2f})")
                continue
            k = int(np.argmax(e["acceptability"]))
            parts.append(
                f"CAVs {grp}: CAV {self.label.get(e['goer'], e['goer'])} goes [{e['mode']}]  "
                f"risk = {e['risk'][k]:.2f}  U = {e['uncertainty'][k]:.2f}  "
                f"A = {e['acceptability'][k]:.2f}  wait = {e['wait']:.1f} s")
        return "   ||   ".join(parts)

    def finish(self, banner: str) -> None:
        for _ in range(30):  # hold the last frame for 2 s
            self.draw(banner)
        if self.save:
            import imageio.v2 as imageio

            Path(self.save).parent.mkdir(parents=True, exist_ok=True)
            if self.save.lower().endswith(".gif"):
                imageio.mimsave(self.save, self.frames[::2], duration=2 / 15, loop=0)
            else:
                imageio.mimsave(self.save, self.frames, fps=15, macro_block_size=1)
            print(f"Saved {len(self.frames)} frames -> {self.save}")
        self.pg.quit()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="intersection", choices=list(SCENARIOS))
    p.add_argument("--arbiter", default="ucra", choices=list(ARBITERS))
    p.add_argument("--noise", default="high", choices=["none", "low", "medium", "high"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="save to .mp4 or .gif instead of opening a window")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (window only)")
    a = p.parse_args(argv)

    if a.save and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        # No screen (e.g. a server): draw off-screen. ("dummy" would disable
        # HighwayEnv's drawing altogether.)
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
    scenario, env_cfg = SCENARIOS[a.scenario]
    view = dict(VIEW[scenario])
    camera = view.pop("camera")
    env = make_env(scenario, {**env_cfg, **view, "noise_level": a.noise}, render_mode="rgb_array")
    kind, params, title = ARBITERS[a.arbiter]
    arbiter = make_arbiter(kind, **params)

    # Same seeding as cavbench.runner.run_episode (matched seeds).
    env.reset(seed=a.seed)
    arbiter.reset(seed=a.seed + 10_000_019)
    noise_rng = np.random.default_rng([a.seed, 2026])
    w = Watcher(env, arbiter, f"{a.scenario}  -  {title}", a.noise, a.seed, a.save, a.speed)
    w.setup(camera)
    w.draw()

    done = False
    while not done and not w.closed:
        obs = env.observe(noise_rng)
        acts = arbiter.act(obs, env.active_cav_vids())
        w.obs, w.actions, w.groups = obs, acts, arbiter.detector.groups(obs)
        w.colour_cavs()
        action = tuple(int(acts.get(v, FASTER)) for v in env.cav_vids())
        _, _, terminated, truncated, _ = env.step(action)
        w.colour_cavs()
        w.draw()
        done = terminated or truncated

    m = episode_metrics(env)
    if m.get("collision"):
        banner = "COLLISION"
    elif m.get("success"):
        banner = "SUCCESS: all CAVs passed"
    else:
        banner = "TIMEOUT"
    keys = ("success", "collision", "severe_conflicts", "min_pet", "avg_speed", "episode_time")
    print(f"{a.scenario} | {title} | noise {a.noise} | seed {a.seed}")
    print("  " + "  ".join(f"{k}={m[k]:.2f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
                           for k in keys if k in m))
    if not w.closed:
        w.finish(banner)


if __name__ == "__main__":
    main()
