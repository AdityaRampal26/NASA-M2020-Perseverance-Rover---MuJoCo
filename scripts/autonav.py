#!/usr/bin/env python3
"""
autonav.py -- autonomous waypoint navigation on the real Jezero terrain.

This reproduces, at rover scale and in simulation, the loop JPL flew in
December 2025 when Perseverance drove 689 ft and 807 ft on AI-planned routes:

    orbital terrain data  ->  hazard map  ->  waypoint path
                          ->  validation in a digital twin  ->  drive it

Here the orbital data is the real HiRISE DTM of the landing site, the digital
twin is the MuJoCo rover, and the safety check settles the full rover model at
poses along the route and measures its tilt, suspension angles and belly
clearance (see terrain_probe.py).

The costmap follows the definition in Abcouwer et al.:

    "the cost of the cell may be a weighted sum of the tilt, the roughness
     (the average squared distance between a fitted plane and the points in
     the plane), and a minimum time needed to traverse a cell. The cell will
     be given an infinite cost if the tilt is extreme, plane-fitting fails,
     or if designated keep-in or keep-out zones are violated."

    python scripts/autonav.py --goal 25 10           # plan, validate, drive
    python scripts/autonav.py --goal 25 10 --plan-only
    python scripts/autonav.py --survey               # show the hazard map
"""

import argparse
import heapq
import json
import math
import os
import sys
import time

import numpy as np
import mujoco

from terrain_probe import TerrainProbe, WHEELS, LIMITS
from m2020_drive import (Mobility, MAX_WHEEL_RATE, WHEEL_RADIUS,
                         hold_pose)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")   # MJCF + assets
OUT = os.path.join(ROOT, "out")         # generated results


def _out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


# Rover footprint radius used for costmap dilation: half the diagonal across
# the wheelbase, so a cell is only free if the whole vehicle fits.
ROVER_RADIUS = 1.6

# Costmap weights (EST -- flight weights are not published).
W_TILT, W_ROUGH, W_STEP = 1.0, 30.0, 8.0


class Costmap:
    """Hazard map built from terrain, per the ENav costmap definition."""

    def __init__(self, probe, centre=(0.0, 0.0), span=80.0, res=0.25,
                 tilt_limit=None, rough_limit=0.01, step_limit=0.30):
        self.probe = probe
        self.res = res
        self.centre = np.array(centre, float)
        self.n = int(round(span / res)) + 1
        self.ax = np.linspace(-span / 2, span / 2, self.n)
        self.tilt_limit = tilt_limit or LIMITS["tilt_deg"]
        self.rough_limit = rough_limit
        self.step_limit = step_limit
        self._build()

    def _build(self):
        n = self.n
        print(f"building hazard map: {n}x{n} cells at {self.res:.2f} m ...")
        Z = np.full((n, n), np.nan)
        for i, dx in enumerate(self.ax):
            for j, dy in enumerate(self.ax):
                g = self.probe.ground(self.centre[0] + dx, self.centre[1] + dy)
                if g is not None:
                    Z[i, j] = g
        self.Z = Z

        # slope from the gradient
        gy, gx = np.gradient(np.nan_to_num(Z, nan=np.nanmean(Z)), self.res)
        self.slope = np.degrees(np.arctan(np.hypot(gx, gy)))

        # Roughness and obstacle height, both as residuals to a locally
        # fitted plane over a rover-sized window. Raw max-min would just
        # re-measure the slope (3 deg over 3.2 m is already 0.17 m) and
        # reject open ground; removing the plane isolates real obstacles.
        #
        # Closed form, so this vectorises. For a centred symmetric window the
        # least-squares plane z = a*dx + b*dy + c has
        #     c = mean(z),  a = <z,dx>/Sxx,  b = <z,dy>/Syy
        # and the mean squared residual is
        #     mean(z^2) - c^2 - a^2*Sxx/N - b^2*Syy/N
        # which is a handful of correlations instead of 100k lstsq calls.
        from scipy.ndimage import correlate, maximum_filter

        w = max(1, int(round(ROVER_RADIUS / self.res)))
        k = 2 * w + 1
        ii, jj = np.meshgrid(np.arange(-w, w + 1), np.arange(-w, w + 1),
                             indexing="ij")
        dx, dy = ii * self.res, jj * self.res
        N = k * k
        Sxx, Syy = float((dx ** 2).sum()), float((dy ** 2).sum())
        Zf = np.nan_to_num(Z, nan=float(np.nanmean(Z)))

        mean_z = correlate(Zf, np.ones((k, k)) / N, mode="nearest")
        mean_z2 = correlate(Zf ** 2, np.ones((k, k)) / N, mode="nearest")
        a_co = correlate(Zf, dx / Sxx, mode="nearest")
        b_co = correlate(Zf, dy / Syy, mode="nearest")

        mss = (mean_z2 - mean_z ** 2
               - (a_co ** 2) * (Sxx / N) - (b_co ** 2) * (Syy / N))
        self.rough = np.maximum(mss, 0.0)

        # obstacle height: how far a post sits off its own local plane, then
        # the worst such post anywhere in the footprint
        self.detrend = np.abs(Zf - mean_z)
        self.step = maximum_filter(self.detrend, size=k, mode="nearest")

        # infinite cost where any limit is exceeded or the fit failed
        bad = (~np.isfinite(Z) | ~np.isfinite(self.rough)
               | (self.slope > self.tilt_limit)
               | (self.rough > self.rough_limit)
               | (self.step > self.step_limit))
        # dilate by the rover radius: a cell is only free if the vehicle fits
        r = max(1, int(round(ROVER_RADIUS / self.res)))
        blocked = bad.copy()
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj > r * r:
                    continue
                blocked |= np.roll(np.roll(bad, di, 0), dj, 1)
        self.blocked = blocked

        cost = (W_TILT * self.slope
                + W_ROUGH * np.nan_to_num(self.rough, nan=1e3)
                + W_STEP * np.nan_to_num(self.step, nan=1e3))
        cost[blocked] = np.inf
        self.cost = cost
        free = np.isfinite(cost).sum()
        print(f"  {free}/{cost.size} cells traversable "
              f"({100*free/cost.size:.1f}%), "
              f"slope max {np.nanmax(self.slope):.1f} deg, "
              f"roughness max {np.nanmax(self.rough):.4f} m^2")

    # -- indexing ------------------------------------------------------
    def to_idx(self, x, y):
        i = int(round((x - self.centre[0] + self.ax[-1]) / self.res))
        j = int(round((y - self.centre[1] + self.ax[-1]) / self.res))
        return i, j

    def to_xy(self, i, j):
        return (self.centre[0] + self.ax[0] + i * self.res,
                self.centre[1] + self.ax[0] + j * self.res)

    def inside(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    # -- planning ------------------------------------------------------
    def plan(self, start, goal):
        """A* over the costmap. Returns a list of (x, y) waypoints."""
        si, sj = self.to_idx(*start)
        gi, gj = self.to_idx(*goal)
        if not (self.inside(si, sj) and self.inside(gi, gj)):
            raise SystemExit("start or goal is outside the map")
        if not np.isfinite(self.cost[gi, gj]):
            raise SystemExit("goal cell is not traversable")

        nbrs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)]
        h = lambda i, j: math.hypot(i - gi, j - gj) * self.res
        openq = [(h(si, sj), 0.0, (si, sj))]
        came, gscore = {}, {(si, sj): 0.0}
        seen = set()
        while openq:
            _, g, cur = heapq.heappop(openq)
            if cur in seen:
                continue
            seen.add(cur)
            if cur == (gi, gj):
                break
            ci, cj = cur
            for di, dj in nbrs:
                ni, nj = ci + di, cj + dj
                if not self.inside(ni, nj):
                    continue
                c = self.cost[ni, nj]
                if not np.isfinite(c):
                    continue
                step = math.hypot(di, dj) * self.res
                ng = g + step * (1.0 + c * 0.02)
                if ng < gscore.get((ni, nj), np.inf):
                    gscore[(ni, nj)] = ng
                    came[(ni, nj)] = cur
                    heapq.heappush(openq, (ng + h(ni, nj), ng, (ni, nj)))
        if (gi, gj) not in came and (si, sj) != (gi, gj):
            raise SystemExit("no traversable path to the goal")

        path, cur = [], (gi, gj)
        while cur != (si, sj):
            path.append(self.to_xy(*cur))
            cur = came[cur]
        path.append(self.to_xy(si, sj))
        path.reverse()
        return self.simplify(path)

    def simplify(self, path, tol=0.35):
        """Collapse collinear runs into waypoints (Douglas-Peucker)."""
        if len(path) < 3:
            return path
        p = np.array(path)
        keep = np.zeros(len(p), bool)
        keep[0] = keep[-1] = True

        def rdp(lo, hi):
            if hi <= lo + 1:
                return
            a, b = p[lo], p[hi]
            ab = b - a
            L = np.linalg.norm(ab)
            if L < 1e-9:
                return
            seg = p[lo + 1:hi] - a
            d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
            k = int(np.argmax(d))
            if d[k] > tol:
                keep[lo + 1 + k] = True
                rdp(lo, lo + 1 + k)
                rdp(lo + 1 + k, hi)

        rdp(0, len(p) - 1)
        return [tuple(v) for v in p[keep]]


def validate(probe, path, spacing=2.0, verbose=True):
    """Settle the rover at poses along the path and check each against LIMITS."""
    pts, checks = [], []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.array(a), np.array(b)
        L = np.linalg.norm(b - a)
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        for t in np.arange(0, L, spacing):
            pts.append((a + (b - a) * (t / L), yaw))
    pts.append((np.array(path[-1]),
                math.atan2(path[-1][1] - path[-2][1],
                           path[-1][0] - path[-2][0])))
    if verbose:
        print(f"validating {len(pts)} poses along the path "
              f"(digital-twin check) ...")
    worst = dict(tilt=0.0, clearance=9.9)
    n_bad = 0
    for k, (p, yaw) in enumerate(pts):
        r = probe.evaluate(p[0], p[1], yaw)
        if r is None:
            n_bad += 1
            checks.append(None)
            continue
        checks.append(r)
        worst["tilt"] = max(worst["tilt"], r["tilt_deg"])
        worst["clearance"] = min(worst["clearance"], r["clearance_m"])
        if not r["safe"]:
            n_bad += 1
    if verbose:
        print(f"  worst tilt {worst['tilt']:.2f} deg, "
              f"min clearance {worst['clearance']:.3f} m, "
              f"{n_bad} pose(s) failing limits")
    return checks, worst, n_bad


class WaypointDriver:
    """Drive a waypoint list the way a rover does: turn in place to the
    heading, then drive an arc. ENav commands the same two primitives."""

    def __init__(self, model, data, heading_tol=0.10, arrive_tol=0.9):
        self.m, self.d = model, data
        self.mob = Mobility(model, data)
        self.heading_tol = heading_tol
        self.arrive_tol = arrive_tol
        self.mode = "turn"

    def yaw(self):
        q = self.d.qpos[3:7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                          1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def step_toward(self, wx, wy):
        """One control update. Returns True when the waypoint is reached."""
        dx, dy = wx - self.d.qpos[0], wy - self.d.qpos[1]
        dist = math.hypot(dx, dy)
        if dist < self.arrive_tol:
            self.mob.stop(self.d.ctrl)
            return True
        err = math.atan2(dy, dx) - self.yaw()
        err = (err + math.pi) % (2 * math.pi) - math.pi

        if self.mode == "turn":
            if abs(err) < self.heading_tol:
                self.mode = "drive"
            else:
                rate = math.copysign(min(0.035, abs(err) * 0.12), err)
                self.mob.point_turn(self.d.ctrl, rate)
                return False
        if abs(err) > 0.45:                 # drifted badly -- re-align
            self.mode = "turn"
            return False
        # gentle arc correction while driving
        radius = None if abs(err) < 0.02 else max(3.0, 2.0 / max(abs(err), 1e-3))
        if radius is not None:
            radius = math.copysign(radius, err)
        self.mob.arc(self.d.ctrl, MAX_WHEEL_RATE * WHEEL_RADIUS, radius)
        return False


def drive_path(model, data, path, max_minutes=240.0, verbose=True):
    """Execute the path in the digital twin and log what actually happened."""
    drv = WaypointDriver(model, data)
    log = []
    t0 = data.time
    limit = max_minutes * 60.0
    idx = 1
    settle = int(2.0 / model.opt.timestep)
    for _ in range(settle):
        mujoco.mj_step(model, data)
    if verbose:
        print(f"driving {len(path)-1} legs ...")
    while idx < len(path) and (data.time - t0) < limit:
        wx, wy = path[idx]
        drv.mode = "turn"
        reached = False
        leg_start = data.time
        while not reached and (data.time - t0) < limit:
            reached = drv.step_toward(wx, wy)
            for _ in range(10):
                mujoco.mj_step(model, data)
            q = data.qpos[3:7]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            tilt = math.degrees(math.acos(max(-1, min(1, R.reshape(3, 3)[2, 2]))))
            log.append((data.time - t0, data.qpos[0], data.qpos[1],
                        data.qpos[2], tilt))
            if tilt > 45:
                print(f"  ABORT: tilt {tilt:.1f} deg at leg {idx}")
                return log, False
        if verbose:
            print(f"  leg {idx}/{len(path)-1} -> ({wx:6.1f},{wy:6.1f})  "
                  f"{data.time-leg_start:6.0f}s  "
                  f"pos ({data.qpos[0]:6.2f},{data.qpos[1]:6.2f})")
        idx += 1
    return log, idx >= len(path)



# ---------------------------------------------------------------- markers
PATH_RGBA = np.array([0.92, 0.41, 0.20, 1.0], np.float32)     # planned
TRACK_RGBA = np.array([0.11, 0.69, 0.48, 1.0], np.float32)    # driven
GOAL_RGBA = np.array([0.95, 0.82, 0.10, 1.0], np.float32)


def draw_route(scn, probe, path, track=None, goal=None, lift=0.35, reset=True):
    """Draw the planned route, the driven track and the goal into a scene.

    Points are lifted onto the terrain so the route lies on the ground
    instead of floating at the datum.

    reset=True for the viewer's user_scn, which is a dedicated overlay scene.
    reset=False for a Renderer's scene, which update_scene has ALREADY filled
    with the model's geoms -- zeroing ngeom there deletes the rover and you
    render an empty landscape with a path drawn on it.
    """
    if reset:
        scn.ngeom = 0

    def put(fn):
        if scn.ngeom >= scn.maxgeom:
            return False
        fn(scn.geoms[scn.ngeom])
        scn.ngeom += 1
        return True

    def on_ground(p):
        g = probe.ground(p[0], p[1])
        return np.array([p[0], p[1], (g if g is not None else 0.0) + lift])

    pts = [on_ground(p) for p in path]
    for a, b in zip(pts[:-1], pts[1:]):
        put(lambda g, a=a, b=b: (
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                PATH_RGBA),
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.05, a, b)))
    for p in pts:
        put(lambda g, p=p: mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.18, 0, 0]), p,
            np.eye(3).ravel(), PATH_RGBA))
    if goal is not None:
        gp = on_ground(goal) + np.array([0, 0, 0.9])
        put(lambda g: mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.12, 0.12, 1.2]), gp,
            np.eye(3).ravel(), GOAL_RGBA))
    if track is not None and len(track) > 1:
        step = max(1, len(track) // 240)
        tp = [np.array([t[0], t[1], t[2] + 0.05]) for t in track[::step]]
        for a, b in zip(tp[:-1], tp[1:]):
            put(lambda g, a=a, b=b: (
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    np.zeros(3), np.zeros(3),
                                    np.eye(3).ravel(), TRACK_RGBA),
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     0.035, a, b)))


def drive_viewer(probe, path, goal, realtime=8.0, max_minutes=240.0):
    """Watch the autonomous traverse in the MuJoCo viewer."""
    import mujoco.viewer
    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", 0) is None:
        print("On macOS the viewer must run under mjpython:\n"
              f"    mjpython {os.path.basename(__file__)} "
              f"{' '.join(sys.argv[1:])}\n")
        return None, False
    m, d = probe.model, probe.data
    drv = WaypointDriver(m, d)
    track, log = [], []
    idx, t0 = 1, d.time
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 14.0, -22, 130
        draw_route(v.user_scn, probe, path, None, goal)
        while v.is_running() and idx < len(path) and (d.time - t0) < max_minutes * 60:
            wx, wy = path[idx]
            drv.mode = "turn"
            reached = False
            while v.is_running() and not reached:
                reached = drv.step_toward(wx, wy)
                for _ in range(8):
                    mujoco.mj_step(m, d)
                track.append(d.qpos[:3].copy())
                log.append((d.time - t0, d.qpos[0], d.qpos[1], d.qpos[2],
                            tilt_of(d)))
                v.cam.lookat[:] = d.qpos[:3] + np.array([0, 0, 0.6])
                if len(track) % 25 == 0:
                    draw_route(v.user_scn, probe, path, track, goal)
                v.sync()
                time.sleep(max(0.0, m.opt.timestep * 8 / realtime))
            print(f"  leg {idx}/{len(path)-1} reached")
            idx += 1
    return np.array(log), idx >= len(path)


def record_drive(probe, path, goal, out="autonav_drive.mp4", fps=30,
                 realtime=60.0, width=1280, height=720, max_minutes=240.0):
    """Render the traverse offscreen to a video."""
    import imageio.v2 as iio
    m, d = probe.model, probe.data
    drv = WaypointDriver(m, d)
    r = mujoco.Renderer(m, height, width)
    scn_opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 9.0, -16, 135
    steps_per_frame = max(1, int(realtime / (fps * m.opt.timestep)))
    frames, track, log = [], [], []
    idx, t0 = 1, d.time
    w = iio.get_writer(_out(out), fps=fps,
                       macro_block_size=None, quality=7)
    try:
        while idx < len(path) and (d.time - t0) < max_minutes * 60:
            wx, wy = path[idx]
            drv.mode = "turn"
            reached = False
            while not reached and (d.time - t0) < max_minutes * 60:
                reached = drv.step_toward(wx, wy)
                for _ in range(steps_per_frame):
                    mujoco.mj_step(m, d)
                track.append(d.qpos[:3].copy())
                log.append((d.time - t0, d.qpos[0], d.qpos[1], d.qpos[2],
                            tilt_of(d)))
                cam.lookat[:] = d.qpos[:3] + np.array([0, 0, 0.6])
                cam.azimuth = 135 + 12 * math.sin((d.time - t0) / 240.0)
                r.update_scene(d, cam, scn_opt)
                draw_route(r.scene, probe, path, track, goal, reset=False)
                w.append_data(r.render())
                frames.append(1)
            print(f"  leg {idx}/{len(path)-1}  ({len(frames)} frames)", flush=True)
            idx += 1
    finally:
        w.close()
    print(f"wrote {out}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    return np.array(log), idx >= len(path)


def tilt_of(d):
    q = d.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return math.degrees(math.acos(max(-1, min(1, R.reshape(3, 3)[2, 2]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="course_jezero.xml")
    ap.add_argument("--goal", type=float, nargs=2, default=[25.0, 10.0])
    ap.add_argument("--start", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--span", type=float, default=80.0)
    ap.add_argument("--res", type=float, default=0.25,
                    help="costmap cell size; must be small enough to "
                         "resolve obstacles (0.25 m catches the rocks, "
                         "0.5 m steps straight over them)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=240.0)
    ap.add_argument("--viewer", action="store_true",
                    help="watch the traverse live (macOS: use mjpython)")
    ap.add_argument("--record", nargs="?", const="autonav_drive.mp4",
                    default=None, metavar="FILE",
                    help="render the traverse offscreen to a video")
    ap.add_argument("--realtime", type=float, default=None,
                    help="speed multiplier (viewer default 8, record 60)")
    args = ap.parse_args()

    probe = TerrainProbe(args.model)
    cm = Costmap(probe, centre=tuple(args.start), span=args.span, res=args.res)

    if args.survey:
        np.savez(_out("hazard_map.npz"),
                 Z=cm.Z, slope=cm.slope, rough=cm.rough, step=cm.step,
                 cost=cm.cost, blocked=cm.blocked, axis=cm.ax,
                 centre=cm.centre, res=cm.res)
        print("\nwrote hazard_map.npz  (Z, slope, rough, step, cost, blocked)")
        return

    print(f"\nplanning {tuple(args.start)} -> {tuple(args.goal)}")
    t = time.time()
    path = cm.plan(tuple(args.start), tuple(args.goal))
    L = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
    print(f"  {len(path)} waypoints, {L:.1f} m, planned in {time.time()-t:.2f}s")
    for k, p in enumerate(path):
        print(f"    wp{k:02d}  ({p[0]:7.2f}, {p[1]:7.2f})")

    if not args.no_validate:
        print()
        checks, worst, n_bad = validate(probe, path)
        if n_bad:
            print(f"  {n_bad} pose(s) violate limits -- this route "
                  f"should be rejected")
        else:
            print("  route passes the safety check at every sampled pose")

    if args.plan_only:
        json.dump(dict(path=[list(p) for p in path], length_m=L),
                  open(_out("planned_path.json"), "w"), indent=1)
        print("\nwrote planned_path.json")
        return

    print()
    model, data = probe.model, probe.data
    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)
    boxes = probe.wheel_boxes(args.start[0], args.start[1], 0.0, pad=0.22)
    top = max(b["hi"] for b in boxes.values())
    data.qpos[0:3] = [args.start[0], args.start[1], top + 0.12]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    hold_pose(model, data)          # keep the arm stowed for the traverse

    if args.viewer:
        log, ok = drive_viewer(probe, path, tuple(args.goal),
                               realtime=args.realtime or 8.0,
                               max_minutes=args.max_minutes)
        if log is None:
            return
    elif args.record:
        log, ok = record_drive(probe, path, tuple(args.goal), out=args.record,
                               realtime=args.realtime or 60.0,
                               max_minutes=args.max_minutes)
    else:
        log, ok = drive_path(model, data, path, args.max_minutes)
    log = np.array(log)
    travelled = float(np.sum(np.linalg.norm(np.diff(log[:, 1:3], axis=0), axis=1)))
    err = math.dist((data.qpos[0], data.qpos[1]), tuple(args.goal))
    print(f"\n{'='*62}")
    print(f"  {'REACHED GOAL' if ok else 'DID NOT COMPLETE'}")
    print(f"  planned      {L:7.1f} m over {len(path)-1} legs")
    print(f"  driven       {travelled:7.1f} m odometry")
    print(f"  sim time     {log[-1,0]/60:7.1f} min "
          f"({travelled/(log[-1,0]/3600):.0f} m/h average)")
    print(f"  final error  {err:7.2f} m from the goal")
    print(f"  max tilt     {log[:,4].max():7.2f} deg "
          f"(limit {LIMITS['tilt_deg']:.0f})")
    print(f"  elevation    {log[0,3]:.2f} -> {log[-1,3]:.2f} m")
    print(f"{'='*62}")
    np.savez(_out("drive_log.npz"), log=log,
             path=np.array(path), planned_m=L, driven_m=travelled)
    print("wrote drive_log.npz")


if __name__ == "__main__":
    main()
