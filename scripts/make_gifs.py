#!/usr/bin/env python3
"""
make_gifs.py -- the animations on the README.

    python scripts/make_gifs.py            # everything (~5 min)
    python scripts/make_gifs.py route      # top-down drive on the hazard map
    python scripts/make_gifs.py rock       # rocker-bogie climbing a rock (2D + MuJoCo)
    python scripts/make_gifs.py steering   # top-down steering diagram (PNG)
    python scripts/make_gifs.py --resim    # rerun the sims instead of using out/ cache

Writes images/anim_*_light.gif and images/anim_*_dark.gif (the README shows the
one matching the viewer's GitHub theme) and images/mujoco_rock.gif.

The MuJoCo clip needs OpenGL. On a headless Linux box, install OSMesa
(apt install libosmesa6) and run with MUJOCO_GL=osmesa.
"""

import math
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m2020_drive import Mobility, hold_pose   # noqa: E402
from terrain_probe import TerrainProbe, WHEEL_RADIUS   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
IMG = os.path.join(ROOT, "images")
FPS = 25

THEMES = {
    "light": dict(bg="#FFFFFF", panel="#F6F8FA", ground="#D0D7DE", line="#8C959F",
                  text="#1F2328", sub="#57606A", route="#CF222E", trail="#1A7F37",
                  body="#57606A", wheel="#1F2328", rocker="#D9822B", bogie="#2B7BB9",
                  rock="#A0735A", blocked="#CF222E"),
    "dark": dict(bg="#0D1117", panel="#161B22", ground="#21262D", line="#6E7681",
                 text="#E6EDF3", sub="#8B949E", route="#FF7B72", trail="#3FB950",
                 body="#8B949E", wheel="#E6EDF3", rocker="#F0883E", bogie="#58A6FF",
                 rock="#B08A6E", blocked="#F85149"),
}

WHEEL_BODIES = ["wheel_left_front", "wheel_left_middle", "wheel_left_rear",
                "wheel_right_front", "wheel_right_middle", "wheel_right_rear"]


def _bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def _jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def _yaw(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _tilt(d):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    return math.degrees(math.acos(max(-1.0, min(1.0, R.reshape(3, 3)[2, 2]))))


# ---------------------------------------------------------------------------
# 1. Autonomous route on the real Jezero terrain, seen from above
# ---------------------------------------------------------------------------

def simulate_route(model="course_jezero_boulder.xml", start=(0.0, 0.0), goal=(18.0, 6.0),
                   span=40.0, snap_s=2.5, max_minutes=30.0):
    from autonav import Costmap, WaypointDriver

    probe = TerrainProbe(model)
    cm = Costmap(probe, centre=start, span=span, res=0.25)
    path = cm.plan(start, goal)
    m, d = probe.model, probe.data

    mujoco.mj_resetDataKeyframe(m, d, 1)
    mujoco.mj_forward(m, d)
    top = max(b["hi"] for b in probe.wheel_boxes(start[0], start[1], 0.0, pad=0.22).values())
    d.qpos[0:3] = [start[0], start[1], top + 0.12]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    hold_pose(m, d)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    wheels = [_bid(m, n) for n in WHEEL_BODIES]
    drv = WaypointDriver(m, d)
    frames, t0, next_snap, idx = [], d.time, 0.0, 1
    odo, last = 0.0, d.qpos[:2].copy()

    def snap():
        frames.append(dict(t=d.time - t0, x=float(d.qpos[0]), y=float(d.qpos[1]),
                           yaw=_yaw(d.qpos[3:7]), tilt=_tilt(d), odo=odo,
                           wheels=np.array([d.xpos[w][:2] for w in wheels])))

    while idx < len(path) and d.time - t0 < max_minutes * 60:
        drv.mode = "turn"
        reached = False
        while not reached and d.time - t0 < max_minutes * 60:
            reached = drv.step_toward(*path[idx])
            for _ in range(10):
                mujoco.mj_step(m, d)
            odo += float(np.linalg.norm(d.qpos[:2] - last))
            last = d.qpos[:2].copy()
            if d.time - t0 >= next_snap:
                snap()
                next_snap += snap_s
        idx += 1
    snap()
    return dict(frames=frames, path=np.array(path), goal=goal, reached=idx >= len(path),
                cost=cm.cost, blocked=cm.blocked, Z=cm.Z, ax=cm.ax, centre=cm.centre)


def _save_gif(fig, update, n, out):
    """Render each frame, map all of them onto one palette, and save.

    A shared palette with no dithering means pixels that don't change between
    frames stay byte-identical, so the GIF only stores what moved."""
    from PIL import Image

    frames, pal = [], None
    for i in range(n):
        update(i)
        fig.canvas.draw()
        im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        if pal is None:
            pal = im.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames.append(im.quantize(palette=pal, dither=Image.Dither.NONE))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 / FPS,
                   loop=0, optimize=True, disposal=1)
    print(f"wrote {os.path.relpath(out, ROOT)}  ({os.path.getsize(out) / 1e6:.1f} MB)")


def _rover_shapes(f):
    """Chassis rectangle and six wheel rectangles, world xy, for one frame."""
    c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
    R = np.array([[c, -s], [s, c]])

    def rect(cx, cy, L, W, rot=R):
        pts = np.array([[-L, -W], [L, -W], [L, W], [-L, W]]) / 2
        return pts @ rot.T + [cx, cy]

    body = rect(f["x"], f["y"], 2.0, 1.4)
    wheels = [rect(wx, wy, 0.52, 0.40) for wx, wy in f["wheels"]]
    nose = np.array([f["x"], f["y"]]) + R @ [1.25, 0.0]
    return body, wheels, nose


def gif_route(run, theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon

    T = THEMES[theme]
    fr, path = run["frames"], run["path"]
    ax_m = run["ax"]
    cx, cy = run["centre"]
    ext = [cx + ax_m[0], cx + ax_m[-1], cy + ax_m[0], cy + ax_m[-1]]

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=100)
    fig.patch.set_facecolor(T["bg"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
    ax.set_facecolor(T["bg"])
    cmap = LinearSegmentedColormap.from_list("cost", [T["bg"], T["ground"], T["line"]])
    cost = np.where(run["blocked"], np.nan, run["cost"]).T
    ax.imshow(cost, origin="lower", extent=ext, cmap=cmap, interpolation="bilinear",
              vmin=0, vmax=np.nanpercentile(cost, 98))
    blk = np.where(run["blocked"].T, 1.0, np.nan)
    ax.imshow(blk, origin="lower", extent=ext, cmap=LinearSegmentedColormap.from_list(
        "b", [T["blocked"], T["blocked"]]), alpha=0.35, interpolation="nearest")

    straight = np.array([path[0], path[-1]])
    ax.plot(straight[:, 0], straight[:, 1], ":", color=T["sub"], lw=1.3)
    ax.plot(path[:, 0], path[:, 1], "--", color=T["route"], lw=1.6)
    ax.plot(path[:, 0], path[:, 1], "o", color=T["route"], ms=4)
    ax.plot(*run["goal"], marker="*", color=T["route"], ms=16, mec=T["bg"])
    trail, = ax.plot([], [], color=T["trail"], lw=2.6, solid_capstyle="round")
    body = Polygon(np.zeros((4, 2)), closed=True, fc=T["body"], ec=T["wheel"], lw=1, zorder=5)
    wheels = [Polygon(np.zeros((4, 2)), closed=True, fc=T["wheel"], ec="none", zorder=6)
              for _ in range(6)]
    nose, = ax.plot([], [], "o", color=T["trail"], ms=4, zorder=7)
    for p in [body] + wheels:
        ax.add_patch(p)
    hud = ax.text(0.015, 0.975, "", transform=ax.transAxes, va="top", ha="left",
                  color=T["text"], fontsize=10.5, family="monospace",
                  bbox=dict(fc=T["panel"], ec=T["ground"], boxstyle="round,pad=0.4"))

    xs = np.r_[path[:, 0], [f["x"] for f in fr]]
    ys = np.r_[path[:, 1], [f["y"] for f in fr]]
    pad = 4.0
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    w = max(x1 - x0, (y1 - y0) * 8.4 / 4.9)
    h = w * 4.9 / 8.4
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(mx - w / 2, mx + w / 2)
    ax.set_ylim(my - h / 2, my + h / 2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Autonomous drive on the real Jezero terrain (top view)", color=T["text"],
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.915, "red = too steep/rough to enter   - - planned route   "
             "··· straight line   ━ driven", color=T["sub"], fontsize=9.5)

    tx, ty = [], []

    def update(i):
        f = fr[i]
        tx.append(f["x"])
        ty.append(f["y"])
        trail.set_data(tx, ty)
        b, ws, n = _rover_shapes(f)
        body.set_xy(b)
        for p, wv in zip(wheels, ws):
            p.set_xy(wv)
        nose.set_data([n[0]], [n[1]])
        hud.set_text(f"time   {f['t'] / 60:4.1f} min\ndriven {f['odo']:5.1f} m\n"
                     f"tilt   {f['tilt']:5.2f}°")

    _save_gif(fig, update, len(fr), os.path.join(IMG, f"anim_route_{theme}.gif"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Rocker-bogie climbing a rock: 2D kinematic side view + MuJoCo render
# ---------------------------------------------------------------------------

ROCK = dict(x=3.2, y=-1.12, a=0.45, b=0.45, h=0.24)  # half-buried, under the right wheels
NEAR, FAR = "right", "left"   # the camera side (on the rock) and the far side


def _rock_model():
    import demo

    rock = (f'<geom name="rock" type="ellipsoid" size="{ROCK["a"]} {ROCK["b"]} {ROCK["h"]}" '
            f'pos="{ROCK["x"]} {ROCK["y"]} 0" contype="2" conaffinity="1" condim="4" '
            f'friction="1.2 0.02 0.001" rgba="0.55 0.38 0.27 1"/>')
    return demo.load(rock)


def simulate_rock(distance=5.4, speed=0.04, snap_s=0.5):
    m = _rock_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 1)
    hold_pose(m, d)   # keep the arm stowed while driving
    mob = Mobility(m)
    for _ in range(1000):
        mujoco.mj_step(m, d)
    x_start = float(d.qpos[0])
    mob.arc(d.ctrl, speed, None)

    jL, jR, jB = (_jid(m, n) for n in ("LEFT_DIFFERENTIAL", "RIGHT_DIFFERENTIAL", "LEFT_BOGIE"))
    jBR = _jid(m, "RIGHT_BOGIE")
    w = {n: _bid(m, n) for n in WHEEL_BODIES}
    chassis = _bid(m, "chassis")
    frames, next_snap = [], 0.0
    t0 = d.time
    while d.qpos[0] - x_start < distance and d.time - t0 < distance / speed * 2:
        mujoco.mj_step(m, d)
        if d.time - t0 >= next_snap:
            next_snap += snap_s
            R = np.array(d.xmat[chassis]).reshape(3, 3)
            frames.append(dict(
                t=d.time - t0, qpos=d.qpos.copy(),
                chassis=np.array(d.xpos[chassis]), R=R,
                pivot_rocker={s: np.array(d.xanchor[j]) for s, j in (("left", jL), ("right", jR))},
                pivot_bogie={s: np.array(d.xanchor[j]) for s, j in (("left", jB), ("right", jBR))},
                wheel={n: np.array(d.xpos[b]) for n, b in w.items()},
                rocker_deg={"left": math.degrees(d.qpos[m.jnt_qposadr[jL]]),
                            "right": math.degrees(d.qpos[m.jnt_qposadr[jR]])},
                bogie_deg=math.degrees(d.qpos[m.jnt_qposadr[jBR if NEAR == "right" else jB]]),
                pitch_deg=math.degrees(math.atan2(-R[2, 0], R[0, 0])),
                tilt=_tilt(d)))
    return dict(frames=frames, model=m)


# Chassis box corners in the chassis frame (side view, x forward, z up).
_BODY = np.array([[-1.0, 0.60], [0.95, 0.60], [0.95, 1.12], [-1.0, 1.12]])


def gif_rock(run, theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    T = THEMES[theme]
    fr = run["frames"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=100)
    fig.patch.set_facecolor(T["bg"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01)
    ax.set_facecolor(T["bg"])
    ax.set_aspect("equal")
    ax.axis("off")

    xs = np.linspace(-5, 12, 400)
    ax.fill_between(xs, 0, -1, color=T["ground"], lw=0, zorder=0)
    ax.plot(xs, 0 * xs, color=T["line"], lw=1.4, zorder=1)
    rx = np.linspace(-ROCK["a"], ROCK["a"], 80)
    ax.fill_between(ROCK["x"] + rx, 0, ROCK["h"] * np.sqrt(1 - (rx / ROCK["a"]) ** 2),
                    color=T["rock"], lw=0, zorder=1)
    ax.text(ROCK["x"], -0.18, f"{ROCK['h'] * 100:.0f} cm rock", ha="center", va="top",
            color=T["sub"], fontsize=9.5)

    # right side drawn faint behind, left side bold in front
    art = {}
    for side, a, z in ((FAR, 0.28, 2), (NEAR, 1.0, 4)):
        art[side] = dict(
            rocker=ax.plot([], [], "-", color=T["rocker"], lw=4, alpha=a, zorder=z,
                           solid_capstyle="round")[0],
            bogie=ax.plot([], [], "-", color=T["bogie"], lw=4, alpha=a, zorder=z,
                          solid_capstyle="round")[0],
            pivots=ax.plot([], [], "o", color=T["text"], ms=5, alpha=a, zorder=z + 1)[0],
            wheels=[ax.add_patch(Circle((0, 0), WHEEL_RADIUS, fc=T["wheel"] if side == NEAR
                                        else "none", ec=T["wheel"], lw=1.5, alpha=a, zorder=z))
                    for _ in range(3)])
    body = ax.add_patch(Polygon(_BODY, closed=True, fc=T["body"], ec="none", alpha=0.55,
                                zorder=3))
    lab = dict(rocker=ax.text(0, 0, "rocker", color=T["rocker"], fontsize=10,
                              fontweight="bold", zorder=6),
               bogie=ax.text(0, 0, "bogie", color=T["bogie"], fontsize=10,
                             fontweight="bold", zorder=6))
    hud = ax.text(0.70, 0.95, "", transform=ax.transAxes, ha="left", va="top",
                  color=T["text"], fontsize=10, family="monospace",
                  bbox=dict(fc=T["panel"], ec=T["ground"], boxstyle="round,pad=0.4"))
    fig.suptitle("Rocker-bogie climbing a rock (2D side view)",
                 color=T["text"], fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.885, "Faded = left side, still on flat ground. The two rockers "
             "tilt equal and opposite through the differential.", color=T["sub"], fontsize=9.5)

    names = {"left": ("wheel_left_front", "wheel_left_middle", "wheel_left_rear"),
             "right": ("wheel_right_front", "wheel_right_middle", "wheel_right_rear")}
    cam = [None]

    def update(i):
        f = fr[i]
        for side in ("left", "right"):
            F, M, Rr = (f["wheel"][n][[0, 2]] for n in names[side])
            pr = f["pivot_rocker"][side][[0, 2]]
            pb = f["pivot_bogie"][side][[0, 2]]
            A = art[side]
            A["rocker"].set_data([F[0], pr[0], pb[0]], [F[1], pr[1], pb[1]])
            A["bogie"].set_data([M[0], pb[0], Rr[0]], [M[1], pb[1], Rr[1]])
            A["pivots"].set_data([pr[0], pb[0]], [pr[1], pb[1]])
            for c, p in zip(A["wheels"], (F, M, Rr)):
                c.center = (p[0], p[1])
            if side == NEAR:
                lab["rocker"].set_position((0.5 * (F[0] + pr[0]) + 0.18, 0.5 * (F[1] + pr[1])))
                lab["bogie"].set_position((pb[0] - 0.22, pb[1] - 0.32))
        Rx = f["R"][[0, 2]][:, [0, 2]]
        body.set_xy(_BODY @ Rx.T + f["chassis"][[0, 2]])
        x = f["chassis"][0]
        cam[0] = x if cam[0] is None else 0.8 * cam[0] + 0.2 * x
        ax.set_xlim(cam[0] - 3.4, cam[0] + 3.4)
        ax.set_ylim(-0.45, 2.75)
        hud.set_text(f"right rocker {f['rocker_deg']['right']:+5.1f}°\n"
                     f"left rocker  {f['rocker_deg']['left']:+5.1f}°\n"
                     f"right bogie  {f['bogie_deg']:+5.1f}°\n"
                     f"body pitch   {f['pitch_deg']:+5.1f}°")

    _save_gif(fig, update, len(fr), os.path.join(IMG, f"anim_rock_{theme}.gif"))
    plt.close(fig)


def mujoco_rock(run, width=760, height=340):
    """The same rock crossing, rendered with the real rover model."""
    from PIL import Image, ImageDraw, ImageFont

    m = run.get("model") or _rock_model()
    d = mujoco.MjData(m)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # A fixed camera: only the rover moves between frames, so the GIF stays small.
    xs = [f["chassis"][0] for f in run["frames"]]
    cam.distance, cam.elevation, cam.azimuth = 5.4, -7.0, 90.0
    cam.lookat[:] = [0.5 * (min(xs) + max(xs)), 0.0, 0.7]
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    frames, pal = [], None
    with mujoco.Renderer(m, height, width) as r:
        for f in run["frames"]:
            d.qpos[:] = f["qpos"]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0   # mirror floor bands badly in a GIF
            im = Image.fromarray(r.render())
            ImageDraw.Draw(im).text((14, 12), "MuJoCo: right wheels over a "
                                    f"{ROCK['h'] * 100:.0f} cm rock", font=font, fill="white",
                                    stroke_width=3, stroke_fill="black")
            if pal is None:
                pal = im.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
            frames.append(im.quantize(palette=pal, dither=Image.Dither.NONE))
    out = os.path.join(IMG, "mujoco_rock.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 / FPS,
                   loop=0, optimize=True, disposal=1)
    print(f"wrote {os.path.relpath(out, ROOT)}  ({os.path.getsize(out) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 3. Steering kinematics: where each wheel points, and how fast it spins
# ---------------------------------------------------------------------------

def fig_steering(theme, arc_radius=3.0):
    """Top view of the two turning modes, computed from Mobility's wheel geometry.

    Every wheel must roll about the same point (the instantaneous centre of
    rotation, ICR): it points at right angles to the line from the ICR, and
    spins in proportion to its distance from it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    T = THEMES[theme]
    m = mujoco.MjModel.from_xml_path(os.path.join(MODELS, "m2020_perseverance.xml"))
    wheels = Mobility(m).wheel_xy
    cases = [("Arc: turning left on a 3 m radius", np.array([0.0, arc_radius])),
             ("Turn in place", np.array([0.0, 0.0]))]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), dpi=150)
    fig.patch.set_facecolor(T["bg"])
    for ax, (title, icr) in zip(axes, cases):
        ax.set_facecolor(T["bg"])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=T["text"], fontsize=12, fontweight="bold")
        ax.add_patch(Polygon([[-1.0, -0.7], [0.95, -0.7], [0.95, 0.7], [-1.0, 0.7]],
                             closed=True, fc=T["ground"], ec=T["line"], lw=1))
        ax.annotate("", xy=(0.9, 0), xytext=(-0.4, 0),
                    arrowprops=dict(arrowstyle="-|>", color=T["sub"], lw=1.2))
        ax.text(0.25, -0.3, "front", color=T["sub"], fontsize=9, ha="center")
        dist = {k: np.linalg.norm(v - icr) for k, v in wheels.items()}
        slow = min(dist.values())
        for k, (wx, wy) in wheels.items():
            r = np.array([wx, wy]) - icr
            vel = np.array([-r[1], r[0]])              # CCW about the ICR
            ang = math.atan2(vel[1], vel[0])
            if ang > math.pi / 2:
                ang -= math.pi
            if ang < -math.pi / 2:
                ang += math.pi
            c, sn = math.cos(ang), math.sin(ang)
            box = np.array([[-0.26, -0.11], [0.26, -0.11], [0.26, 0.11], [-0.26, 0.11]])
            ax.add_patch(Polygon(box @ np.array([[c, -sn], [sn, c]]).T + [wx, wy], closed=True,
                                 fc=T["rocker"] if "M" not in k[1] else T["bogie"], ec="none"))
            ax.plot([icr[0], wx], [icr[1], wy], ":", color=T["line"], lw=0.9)
            ratio = dist[k] / slow
            ax.text(wx + (0.42 if wx >= 0 else -0.42), wy + (0.26 if wy > 0 else -0.34),
                    f"{ratio:.2f}×", color=T["text"], fontsize=9.5,
                    ha="left" if wx >= 0 else "right")
        ax.plot(*icr, "o", color=T["route"], ms=7)
        ax.text(icr[0] + 0.15, icr[1] + 0.12, "turn centre", color=T["route"], fontsize=9.5)
        ax.set_xlim(-2.4, 2.4)                          # same scale in both panels
        ax.set_ylim(-1.9, arc_radius + 0.6)
    fig.text(0.5, 0.03, "Orange = steerable corner wheels, blue = fixed middle wheels. "
             "Numbers = how fast each wheel spins compared with the slowest one.",
             ha="center", color=T["sub"], fontsize=9.5)
    out = os.path.join(IMG, f"fig_steering_{theme}.png")
    fig.savefig(out, facecolor=T["bg"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)}")


def _cached(name, fn):
    """Simulations are slow and deterministic: keep the result in out/."""
    import pickle

    path = os.path.join(ROOT, "out", f"{name}.pkl")
    if os.path.exists(path) and "--resim" not in sys.argv:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    run = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep = {k: v for k, v in run.items() if k != "model"}
    with open(path, "wb") as fh:
        pickle.dump(keep, fh)
    return run


def main(which):
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(IMG, exist_ok=True)
    if "rock" in which:
        print("simulating the rock crossing ...", flush=True)
        run = _cached("rock_run", simulate_rock)
        f = run["frames"]
        print(f"  {len(f)} frames, peak left rocker {max(abs(x['rocker_deg']['left']) for x in f):.1f}°, "
              f"peak bogie {max(abs(x['bogie_deg']) for x in f):.1f}°", flush=True)
        for th in THEMES:
            gif_rock(run, th)
        try:
            mujoco_rock(run)
        except Exception as exc:   # no OpenGL available
            print(f"  skipped the MuJoCo clip ({exc})")
    if "steering" in which:
        for th in THEMES:
            fig_steering(th)
    if "route" in which:
        print("planning and driving the Jezero route ...", flush=True)
        run = _cached("route_run", simulate_route)
        print(f"  {len(run['frames'])} frames, reached goal: {run['reached']}", flush=True)
        for th in THEMES:
            gif_route(run, th)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or ["rock", "route", "steering"])
