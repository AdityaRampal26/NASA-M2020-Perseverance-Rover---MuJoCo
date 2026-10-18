#!/usr/bin/env python3
"""
demo.py -- exercise the MuJoCo M2020 Perseverance model.

    python scripts/demo.py viewer          interactive viewer (needs a display)
    python scripts/demo.py drive           straight / arc / point turn / crab
    python scripts/demo.py obstacle        drive over a rock field
    python scripts/demo.py arm             stow -> deploy -> work pose
    python scripts/demo.py cameras         render every flight camera
    python scripts/demo.py validate        the full physical validation suite
"""

import math
import os
import sys

import numpy as np
import mujoco

from m2020_drive import Mobility, arm_pose, mast_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "out")
XML = os.path.join(MODELS, "m2020_perseverance.xml")
SCENE = os.path.join(MODELS, "_scene.xml")   # scratch scene; must sit beside assets/
DRIVES = ["lf_drive", "lm_drive", "lr_drive", "rf_drive", "rm_drive", "rr_drive"]


def load(extra_geoms=""):
    src = open(XML).read()
    if extra_geoms:
        src = src.replace('<geom name="terrain"', extra_geoms + '\n    <geom name="terrain"')
    open(SCENE, "w").write(src)
    return mujoco.MjModel.from_xml_path(SCENE)


def yaw_of(d):
    q = d.qpos[3:7]
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def settle(m, d, mob, seconds=10.0):
    """The real rover stops to steer before driving; so do we."""
    keep = d.ctrl.copy()
    mob.stop(d.ctrl)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    d.ctrl[:] = keep


# ---------------------------------------------------------------- viewer
def cmd_viewer():
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 1)          # stowed
    mob = Mobility(m)
    mob.arc(d.ctrl, 0.04, None)
    print("driving forward; press Tab for the control panel, '[' / ']' to cycle cameras")
    mujoco.viewer.launch(m, d)


# ---------------------------------------------------------------- drive
def cmd_drive():
    m = mujoco.MjModel.from_xml_path(XML)
    mob = Mobility(m)
    print(f"{'manoeuvre':<24} {'result':<46}")
    print("-" * 72)

    def run(name, setup, seconds):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 1)
        for _ in range(1000):
            mujoco.mj_step(m, d)
        setup(d.ctrl)
        settle(m, d, mob)
        p0, y0, t0 = d.qpos[:3].copy(), yaw_of(d), d.time
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
        dp = d.qpos[:3] - p0
        dy = math.degrees((yaw_of(d) - y0 + math.pi) % (2 * math.pi) - math.pi)
        print(f"{name:<24} {np.linalg.norm(dp[:2]):.2f} m travelled, "
              f"{dy:+.1f} deg yaw, {d.time - t0:.0f} s")
        return dp, dy

    run("straight 0.04 m/s", lambda c: mob.arc(c, 0.04, None), 60)
    run("arc, radius 4 m left", lambda c: mob.arc(c, 0.04, 4.0), 100)
    run("arc, radius 4 m right", lambda c: mob.arc(c, 0.04, -4.0), 100)
    run("point turn left", lambda c: mob.point_turn(c, 0.03), 90)
    run("crab 90 deg (sideways)", lambda c: mob.crab(c, 0.04, math.pi / 2), 80)


# ------------------------------------------------------------- obstacle
def cmd_obstacle():
    rocks = "".join(
        f'<geom name="rock{i}" type="ellipsoid" size="{0.18+0.05*(i%3):.2f} '
        f'{0.22+0.04*(i%2):.2f} {0.11+0.035*(i%4):.3f}" '
        f'pos="{2.0+1.5*i:.2f} {(-1)**i*0.55:.2f} {0.05+0.02*(i%3):.2f}" '
        f'euler="0 0 {0.4*i:.2f}" contype="2" conaffinity="1" condim="4" '
        f'friction="1.2 0.02 0.001" rgba="0.42 0.31 0.24 1"/>'
        for i in range(7))
    m = load(rocks)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 1)
    mob = Mobility(m)
    for _ in range(1000):
        mujoco.mj_step(m, d)
    mob.arc(d.ctrl, 0.04, None)
    J = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
    qa = lambda n: d.qpos[m.jnt_qposadr[J[n]]]
    print(f"{'t[s]':>6} {'x[m]':>7} {'z[m]':>7} {'roll':>7} {'pitch':>7} "
          f"{'rockL':>7} {'rockR':>7} {'bogieL':>7}")
    for i in range(300001):
        mujoco.mj_step(m, d)
        if i % 30000 == 0:
            q = d.qpos[3:7]
            roll = math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                           1 - 2 * (q[1] ** 2 + q[2] ** 2)))
            pit = math.degrees(math.asin(max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1])))))
            print(f"{d.time:6.0f} {d.qpos[0]:7.3f} {d.qpos[2]:7.3f} {roll:7.2f} {pit:7.2f} "
                  f"{math.degrees(qa('LEFT_DIFFERENTIAL')):7.2f} "
                  f"{math.degrees(qa('RIGHT_DIFFERENTIAL')):7.2f} "
                  f"{math.degrees(qa('LEFT_BOGIE')):7.2f}")
    print("\nrocker L and R stay equal and opposite: that is the differential working.")


# ------------------------------------------------------------------ arm
def cmd_arm():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 1)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "drill_tip")
    poses = [("stowed", (0, -0.90, -2.55, 0, 0)),
             ("mid-deploy", (0, -0.30, -1.90, 0, 0)),
             ("work pose", (0.329, 0.785, -1.046, -1.901, 0.345))]
    for name, p in poses:
        arm_pose(m, d.ctrl, *p)
        mast_pose(m, d.ctrl, az=0.0, el=-0.5)
        for _ in range(12000):
            mujoco.mj_step(m, d)
        print(f"{name:<12} drill tip at {np.round(d.site_xpos[site], 3)} "
              f"(height above ground {d.site_xpos[site][2]:.3f} m)")


# -------------------------------------------------------------- cameras
def cmd_cameras():
    import imageio.v3 as iio
    rocks = "".join(
        f'<geom name="rk{i}" type="ellipsoid" size="{0.2+0.08*(i%4):.2f} '
        f'{0.25+0.06*(i%3):.2f} {0.13+0.05*(i%3):.2f}" '
        f'pos="{1.6+1.1*i:.2f} {((-1)**i)*(0.4+0.25*(i%4)):.2f} {0.06+0.02*(i%3):.2f}" '
        f'euler="0 0 {0.5*i:.2f}" contype="2" conaffinity="1" condim="4" '
        f'friction="1.2 0.02 0.001" rgba="0.42 0.31 0.24 1"/>'
        for i in range(9))
    m = load(rocks)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 1)
    mast_pose(m, d.ctrl, az=0.0, el=math.radians(-12))
    arm_pose(m, d.ctrl, 0.329, 0.785, -1.046, -1.901, 0.345)
    for _ in range(12000):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, 420, 560)
    tiles, names = [], []
    for i in range(m.ncam):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
        r.update_scene(d, camera=n)
        tiles.append(r.render())
        names.append(n)
        print(f"  {n:22s} fovy={m.cam_fovy[i]:6.2f} deg")
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    os.makedirs(OUT, exist_ok=True)
    iio.imwrite(os.path.join(OUT, "render_cameras.png"), np.vstack(rows))
    print("wrote out/render_cameras.png")


# ------------------------------------------------------------- validate
def cmd_validate():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print("=" * 66)
    print("M2020 MuJoCo model -- physical validation")
    print("=" * 66)
    print(f"bodies {m.nbody}  joints {m.njnt}  dofs {m.nv}  actuators {m.nu}  "
          f"cameras {m.ncam}  geoms {m.ngeom}")
    print(f"total mass          {m.body_subtreemass[1]:8.2f} kg   (published 1025)")
    print(f"CoM height          {d.subtree_com[1][2]:8.3f} m    (EST)")
    tip = math.degrees(math.atan2(1.0625, d.subtree_com[1][2]))
    print(f"static tip angle    {tip:8.1f} deg  (published tolerance 45)")

    for _ in range(3000):
        mujoco.mj_step(m, d)
    f = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f)
        tot += f[0]
    print(f"settled contact force {tot:7.1f} N   (m*g = {1025 * 3.721:.1f} N)")

    A = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    x0, t0 = d.qpos[0], d.time
    for k in DRIVES:
        d.ctrl[A[k]] = 0.161
    for _ in range(30000):
        mujoco.mj_step(m, d)
    speed = (d.qpos[0] - x0) / (d.time - t0) * 3600
    print(f"flat top speed      {speed:8.1f} m/h  (published 152)")
    print("=" * 66)

    # Fail loudly (exit 1) when the model drifts from the published numbers,
    # so setup.sh and CI catch a regression instead of printing it. Bands are
    # wide enough for solver noise across MuJoCo versions and platforms.
    mass = m.body_subtreemass[1]
    checks = [
        ("total mass", abs(mass - 1025.0) <= 0.5, f"{mass:.2f} kg, want 1025 +/- 0.5"),
        ("tip angle", tip >= 45.0, f"{tip:.1f} deg, want >= 45 (NASA rating)"),
        ("contact force", abs(tot - 1025 * 3.721) <= 0.01 * 1025 * 3.721,
         f"{tot:.1f} N, want m*g +/- 1%"),
        ("top speed", abs(speed - 152.0) <= 3.0, f"{speed:.1f} m/h, want 152 +/- 3"),
    ]
    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:14s} {detail}")
    if failed:
        print(f"{len(failed)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    globals()[f"cmd_{cmd}"]()
