#!/usr/bin/env python3
"""
teleop.py -- drive the M2020 Perseverance model from the keyboard.

    macOS :  mjpython scripts/teleop.py            <- must be mjpython, not python
    Linux :  python scripts/teleop.py

    mjpython scripts/teleop.py --course            drive the Jezero terrain course
    mjpython scripts/teleop.py --flat              flat ground
    mjpython scripts/teleop.py --realtime 4        run 4x faster than real time

The rover's real top speed is 152 m/h -- about 4 cm/s -- so the simulation
runs at a speed multiplier by default, otherwise you would be watching paint
dry. The physics is unchanged; only the wall-clock pacing is scaled.

CONTROLS
    W / S      forward / reverse   (hold to build speed, tap X to stop)
    A / D      steer left / right  (smaller radius the longer you hold)
    Q / E      turn in place, left / right
    Z / C      crab left / right (translate without turning)
    X          all stop
    R          reset to the start

    1 / 2 / 3  arm: stowed / mid-deploy / work pose
    I / K      mast up / down
    J / L      mast left / right
    M          mast back to centre

    TAB        MuJoCo control panel      [ / ]   cycle cameras
    SPACE      pause                     ESC     quit
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from m2020_drive import Mobility, arm_pose, mast_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")   # MJCF + assets
OUT = os.path.join(ROOT, "out")         # generated results
HERE = MODELS

# GLFW key codes
K_W, K_A, K_S, K_D = 87, 65, 83, 68
K_Q, K_E, K_Z, K_C = 81, 69, 90, 67
K_X, K_R, K_M = 88, 82, 77
K_I, K_K, K_J, K_L = 73, 75, 74, 76
K_1, K_2, K_3 = 49, 50, 51

MAX_SPEED = 0.161 * 0.2625          # m/s, the published 152 m/h
# Steer levels 0..5; level 0 is straight, 5 is the tightest arc the corner
# wheels can hold. Tap A/D to step through them.
RADII = [None, 20.0, 12.0, 7.0, 4.0, 2.5]

ARM_POSES = {
    K_1: ("stowed", (0.0, -0.90, -2.55, 0.0, 0.0)),
    K_2: ("mid-deploy", (0.0, -0.30, -1.90, 0.0, 0.0)),
    K_3: ("work pose", (0.329, 0.785, -1.046, -1.901, 0.345)),
}


class Rover:
    def __init__(self, model, data):
        self.m, self.d = model, data
        self.mob = Mobility(model, data)
        self.reset_state()

    def reset_state(self):
        self.speed = 0.0            # m/s, signed
        self.steer = 0              # -5..+5, positive = left
        self.yaw_rate = 0.0         # rad/s, for point turns
        self.crab_angle = None      # rad, for crab mode
        self.mode = "arc"
        self.mast_az = 0.0
        self.mast_el = math.radians(-10)
        self.arm = ARM_POSES[K_1][1]
        self.status = "ready"

    # -- input ---------------------------------------------------------
    def key(self, code):
        s = self.speed
        if code == K_W:
            self.mode = "arc"
            self.speed = min(MAX_SPEED, s + MAX_SPEED * 0.25)
        elif code == K_S:
            self.mode = "arc"
            self.speed = max(-MAX_SPEED, s - MAX_SPEED * 0.25)
        elif code == K_A:
            self.mode = "arc"
            self.steer = min(5, self.steer + 1)
        elif code == K_D:
            self.mode = "arc"
            self.steer = max(-5, self.steer - 1)
        elif code == K_Q:
            self.mode = "spin"
            self.yaw_rate = 0.035
        elif code == K_E:
            self.mode = "spin"
            self.yaw_rate = -0.035
        elif code == K_Z:
            self.mode = "crab"
            self.crab_angle = math.radians(60)
            self.speed = MAX_SPEED
        elif code == K_C:
            self.mode = "crab"
            self.crab_angle = math.radians(-60)
            self.speed = MAX_SPEED
        elif code == K_X:
            self.speed = 0.0
            self.yaw_rate = 0.0
            self.mode = "arc"
            self.steer = 0
        elif code == K_R:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 1)
            self.reset_state()
            self.status = "reset"
        elif code in ARM_POSES:
            name, pose = ARM_POSES[code]
            self.arm = pose
            self.status = f"arm -> {name}"
        elif code == K_I:
            self.mast_el = min(math.radians(85), self.mast_el + math.radians(8))
        elif code == K_K:
            self.mast_el = max(math.radians(-85), self.mast_el - math.radians(8))
        elif code == K_J:
            self.mast_az += math.radians(12)
        elif code == K_L:
            self.mast_az -= math.radians(12)
        elif code == K_M:
            self.mast_az, self.mast_el = 0.0, math.radians(-10)

    @property
    def radius(self):
        """Signed turn radius, metres; None = straight. +ve = left."""
        if self.steer == 0:
            return None
        return math.copysign(RADII[abs(self.steer)], self.steer)

    # -- output --------------------------------------------------------
    def apply(self):
        c = self.d.ctrl
        if self.mode == "spin":
            self.mob.point_turn(c, self.yaw_rate)
        elif self.mode == "crab":
            self.mob.crab(c, self.speed, self.crab_angle)
        else:
            self.mob.arc(c, self.speed, self.radius)
        arm_pose(self.m, c, *self.arm)
        mast_pose(self.m, c, az=self.mast_az, el=self.mast_el)

    def hud(self):
        q = self.d.qpos[3:7]
        roll = math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                       1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1])))))
        v = float(np.linalg.norm(self.d.qvel[:2]))
        if self.mode == "spin":
            cmd = f"point turn {math.degrees(self.yaw_rate):+.1f} deg/s"
        elif self.mode == "crab":
            cmd = f"crab {math.degrees(self.crab_angle):+.0f} deg"
        else:
            r = "straight" if self.radius is None else f"R={self.radius:+.1f} m"
            cmd = f"{self.speed * 3600:+.0f} m/h  {r}"
        return (f"\r  {cmd:<30} | {v * 3600:5.0f} m/h "
                f"| xy ({self.d.qpos[0]:+6.1f},{self.d.qpos[1]:+6.1f}) "
                f"| roll{roll:+5.1f} pitch{pitch:+5.1f} "
                f"| {self.status:<16}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--course", action="store_true",
                   help="Jezero terrain course (default)")
    g.add_argument("--flat", action="store_true", help="flat ground")
    ap.add_argument("--model", default=None)
    ap.add_argument("--realtime", type=float, default=6.0,
                    help="wall-clock speed multiplier (default 6x)")
    args = ap.parse_args()

    if args.model:
        xml = args.model
    elif args.flat:
        xml = "m2020_perseverance.xml"
    else:
        xml = "course_jezero.xml"
        if not os.path.exists(os.path.join(HERE, xml)):
            print("course_jezero.xml not found -- run:  python scripts/make_course.py")
            xml = "m2020_perseverance.xml"
    path = os.path.join(HERE, xml)

    # mujoco.viewer needs the Cocoa main thread on macOS, which the `mjpython`
    # launcher provides. Detect it via mujoco.viewer._MJPYTHON, which the
    # launcher initialises -- NOT via sys.executable, which mjpython leaves
    # pointing at the plain python binary.
    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", 0) is None:
        print("On macOS the MuJoCo viewer must run under mjpython:\n"
              f"    mjpython {os.path.basename(__file__)} "
              f"{' '.join(sys.argv[1:])}\n")
        return

    print(f"loading {xml} ...")
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 1)          # stowed
    rover = Rover(m, d)
    print(__doc__.split("CONTROLS")[1])
    print(f"  physics at {args.realtime:g}x wall clock "
          f"(real rover tops out at 152 m/h)\n")

    try:
        viewer_ctx = mujoco.viewer.launch_passive(m, d, key_callback=rover.key)
    except RuntimeError as e:
        print(f"\ncould not open the viewer: {e}")
        if sys.platform == "darwin":
            print("try:  mjpython " + os.path.basename(__file__))
        return

    with viewer_ctx as v:
        v.cam.distance = 9.0
        v.cam.elevation = -18
        v.cam.azimuth = 130
        last_hud = 0.0
        while v.is_running():
            t0 = time.time()
            rover.apply()
            mujoco.mj_step(m, d)
            # keep the camera following the rover
            v.cam.lookat[:] = d.qpos[:3] + np.array([0, 0, 0.6])
            v.sync()
            if time.time() - last_hud > 0.15:
                sys.stdout.write(rover.hud())
                sys.stdout.flush()
                last_hud = time.time()
            lag = m.opt.timestep / args.realtime - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
    print("\ndone.")


if __name__ == "__main__":
    main()
