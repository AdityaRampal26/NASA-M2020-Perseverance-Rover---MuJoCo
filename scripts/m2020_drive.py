#!/usr/bin/env python3
"""
m2020_drive.py -- rocker-bogie mobility kinematics for the MuJoCo M2020 model.

Handles the three mobility modes the real rover uses:

    straight / arc   drive along a circular arc of a given radius (Ackermann
                     for six wheels: every wheel is steered and spun so that
                     all six share one instantaneous centre of rotation)
    point turn       rotate in place about the rover centre
    crab             translate in any direction without changing heading

SIGN CONVENTION
    Joint angles in the model match the JPL URDF exactly.  Because the JPL
    rover frame is Z-DOWN, a POSITIVE steer angle turns the wheel CLOCKWISE
    seen from above (i.e. to the right).  The body frame of the model itself
    is Z-up / X-forward / Y-left.  This module takes care of the conversion:
    you give it ordinary right-handed commands (positive yaw rate = left).
"""

import math

import numpy as np
import mujoco

WHEEL_RADIUS = 0.2625          # m, from the published 1.65 m per revolution
MAX_WHEEL_RATE = 0.161         # rad/s, from the published 152 m/h top speed

DRIVE = ["LF_DRIVE", "LM_DRIVE", "LR_DRIVE", "RF_DRIVE", "RM_DRIVE", "RR_DRIVE"]
STEER = {"LF_DRIVE": "LF_STEER", "LR_DRIVE": "LR_STEER",
         "RF_DRIVE": "RF_STEER", "RR_DRIVE": "RR_STEER"}   # middle wheels fixed


class Mobility:
    """Six-wheel steering/drive resolver bound to a compiled MuJoCo model."""

    def __init__(self, model, data=None):
        self.m = model
        self.act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                    for i in range(model.nu)}
        # Wheel positions in the chassis frame, measured from the model itself
        # rather than hard-coded, so this stays correct if the model changes.
        # Measure on a PRIVATE MjData. An earlier version reset whatever
        # data it was handed, which silently wiped the caller's keyframe and
        # ctrl -- the arm would deploy itself the moment you constructed a
        # Mobility. Never mutate the caller's state to take a measurement.
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        Rc = d.xmat[chassis].reshape(3, 3)
        pc = d.xpos[chassis]
        self.wheel_xy = {}
        for j in DRIVE:
            bid = model.jnt_bodyid[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            local = Rc.T @ (d.xpos[bid] - pc)
            self.wheel_xy[j] = np.array([local[0], local[1]])
        self.steer_range = 1.745       # rad, matches the model

    # -- low level ------------------------------------------------------
    def _set(self, ctrl, name, value):
        i = self.act.get(name.lower())
        if i is not None:
            ctrl[i] = value

    def stop(self, ctrl):
        for j in DRIVE:
            self._set(ctrl, j, 0.0)

    @staticmethod
    def _fold(psi, v):
        """Bring a steer angle into [-pi/2, pi/2].

        Folding by pi points the wheel the other way round, so the drive
        direction must be reversed with it -- forgetting this makes every
        wheel spin the same way and the rover just scrubs in place.
        """
        while psi > math.pi / 2:
            psi -= math.pi
            v = -v
        while psi < -math.pi / 2:
            psi += math.pi
            v = -v
        return psi, v

    def _resolve(self, ctrl, icr, omega):
        """Steer + spin all six wheels about an instantaneous centre of
        rotation `icr` (chassis frame) at angular rate `omega` (CCW +ve)."""
        for j in DRIVE:
            r = self.wheel_xy[j] - icr
            psi = math.atan2(r[0], -r[1])          # CCW-positive maths angle
            v = omega * np.linalg.norm(r)          # tangential speed
            psi, v = self._fold(psi, v)
            s = STEER.get(j)
            if s is not None:
                # model steer is CLOCKWISE-positive (JPL Z-down) -> negate
                self._set(ctrl, s, float(np.clip(-psi, -self.steer_range,
                                                 self.steer_range)))
            # middle wheels have no steer actuator and simply scrub if psi != 0
            self._set(ctrl, j, v / WHEEL_RADIUS)

    # -- modes ----------------------------------------------------------
    def arc(self, ctrl, speed, radius=None):
        """Drive along an arc.

        speed  : m/s at the rover centre; negative drives backwards.
        radius : turn radius in metres, POSITIVE = centre to the LEFT
                 (a left-hand turn).  None or inf = straight ahead.
        """
        if radius is None or not np.isfinite(radius):
            for j in DRIVE:
                self._set(ctrl, j, speed / WHEEL_RADIUS)
            for s in STEER.values():
                self._set(ctrl, s, 0.0)
            return

        self._resolve(ctrl, np.array([0.0, radius]), speed / radius)

    def point_turn(self, ctrl, yaw_rate):
        """Rotate in place about the rover centre.  yaw_rate > 0 = turn LEFT.

        Note the four corner wheels sit further from the centre than the two
        middle wheels, so they must be spun proportionally faster; commanding
        all six at the same rate just scrubs."""
        self._resolve(ctrl, np.zeros(2), yaw_rate)

    def crab(self, ctrl, speed, heading):
        """Translate without rotating.  heading in rad, 0 = forward,
        +ve = to the left."""
        psi, v = self._fold(heading, speed)
        for j in DRIVE:
            s = STEER.get(j)
            if s is not None:
                self._set(ctrl, s, float(np.clip(-psi, -self.steer_range,
                                                 self.steer_range)))
            self._set(ctrl, j, v / WHEEL_RADIUS)

    # -- helpers --------------------------------------------------------
    def settle_steer(self, data, seconds=6.0):
        """Run the sim with drives held at zero until the steer actuators
        have reached their targets (the real rover also stops to steer)."""
        n = int(seconds / self.m.opt.timestep)
        for j in DRIVE:
            self._set(data.ctrl, j, 0.0)
        for _ in range(n):
            mujoco.mj_step(self.m, data)


def arm_pose(model, ctrl, j1=0.0, j2=-0.9, j3=-2.55, j4=0.0, j5=0.0):
    """Command the 5-DOF robotic arm (radians, JPL joint order)."""
    act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
           for i in range(model.nu)}
    for name, v in [("joint1_enc", j1), ("joint2_enc", j2), ("joint3_enc", j3),
                    ("joint4_enc", j4), ("joint5_enc", j5)]:
        if name in act:
            ctrl[act[name]] = v


# The RSM encoders do not zero at "looking forward along the horizon":
# elevation zero points the mast straight DOWN.  These offsets were measured
# from the model by solving for a level, forward-pointing Navcam boresight.
RSM_EL_HORIZON = -1.5446       # rad
RSM_AZ_FORWARD = 0.0117        # rad


def mast_pose(model, ctrl, az=0.0, el=0.0, raw=False):
    """Point the Remote Sensing Mast (Mastcam-Z / Navcam / SuperCam).

    az : rad, 0 = straight ahead, positive = to the LEFT
    el : rad, 0 = horizon, positive = UP
    raw: True to pass raw encoder values through unchanged
    """
    act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
           for i in range(model.nu)}
    if not raw:
        az = RSM_AZ_FORWARD - az
        el = RSM_EL_HORIZON - el
    if "rsm_az_enc" in act:
        ctrl[act["rsm_az_enc"]] = az
    if "rsm_el_enc" in act:
        ctrl[act["rsm_el_enc"]] = el


def hold_pose(model, data, skip_drives=True):
    """Command every position actuator to the joint angle it is currently at.

    MuJoCo's position servos drive toward ctrl, and ctrl is ZERO after a reset.
    Zero is the URDF pose -- arm straight out -- so loading the "stowed"
    keyframe and then stepping makes the arm swing out and sag under gravity,
    no matter what the keyframe said. Calling this straight after a reset
    latches the servos onto the pose the keyframe actually set.

    Drive joints are velocity servos, so their ctrl stays 0 (stationary).
    """
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if skip_drives and name.endswith("_drive"):
            data.ctrl[i] = 0.0
            continue
        jid = model.actuator_trnid[i, 0]
        if jid < 0:
            continue
        data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]
