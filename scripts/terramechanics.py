#!/usr/bin/env python3
"""
terramechanics.py -- optional soft-soil wheel model (Bekker / Janosi-Hanamoto).

MuJoCo treats the ground as rigid: a wheel either grips (up to its Coulomb
friction) or slides. Real rovers on loose regolith do neither. The wheel sinks
in, has to keep compacting soil in front of it, and its grip depends on how
much it slips. This module adds those effects on top of MuJoCo's contacts,
without changing the model files:

  sinkage       z    Bekker pressure-sinkage for a rigid wheel:
                     z = [3W / ((3 - n) b (kc/b + kphi) sqrt(D))]^(2/(2n+1))
  compaction    Rc   the work of pressing that rut, as a force opposing rolling:
                     Rc = b (kc/b + kphi) z^(n+1) / (n + 1)
  traction      H    Mohr-Coulomb soil strength, mobilised by slip i over the
                     contact length L (Janosi-Hanamoto):
                     H = (c A + W tan(phi)) (1 - K/(i L) (1 - exp(-i L / K)))

Each step it measures every wheel's normal load W and slip i from the live
contacts, sets that wheel's friction coefficient to H/W (so grip builds with
slip, as it does in sand), and applies Rc at the hub against the direction
of travel. Wheels slip before they climb, and slip runs away on steep sand.

Limitations, stated up front: the rut is not geometric (wheels still ride on
the rigid surface, only its resistance is modelled); no bulldozing, multi-pass
or grouser-specific terms; lateral grip uses the same coefficient as
longitudinal; soil parameters are textbook values, not measured at Jezero.

    from terramechanics import SoftSoil
    soil = SoftSoil(model, "lunar_bekker")       # or "dry_sand", or SoftSoil(model, n=..)
    while ...:
        soil.apply(data)                          # before every mj_step
        mujoco.mj_step(model, data)
    soil.wheels                                   # per-wheel W, slip, sinkage, mu

    python scripts/terramechanics.py              # slope sweep: rigid vs soft soil
"""

import math
import os
import sys

import numpy as np
import mujoco

from m2020_drive import DRIVE, WHEEL_RADIUS, Mobility, hold_pose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

# Bekker parameters: n (-), kc (N/m^(n+1)), kphi (N/m^(n+2)), c (Pa), phi (deg),
# K shear deformation modulus (m). n, kc, kphi, c, phi as tabulated in J. Y.
# Wong, "Theory of Ground Vehicles" (terrain values table); K is EST, a
# typical value for loose sands. None of these were measured on Mars.
SOILS = {
    # Bekker's lunar soil values, the set most often reused for Martian
    # regolith in rover mobility studies
    "lunar_bekker": dict(n=1.0, kc=1.4e3, kphi=8.2e5, c=170.0, phi=35.0, K=0.018),
    # dry sand, Land Locomotion Laboratory measurements
    "dry_sand": dict(n=1.1, kc=0.99e3, kphi=1528.43e3, c=1040.0, phi=28.0, K=0.025),
}


class SoftSoil:
    """Per-wheel Bekker sinkage/compaction and slip-dependent traction."""

    def __init__(self, model, soil="lunar_bekker", mu_min=0.05, mu_max=None,
                 tau=0.05, **override):
        p = dict(SOILS[soil]) if soil else {}
        p.update(override)
        missing = {"n", "kc", "kphi", "c", "phi", "K"} - set(p)
        if missing:
            raise ValueError(f"soil parameters missing: {sorted(missing)}")
        self.p = p
        self.m = model
        self.mu_min = mu_min
        self.mu_max = mu_max if mu_max is not None else math.tan(math.radians(p["phi"])) + 0.3
        self.tau = tau

        self.wheels = {}
        for j in DRIVE:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            bid = model.jnt_bodyid[jid]
            geoms = [g for g in range(model.ngeom)
                     if model.geom_bodyid[g] == bid and model.geom_contype[g]]
            # wheel width from the tyre cylinder (type 5), as built from JPL's mesh
            cyl = [g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
            b = 2.0 * model.geom_size[cyl[0], 1] if cyl else 0.33
            self.wheels[j] = dict(jid=jid, bid=bid, geoms=np.array(geoms),
                                  dof=model.jnt_dofadr[jid], b=b,
                                  mu=float(model.geom_friction[geoms[0], 0]),
                                  W=0.0, slip=0.0, sinkage=0.0, Rc=0.0, v=0.0)
        self._geom_wheel = {}
        for j, w in self.wheels.items():
            for g in w["geoms"]:
                self._geom_wheel[int(g)] = j
        self._f = np.zeros(6)
        self._vel = np.zeros(6)

    # -- soil mechanics, per wheel -------------------------------------
    def sinkage(self, W, b, D=2 * WHEEL_RADIUS):
        n, kc, kphi = self.p["n"], self.p["kc"], self.p["kphi"]
        if W <= 0:
            return 0.0
        z = (3 * W / ((3 - n) * b * (kc / b + kphi) * math.sqrt(D))) ** (2 / (2 * n + 1))
        return min(z, WHEEL_RADIUS)

    def compaction(self, z, b):
        n, kc, kphi = self.p["n"], self.p["kc"], self.p["kphi"]
        return b * (kc / b + kphi) * z ** (n + 1) / (n + 1)

    def thrust(self, W, z, b, slip):
        c, phi, K = self.p["c"], math.radians(self.p["phi"]), self.p["K"]
        L = WHEEL_RADIUS * math.acos(max(-1.0, 1.0 - z / WHEEL_RADIUS))
        A = b * L
        j = abs(slip) * L                       # shear displacement at the rear
        if j < 1e-9:
            return 0.0
        return (c * A + W * math.tan(phi)) * (1 - K / j * (1 - math.exp(-j / K)))

    # -- per step --------------------------------------------------------
    def apply(self, data):
        m = self.m
        load = {j: 0.0 for j in self.wheels}
        normal = {j: np.zeros(3) for j in self.wheels}
        for i in range(data.ncon):
            con = data.contact[i]
            j = self._geom_wheel.get(int(con.geom1)) or self._geom_wheel.get(int(con.geom2))
            if j is None:
                continue
            mujoco.mj_contactForce(m, data, i, self._f)
            fn = self._f[0]
            nrm = con.frame[:3].copy()
            # orient the normal from the ground into the wheel
            if self._geom_wheel.get(int(con.geom1)) == j:
                nrm = -nrm
            load[j] += fn
            normal[j] += fn * nrm

        dt = m.opt.timestep
        for j, w in self.wheels.items():
            data.xfrc_applied[w["bid"], :3] = 0.0
            W = load[j]
            w["W"] = W
            if W < 1.0:
                w["slip"] = w["sinkage"] = w["Rc"] = 0.0
                continue
            n_hat = normal[j] / (np.linalg.norm(normal[j]) + 1e-12)
            axle = data.xaxis[w["jid"]]
            t = np.cross(axle, n_hat)
            t /= np.linalg.norm(t) + 1e-12            # rolling direction
            mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY,
                                     w["bid"], self._vel, 0)
            v = float(self._vel[3:] @ t)               # hub speed along it
            rim = float(data.qvel[w["dof"]]) * WHEEL_RADIUS
            denom = max(abs(rim), abs(v), 1e-3)
            slip = (rim - v) / denom                   # >0 driving, <0 braking

            z = self.sinkage(W, w["b"])
            Rc = self.compaction(z, w["b"])
            H = self.thrust(W, z, w["b"], slip)
            target = min(self.mu_max, max(self.mu_min, H / W))
            w["mu"] += (target - w["mu"]) * min(1.0, dt / self.tau)
            m.geom_friction[w["geoms"], 0] = w["mu"]

            # compaction resistance at the hub, against the travel direction
            data.xfrc_applied[w["bid"], :3] = -Rc * math.tanh(v / 0.005) * t
            w.update(slip=slip, sinkage=z, Rc=Rc, v=v)

    def reset_friction(self, mu=0.9):
        for w in self.wheels.values():
            self.m.geom_friction[w["geoms"], 0] = mu
            w["mu"] = mu


# ---------------------------------------------------------------- demo
def drive_on_slope(xml, slope_deg, soil=None, seconds=60.0):
    """Drive straight uphill at full speed on a plane tilted by slope_deg
    (gravity rotated, so the flat model stands in for a uniform slope).
    Returns (speed m/h, mean slip, mean sinkage m)."""
    m = mujoco.MjModel.from_xml_path(xml)
    g = 3.721
    th = math.radians(slope_deg)
    m.opt.gravity[:] = (-g * math.sin(th), 0.0, -g * math.cos(th))   # uphill = +x
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stowed"))
    hold_pose(m, d)
    s = SoftSoil(m, soil) if soil else None
    mob = Mobility(m)

    def step(n):
        for _ in range(n):
            if s:
                s.apply(d)
            mujoco.mj_step(m, d)

    step(int(3.0 / m.opt.timestep))                     # settle
    mob.arc(d.ctrl, 0.042, None)
    step(int(10.0 / m.opt.timestep))                    # spin up
    x0, t0 = d.qpos[0], d.time
    slips, sinks = [], []
    for _ in range(int(seconds / 0.5)):
        step(int(0.5 / m.opt.timestep))
        if s:
            slips.append(np.mean([w["slip"] for w in s.wheels.values()]))
            sinks.append(np.mean([w["sinkage"] for w in s.wheels.values()]))
    speed = (d.qpos[0] - x0) / (d.time - t0) * 3600
    if not s:
        return speed, 0.0, 0.0
    return speed, float(np.mean(slips)), float(np.mean(sinks))


def main():
    xml = os.path.join(MODELS, sys.argv[1] if len(sys.argv) > 1
                       else "m2020_perseverance_smooth.xml")
    print(f"model {os.path.basename(xml)}: straight uphill at full speed, 60 s each")
    print(f"\n  {'slope':>5}   {'rigid ground':>13}   {'soft soil (lunar_bekker)':>36}")
    print(f"  {'deg':>5}   {'m/h':>6} {'travel':>6}   {'m/h':>6} {'travel':>6} "
          f"{'slip':>6} {'sinkage':>9}")
    v0 = None
    for slope in (0, 5, 10, 15, 20, 25):
        r = drive_on_slope(xml, slope)
        s = drive_on_slope(xml, slope, "lunar_bekker")
        v0 = v0 or r[0]
        print(f"  {slope:5.0f}   {r[0]:6.1f} {1 - r[0] / v0:6.2f}   {s[0]:6.1f} "
              f"{1 - s[0] / v0:6.2f} {s[1]:6.2f} {s[2] * 100:7.1f} cm")
    print("\n  travel = travel reduction, 1 - speed / (rigid flat speed); it also")
    print("  counts the drive motors slowing under load. slip = mean wheel slip")
    print("  ratio (rim speed - hub speed) / rim speed, measured at the contacts.")


if __name__ == "__main__":
    main()
