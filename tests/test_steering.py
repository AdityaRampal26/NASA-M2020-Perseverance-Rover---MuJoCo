"""Six-wheel steering: every wheel must roll about the same turn centre.

These check the numbers the README quotes (the 1.34x turn-in-place ratio, the
roughly 2x outer/inner ratio on a 3 m arc) straight from m2020_drive.Mobility.
"""

import math

import numpy as np
import pytest

from m2020_drive import DRIVE, STEER, WHEEL_RADIUS, Mobility


def commands(model, mode, *args):
    """Run one Mobility mode and return {wheel: (steer_rad, rate_rad_s)}."""
    mob = Mobility(model)
    ctrl = np.zeros(model.nu)
    getattr(mob, mode)(ctrl, *args)
    out = {}
    for j in DRIVE:
        rate = ctrl[mob.act[j.lower()]]
        s = STEER.get(j)
        steer = ctrl[mob.act[s.lower()]] if s else 0.0
        out[j] = (steer, rate)
    return mob, out


def test_straight_is_uniform(rover):
    _, cmd = commands(rover, "arc", 0.04, None)
    rates = [r for _, r in cmd.values()]
    assert np.allclose(rates, 0.04 / WHEEL_RADIUS)
    assert all(s == 0.0 for s, _ in cmd.values())


def test_point_turn_ratio(rover):
    mob, cmd = commands(rover, "point_turn", 0.05)
    corner = abs(cmd["LF_DRIVE"][1])
    middle = abs(cmd["LM_DRIVE"][1])
    # README: corners 1.59 m out, middle wheels 1.18 m, so 1.34x faster
    assert corner / middle == pytest.approx(1.34, abs=0.01)
    # left and right sides spin opposite ways
    assert np.sign(cmd["LM_DRIVE"][1]) == -np.sign(cmd["RM_DRIVE"][1])


def test_arc_wheels_share_one_turn_centre(rover):
    radius = 3.0
    mob, cmd = commands(rover, "arc", 0.04, radius)
    icr = np.array([0.0, radius])
    omega = 0.04 / radius
    for j in DRIVE:
        r = mob.wheel_xy[j] - icr
        # wheel speed is proportional to its distance from the turn centre
        assert abs(cmd[j][1]) * WHEEL_RADIUS == pytest.approx(
            omega * np.linalg.norm(r), rel=1e-6)
        if j in STEER:
            # the wheel's rolling direction is perpendicular to the radius
            # (model steer is clockwise-positive, hence the minus sign)
            psi = -cmd[j][0]
            heading = np.array([math.cos(psi), math.sin(psi)])
            assert abs(heading @ r) / np.linalg.norm(r) < 1e-6


def test_arc_outer_wheels_about_twice_inner(rover):
    _, cmd = commands(rover, "arc", 0.04, 3.0)        # left turn
    outer = abs(cmd["RM_DRIVE"][1])
    inner = abs(cmd["LM_DRIVE"][1])
    assert 1.8 < outer / inner < 2.4                  # README: "about twice"


def test_fold_reverses_drive():
    psi, v = Mobility._fold(math.radians(120), 1.0)
    assert psi == pytest.approx(math.radians(-60))
    assert v == -1.0
    psi, v = Mobility._fold(math.radians(-100), 1.0)
    assert psi == pytest.approx(math.radians(80))
    assert v == -1.0
    assert Mobility._fold(0.3, 1.0) == (0.3, 1.0)
