"""Hazard map and A*: tested against made-up terrain, no MuJoCo model needed.

Costmap only ever calls probe.ground(x, y), so a tiny analytic stand-in for
TerrainProbe is enough to check the map and the planner in isolation.
"""

import math

import numpy as np
import pytest

from autonav import ROVER_RADIUS, Costmap


class FakeProbe:
    """ground(x, y) from a plain Python function."""

    def __init__(self, height):
        self.height = height

    def ground(self, x, y):
        return self.height(x, y)


def flat(x, y):
    return 0.0


def boulder(x, y, cx=6.0, cy=0.0, r=1.4, h=1.0):
    """A flat plain with one steep-sided boulder."""
    return h if math.hypot(x - cx, y - cy) < r else 0.0


def tilted(deg):
    return lambda x, y: math.tan(math.radians(deg)) * x


@pytest.fixture(scope="module")
def boulder_map():
    return Costmap(FakeProbe(boulder), span=30.0, res=0.25)


def test_flat_ground_is_all_free():
    cm = Costmap(FakeProbe(flat), span=10.0, res=0.25)
    assert np.isfinite(cm.cost).all()
    assert np.nanmax(cm.slope) == pytest.approx(0.0, abs=1e-9)


def test_slope_is_measured():
    cm = Costmap(FakeProbe(tilted(10.0)), span=10.0, res=0.25)
    assert np.median(cm.slope) == pytest.approx(10.0, abs=0.1)
    # a plane has no roughness and no obstacles, only slope
    assert np.median(cm.rough) == pytest.approx(0.0, abs=1e-6)


def test_too_steep_is_blocked():
    cm = Costmap(FakeProbe(tilted(35.0)), span=10.0, res=0.25)   # limit is 30
    assert not np.isfinite(cm.cost).any()


def test_boulder_is_blocked_with_rover_margin(boulder_map):
    cm = boulder_map
    i, j = cm.to_idx(6.0, 0.0)
    assert not np.isfinite(cm.cost[i, j])
    # the margin keeps the whole rover off it: still blocked a rover radius out
    i, j = cm.to_idx(6.0 + 1.4 + ROVER_RADIUS * 0.9, 0.0)
    assert not np.isfinite(cm.cost[i, j])
    # but open ground well clear of it is free
    i, j = cm.to_idx(0.0, 6.0)
    assert np.isfinite(cm.cost[i, j])


def test_astar_goes_around_the_boulder(boulder_map):
    path = boulder_map.plan((0.0, 0.0), (13.0, 0.0))
    assert path[0] == pytest.approx((0.0, 0.0), abs=boulder_map.res)
    assert path[-1] == pytest.approx((13.0, 0.0), abs=boulder_map.res)
    # Densify the waypoint polyline. Simplification may shave the dilated
    # margin by up to its 0.35 m tolerance, so check the physical rule the
    # README states instead: the rover centre stays at least rover radius +
    # boulder radius (1.6 + 1.4 = 3.0 m) from the boulder centre.
    pts = []
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, 40):
            pts.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    closest = min(math.hypot(x - 6.0, y) for x, y in pts)
    assert closest >= ROVER_RADIUS + 1.4


def test_simplify_collapses_a_straight_line():
    cm = Costmap(FakeProbe(flat), span=4.0, res=0.25)
    line = [(x, 0.0) for x in np.linspace(0, 1.5, 7)]
    assert cm.simplify(line) == [line[0], line[-1]]
