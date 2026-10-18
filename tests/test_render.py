"""Offscreen render of all 13 flight cameras.

The physics checks can't see a broken mesh, a missing texture or a camera
pointed at the sky. This renders every camera and checks each image actually
shows something. Images aren't compared pixel-for-pixel against references,
because software and GPU OpenGL differ in the last few bits.

Skipped when no OpenGL context is available, unless REQUIRE_RENDER=1 is set
(CI sets it on Linux, where it renders with OSMesa).
"""

import os

import numpy as np
import pytest

from m2020_drive import arm_pose, mast_pose

W, H = 160, 120


@pytest.fixture(scope="module")
def frames(rover):
    import mujoco
    d = mujoco.MjData(rover)
    mujoco.mj_resetDataKeyframe(rover, d, 1)          # stowed, like demo.py
    mast_pose(rover, d.ctrl, az=0.0, el=np.radians(-12))
    arm_pose(rover, d.ctrl, 0.329, 0.785, -1.046, -1.901, 0.345)
    for _ in range(3000):
        mujoco.mj_step(rover, d)
    try:
        r = mujoco.Renderer(rover, H, W)
    except Exception as e:                            # no OpenGL context
        if os.environ.get("REQUIRE_RENDER") == "1":
            raise
        pytest.skip(f"no OpenGL context: {e}")
    out = {}
    for i in range(rover.ncam):
        name = mujoco.mj_id2name(rover, mujoco.mjtObj.mjOBJ_CAMERA, i)
        r.update_scene(d, camera=name)
        out[name] = r.render().copy()
    r.close()
    return out


def test_all_thirteen_cameras_render(frames):
    assert len(frames) == 13
    for name, img in frames.items():
        assert img.shape == (H, W, 3), name


def test_every_camera_sees_something(frames):
    # A camera looking at nothing (sky, or inside a mesh) gives a near-uniform
    # image. Every flight camera here looks at terrain or the rover itself.
    for name, img in frames.items():
        assert img.std() > 4.0, f"{name} image is nearly uniform"


def test_cameras_differ(frames):
    imgs = list(frames.values())
    same = sum(np.array_equal(a, b) for k, a in enumerate(imgs)
               for b in imgs[k + 1:])
    assert same == 0, "two cameras rendered identical images"
