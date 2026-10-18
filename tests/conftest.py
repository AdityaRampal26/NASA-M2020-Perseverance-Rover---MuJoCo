"""Shared pytest setup: put scripts/ on the import path and load the rover once."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODEL_XML = os.path.join(ROOT, "models", "m2020_perseverance.xml")
MESHES = os.path.join(ROOT, "models", "assets", "CHASSIS.obj")


@pytest.fixture(scope="session")
def rover():
    """The compiled flat-ground rover. Needs the JPL meshes from setup."""
    if not os.path.exists(MESHES):
        pytest.skip("JPL meshes not built yet: run setup.sh / setup.ps1 first")
    import mujoco
    return mujoco.MjModel.from_xml_path(MODEL_XML)
