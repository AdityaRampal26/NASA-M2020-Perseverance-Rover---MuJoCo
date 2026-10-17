#!/usr/bin/env python3
"""
build_m2020.py -- generate a MuJoCo MJCF model of the NASA/JPL Mars 2020
Perseverance rover from JPL's officially released URDF.

Source of geometry and kinematics:
    https://github.com/nasa-jpl/m2020-urdf-models   (NASA/JPL-Caltech, URS307049)

What this script does
---------------------
1. Parses rover/m2020.urdf -- the flight kinematic tree as released by the
   JPL RSVP (Rover Sequencing and Visualization Program) team.
2. Converts every mesh from glTF to OBJ, rotating 180 deg about +X to take the
   JPL rover frame (X fwd, Y right, Z down) into the MuJoCo/ROS convention
   (X fwd, Y left, Z up).  That rotation has determinant +1, so face winding
   and normals are preserved.
3. Emits MJCF with:
     - the exact JPL link/joint topology for the rocker-bogie mobility system,
       the 5-DOF robotic arm, the RSM (mast), the HGA and the turret;
     - a documented mass budget summing to the published 1025 kg;
     - inertia tensors computed from the actual mesh geometry;
     - primitive collision geometry (wheels, belly, rocker/bogie tubes, arm);
     - an equality constraint reproducing the rocker-bogie differential;
     - drive / steer / arm / mast / HGA actuators;
     - MuJoCo cameras derived from the 27 released CAHV/CAHVOR/CAHVORE
       flight camera calibrations.

Every number that is *not* traceable to JPL data is tagged EST in the output
XML comments.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------
# Frame conversion: JPL rover frame (X fwd, Y right, Z down)  ->  MuJoCo Z-up
# 180 deg rotation about +X.  det = +1.
# --------------------------------------------------------------------------
M = np.diag([1.0, -1.0, -1.0])


def flip_p(p):
    return M @ np.asarray(p, float)


def flip_R(R):
    return M @ R @ M


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_quat(R):
    """Rotation matrix -> MuJoCo (w, x, y, z) quaternion."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def f3(v, nd=6):
    return " ".join(f"{x:.{nd}g}" for x in np.asarray(v, float))


# --------------------------------------------------------------------------
# Mass budget.  Total is the published 1025 kg (NASA).
# Only the total and the MMRTG (45 kg, inside the chassis) are published
# figures; the split below is an engineering estimate and is marked EST.
# --------------------------------------------------------------------------
MASS = {
    # --- mobility: ~132 kg EST, consistent with MSL-class rocker-bogie ---
    "Body_WheelLeftFront": 8.0, "Body_WheelLeftMiddle": 8.0, "Body_WheelLeftRear": 8.0,
    "Body_WheelRightFront": 8.0, "Body_WheelRightMiddle": 8.0, "Body_WheelRightRear": 8.0,
    "Body_RockerLeft": 15.0, "Body_RockerRight": 15.0,
    "Body_BogieLeft": 11.0, "Body_BogieRight": 11.0,
    "Body_SteerLeftFront": 6.0, "Body_SteerLeftRear": 6.0,
    "Body_SteerRightFront": 6.0, "Body_SteerRightRear": 6.0,
    "Body_Differential": 8.0,
    # --- remote sensing mast (RSM): ~25 kg EST ---
    "Body_RSM_AZ": 10.0, "Body_RSM_EL": 15.0,
    # --- high gain antenna: ~15 kg EST ---
    "Body_HGA_AZ": 7.0, "Body_HGA_EL": 8.0,
    # --- robotic arm + sample caching turret: ~80 kg EST ---
    "Body_RA_Link1": 12.0, "Body_RA_Link2": 9.0, "Body_RA_Link3": 8.0,
    "Body_RA_Link4": 4.0, "Body_RA_Link5": 2.0,
    "Body_Turret": 40.0,
    "Body_CorerFeed": 3.0,
    "Body_StabilizerFar": 0.5, "Body_StabilizerNear": 0.5,
    "Body_WATSON_Cap": 0.2, "Body_SHERLOC_Cap": 0.2,
    # --- chassis: remainder (WEB, electronics, MMRTG 45 kg, instruments,
    #     adaptive caching assembly, sample tubes) ---
    "Body_Chassis": None,   # filled so the total is exactly 1025.0
}
TOTAL_MASS = 1025.0

# Chassis inertia: uniform solid box approximating the warm electronics box
# envelope (EST).  Chassis CoM height is the dominant stability parameter.
CHASSIS_BOX = (2.2, 1.6, 1.0)          # m, L x W x H (EST)
CHASSIS_COM_RNAV = (-0.05, 0.0, 0.80)  # m, in Z-up rover nav frame (EST)

# --------------------------------------------------------------------------
# Joint limits / actuator parameters.
# Published: 6 wheels, 4 corner wheels steerable, 52.5 cm wheels,
# max speed 152 m/h on flat ground, 45 deg tilt tolerance.
# Individual joint ranges are EST unless noted.
# --------------------------------------------------------------------------
WHEEL_RADIUS = 0.2625      # m; from published 1.65 m travel per revolution
                           # (NASA quotes 52.5 cm dia, JPL 52.6 cm)
N_GROUSERS = 0             # set by --grousers; the flight wheels have 48
TEXTURE_FILE = "M2020_Rover_Texture.png"
USE_TEXTURE = "2k"         # set by --texture; "" disables
GROUSER_H = 0.010          # m, protrusion EST
GROUSER_HW = 0.006         # m, half-thickness EST
WHEEL_HALFWIDTH = 0.1653   # m; measured from the JPL wheel mesh
MAX_WHEEL_RATE = 0.161     # rad/s; 152 m/h / WHEEL_RADIUS
# Drive torque EST = 250 N*m at the wheel.  Rationale: this puts the model in
# the traction-limited regime, which is where the real rover lives.  At 250 N*m
# the tangential force at the contact is 250/0.2625 = 952 N, above the friction
# limit mu*N = 0.9 * (1025*3.721/6) = 572 N, so the wheel slips before the
# actuator stalls.  At 100 N*m the model was torque-limited and could not climb
# even a 15 cm step, which contradicts the published 40 cm obstacle capability.
# For reference, holding a 30 deg slope needs only 83 N*m per wheel.
DRIVE_TORQUE = 250.0       # N*m EST, geared output at the wheel
STEER_TORQUE = 120.0       # N*m EST
STEER_RANGE = (-1.745, 1.745)   # +-100 deg EST (permits full crab + point turn)

ARM_JOINTS = {   # name: (range_lo, range_hi, torque)  -- ranges EST
    "JOINT1_ENC": (-3.142, 1.571, 250.0),
    "JOINT2_ENC": (-1.396, 0.785, 250.0),
    "JOINT3_ENC": (-3.142, 0.175, 200.0),
    "JOINT4_ENC": (-3.054, 3.054, 100.0),
    "JOINT5_ENC": (-3.142, 3.142, 60.0),
}
MISC_JOINTS = {
    "RSM_AZ_ENC": (-3.16, 3.16, 40.0),
    # The RSM elevation encoder's zero points the mast STRAIGHT DOWN; the
    # horizon is at -1.5446 rad and forward azimuth is +0.0117 rad (both
    # measured from the model).  Range below is horizon +-90 deg.
    # m2020_drive.mast_pose() takes elevation above the horizon instead.
    "RSM_EL_ENC": (-3.12, 0.03, 40.0),
    "HGA_AZ": (-3.142, 3.142, 30.0),
    "HGA_EL": (-0.087, 1.571, 30.0),
    "WATSON_CAP_ENC": (0.0, 2.094, 5.0),
    "SHERLOC_CAP_ENC": (0.0, 2.094, 5.0),
}
PRISM_JOINTS = {   # stroke ranges EST
    "DRILL_FEED": (0.0, 0.15, 2000.0),
    "STABILIZER_LOWER": (0.0, 0.10, 2000.0),
    "STABILIZER_UPPER": (0.0, 0.10, 2000.0),
}

# Passive suspension DOFs -- a rocker-bogie has no springs, only bearings.
PASSIVE = {
    "LEFT_DIFFERENTIAL": dict(damping=40.0, armature=0.5, frictionloss=5.0),
    "RIGHT_DIFFERENTIAL": dict(damping=40.0, armature=0.5, frictionloss=5.0),
    "LEFT_BOGIE": dict(damping=20.0, armature=0.3, frictionloss=3.0),
    "RIGHT_BOGIE": dict(damping=20.0, armature=0.3, frictionloss=3.0),
    "CENTER_DIFFERENTIAL": dict(damping=5.0, armature=0.1, frictionloss=1.0),
}

DRIVE_JOINTS = ["LF_DRIVE", "LM_DRIVE", "LR_DRIVE", "RF_DRIVE", "RM_DRIVE", "RR_DRIVE"]
STEER_JOINTS = ["LF_STEER", "LR_STEER", "RF_STEER", "RR_STEER"]

# Bodies we keep (everything with geometry or a DOF).  The ~80 empty
# Frame_* links in the URDF become sites / cameras instead of bodies.
SKIP_LINKS = {"ground", "Body_MHS_DebrisShield"}

# Cameras to emit by default (a representative subset of the 27 released).
DEFAULT_CAMS = [
    "HAZFLA", "HAZFRA", "HAZRL", "HAZRR",
    "NAVL", "NAVR", "SuperCam",
    "MCAM_Z_LEFT-Z034", "MCAM_Z_RIGHT-Z034",
    "MCAM_Z_LEFT-Z110", "MCAM_Z_RIGHT-Z110",
    "WATSON", "SHERLOC_CAMERA",
]
CAM_FRAME_TO_BODY = {
    "Body_Chassis": "chassis",
    "Body_RSM_EL": "rsm_el",
    "Frame_TURRET": "turret",
}
# Frame_TURRET is a fixed frame on Body_RA_Link5; Body_Turret is also fixed on
# Body_RA_Link5.  Offset of Frame_TURRET relative to Body_Turret (JPL frame):
FRAME_TURRET_IN_TURRET = np.array([2.7386, -0.2724, -0.3116 + (-0.3285) - (-0.3116)])


def short(link):
    """Body_WheelLeftFront -> wheel_left_front style short names."""
    s = link.replace("Body_", "").replace("Frame_", "")
    out = []
    for i, ch in enumerate(s):
        if ch.isupper() and i and not s[i - 1].isupper() and s[i - 1] != "_":
            out.append("_")
        out.append(ch.lower())
    return "".join(out).replace("__", "_")


# --------------------------------------------------------------------------
# Mesh handling
# --------------------------------------------------------------------------
def convert_meshes(mesh_src, out_dir, decimate_above, texture="2k"):
    """glTF -> OBJ, reframed to Z-up, keeping the UV coordinates.

    Every JPL mesh is UV-mapped into ONE shared texture atlas
    (Textures/M2020_Rover_Texture*.jpg), so exporting the UVs and pointing
    MuJoCo at that atlas reproduces the flight paint scheme exactly.

    trimesh drops UVs when include_texture=False and writes a redundant copy
    of the atlas per mesh when it is on, so we export with textures, strip the
    per-mesh material references, and keep a single shared atlas.
    """
    import shutil
    import tempfile
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    T = np.eye(4)
    T[:3, :3] = M
    props = {}
    n_uv = 0
    for fn in sorted(os.listdir(mesh_src)):
        if not fn.endswith(".gltf"):
            continue
        name = fn[:-5]
        m = trimesh.load(os.path.join(mesh_src, fn), force="mesh")
        m.apply_transform(T)
        if decimate_above and len(m.faces) > decimate_above:
            try:
                m = m.simplify_quadric_decimation(decimate_above)
            except Exception:
                pass
        dst = os.path.join(out_dir, name + ".obj")
        if texture and getattr(m.visual, "uv", None) is not None:
            with tempfile.TemporaryDirectory() as td:
                m.export(os.path.join(td, "m.obj"))
                lines = open(os.path.join(td, "m.obj")).read().splitlines()
            keep = [l for l in lines
                    if not l.startswith(("mtllib", "usemtl"))]
            open(dst, "w").write("\n".join(keep) + "\n")
            n_uv += 1
        else:
            m.export(dst, include_texture=False, include_normals=False)
        props[name] = shell_inertia(m)

    if texture:
        src = os.path.join(mesh_src, "Textures",
                           f"M2020_Rover_Texture_{texture}.jpg")
        if not os.path.exists(src):
            src = os.path.join(mesh_src, "Textures", "M2020_Rover_Texture.jpg")
        if os.path.exists(src):
            # MuJoCo only reads PNG textures, not JPEG.
            from PIL import Image
            Image.open(src).convert("RGB").save(
                os.path.join(out_dir, TEXTURE_FILE), optimize=True)
            print(f"[texture] {os.path.basename(src)} -> {TEXTURE_FILE}, "
                  f"{n_uv} meshes UV-mapped")
        else:
            print("[texture] atlas not found in the URDF repo; "
                  "falling back to flat materials")
    return props


def shell_inertia(mesh):
    """Centroid + unit-mass inertia treating the surface as a uniform thin
    shell (mass proportional to triangle area).  Robust for the non-watertight
    JPL meshes, and a good model for thin-walled aerospace structure."""
    tris = mesh.triangles                      # (n, 3, 3)
    areas = mesh.area_faces
    A = areas.sum()
    bbox = (mesh.bounds[0].copy(), mesh.bounds[1].copy())
    if A <= 0:
        return dict(com=np.zeros(3), I=np.eye(3) * 1e-3, bbox=bbox)
    w = areas / A                              # mass fraction per triangle
    cen = tris.mean(axis=1)                    # triangle centroids
    com = (w[:, None] * cen).sum(axis=0)
    # Per-triangle inertia about its own centroid (uniform triangular lamina)
    # plus parallel-axis to the body CoM.
    d = cen - com
    I = np.zeros((3, 3))
    # parallel-axis term
    for k in range(3):
        for j in range(3):
            I[k, j] -= (w * d[:, k] * d[:, j]).sum()
    I += np.eye(3) * (w * (d ** 2).sum(axis=1)).sum()
    # local triangle lamina term: J = (1/12) sum over vertices of (v-c)(v-c)^T
    loc = tris - cen[:, None, :]
    for v in range(3):
        lv = loc[:, v, :]
        outer = np.zeros((3, 3))
        for k in range(3):
            for j in range(3):
                outer[k, j] = (w * lv[:, k] * lv[:, j]).sum()
        I += np.eye(3) * np.trace(outer) / 12.0 - outer / 12.0
    return dict(com=com, I=I, bbox=bbox)


def box_inertia(mass, dims):
    x, y, z = dims
    return mass / 12.0 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])


def diagonalize(I):
    """Symmetric inertia tensor -> (diagonal inertia, quaternion of principal
    axes) in MuJoCo's fullinertia-free form."""
    I = 0.5 * (I + I.T)
    vals, vecs = np.linalg.eigh(I)
    if np.linalg.det(vecs) < 0:
        vecs[:, 0] *= -1
    return vals, R_to_quat(vecs)


# --------------------------------------------------------------------------
# CAHV(ORE) -> MuJoCo camera
# --------------------------------------------------------------------------
def cahv_to_mjcf_cam(model):
    C = np.array(model["C"], float)
    A = np.array(model["A"], float)
    H = np.array(model["H"], float)
    V = np.array(model["V"], float)
    A = A / np.linalg.norm(A)
    h = H - (H @ A) * A
    v = V - (V @ A) * A
    hs, vs = np.linalg.norm(h), np.linalg.norm(v)
    h, v = h / hs, v / vs
    # image +x = h, image +y (down) = v, boresight = A
    # MuJoCo camera looks down its own -Z, with +X right and +Y up.
    Rc = np.column_stack([h, -v, -A])
    fovy = np.degrees(2.0 * np.arctan(0.5 * model["height"] / vs))
    # Frame conversion for a CAMERA is M @ Rc, NOT M @ Rc @ M.  The columns of
    # Rc are the camera's axes expressed in the parent frame, so only the
    # parent frame is being re-expressed -- the camera's own axis convention
    # (+x right, +y up, looking down -z) is fixed by MuJoCo and must not be
    # flipped as well.  Using M @ Rc @ M here inverts every boresight.
    # (Body rotations DO use M R M, because there both frames are flipped.)
    return flip_p(C), R_to_quat(M @ Rc), fovy


# --------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-repo",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "m2020-urdf-models"),
                    help="JPL m2020-urdf-models checkout (default: the "
                         "m2020-urdf-models/ clone in the repo root)")
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
                    help="directory to write the MJCF + assets into (default: models/)")
    ap.add_argument("--decimate-above", type=int, default=0,
                    help="decimate meshes above N faces (0 = keep full res)")
    ap.add_argument("--skip-mesh-convert", action="store_true")
    ap.add_argument("--assets-only", action="store_true",
                    help="only convert JPL's meshes and texture into models/assets "
                         "and stop; leaves the committed MJCF untouched (setup.sh "
                         "uses this, since the JPL files aren't stored in this repo)")
    ap.add_argument("--all-cameras", action="store_true")
    ap.add_argument("--texture", default="2k",
                    choices=["", "none", "1k", "2k", "4k", "8k"],
                    help="JPL rover texture atlas resolution; "
                         "none = flat materials (default 2k)")
    ap.add_argument("--grousers", type=int, default=0, metavar="N",
                    help="add N grouser (cleat) boxes per wheel; the flight "
                         "wheels have 48.  Greatly improves step climbing at "
                         "the cost of 6*N extra collision geoms.")
    args = ap.parse_args()
    global N_GROUSERS, USE_TEXTURE
    N_GROUSERS = args.grousers
    USE_TEXTURE = "" if args.texture in ("", "none") else args.texture

    repo = args.urdf_repo
    out = args.out
    os.makedirs(out, exist_ok=True)
    assets = os.path.join(out, "assets")

    if args.skip_mesh_convert and os.path.isdir(assets):
        import trimesh
        props = {}
        for fn in sorted(os.listdir(assets)):
            if fn.endswith(".obj"):
                props[fn[:-4]] = shell_inertia(trimesh.load(
                    os.path.join(assets, fn), force="mesh"))
    else:
        props = convert_meshes(os.path.join(repo, "rover", "meshes"),
                               assets, args.decimate_above, USE_TEXTURE)
    print(f"[meshes] {len(props)} parts ready in {assets}")
    if args.assets_only:
        return

    # ---- parse URDF -----------------------------------------------------
    root = ET.parse(os.path.join(repo, "rover", "m2020.urdf")).getroot()
    links, visuals = {}, {}
    for l in root.findall("link"):
        n = l.get("name")
        links[n] = l
        vs = []
        for v in l.findall("visual"):
            g = v.find("geometry/mesh")
            if g is None:
                continue
            o = v.find("origin")
            vs.append(dict(
                mesh=os.path.basename(g.get("filename")).replace(".gltf", ""),
                pos=[float(x) for x in o.get("xyz").split()],
                rpy=[float(x) for x in o.get("rpy").split()]))
        visuals[n] = vs

    joints, children = {}, defaultdict(list)
    for j in root.findall("joint"):
        o = j.find("origin")
        ax = j.find("axis")
        jd = dict(
            name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            pos=[float(x) for x in o.get("xyz").split()] if o is not None else [0, 0, 0],
            rpy=[float(x) for x in o.get("rpy").split()] if o is not None else [0, 0, 0],
            axis=[float(x) for x in ax.get("xyz").split()] if ax is not None else None)
        joints[jd["child"]] = jd
        children[jd["parent"]].append(jd["child"])

    # Bodies we actually emit: links with a mesh or a movable joint, plus the
    # fixed links needed to connect them.
    def has_content(link):
        if link in SKIP_LINKS:
            return False
        if visuals.get(link):
            return True
        jd = joints.get(link)
        return bool(jd and jd["type"] in ("revolute", "continuous", "prismatic"))

    keep = set()
    for link in links:
        if has_content(link):
            keep.add(link)
            p = joints.get(link, {}).get("parent")
            while p and p not in ("ground", None):
                keep.add(p)
                p = joints.get(p, {}).get("parent")
    keep.discard("ground")
    keep -= SKIP_LINKS

    # ---- masses ---------------------------------------------------------
    assigned = {k: v for k, v in MASS.items() if v is not None and k in keep}
    chassis_mass = TOTAL_MASS - sum(assigned.values())
    assigned["Body_Chassis"] = chassis_mass
    print(f"[mass]   chassis {chassis_mass:.1f} kg, "
          f"total {sum(assigned.values()):.1f} kg")

    # ---- cameras --------------------------------------------------------
    cams = json.load(open(os.path.join(repo, "m2020-camera-models.json")))
    wanted = None if args.all_cameras else set(DEFAULT_CAMS)
    by_body = defaultdict(list)
    for c in cams:
        if wanted is not None and c["name"] not in wanted:
            continue
        if c["frame"] not in CAM_FRAME_TO_BODY:
            continue
        pos, quat, fovy = cahv_to_mjcf_cam(c["model"])
        if c["frame"] == "Frame_TURRET":
            pos = pos + flip_p(FRAME_TURRET_IN_TURRET)
        by_body[CAM_FRAME_TO_BODY[c["frame"]]].append(
            (c["name"].replace("-", "_").lower(), pos, quat, fovy))

    # ---- emit -----------------------------------------------------------
    xml = emit(keep, links, visuals, joints, children, props, assigned, by_body)
    path = os.path.join(out, "m2020_perseverance.xml")
    open(path, "w").write(xml)
    print(f"[mjcf]   wrote {path} ({len(xml)/1024:.0f} kB)")
    return path


JOINT_ORDER = []           # filled during emit, in MJCF tree order
# Working poses (rad).  EST -- chosen so the arm clears the deck and the
# ground; the flight stow pose is not published.
RSM_EL_HORIZON = -1.5446   # rad, measured: mast boresight level
RSM_AZ_FORWARD = 0.0117    # rad, measured: mast boresight straight ahead
STOW_POSE = {"JOINT1_ENC": 0.0, "JOINT2_ENC": -0.90, "JOINT3_ENC": -2.55,
             "JOINT4_ENC": 0.0, "JOINT5_ENC": 0.0,
             "RSM_AZ_ENC": RSM_AZ_FORWARD, "RSM_EL_ENC": RSM_EL_HORIZON}
# WORK_POSE solved by IK: drill tip 2.26 m ahead of the RNAV origin and
# 0.33 m above the surface, with no arm hardware below 0.10 m.
WORK_POSE = {"JOINT1_ENC": 0.329, "JOINT2_ENC": 0.785, "JOINT3_ENC": -1.046,
             "JOINT4_ENC": -1.901, "JOINT5_ENC": 0.345,
             "RSM_AZ_ENC": RSM_AZ_FORWARD, "RSM_EL_ENC": RSM_EL_HORIZON - 0.6}


def emit(keep, links, visuals, joints, children, props, mass, cams_by_body):
    L = []
    w = L.append
    JOINT_ORDER.clear()

    w('<mujoco model="M2020 Perseverance">')
    w('''
  <!--
  ===========================================================================
  NASA/JPL Mars 2020 "Perseverance" rover  -  MuJoCo MJCF model
  ===========================================================================
  Geometry + kinematics : github.com/nasa-jpl/m2020-urdf-models
                          Credit NASA/JPL-Caltech (release URS307049).
                          Rover modeling and texturing by Zareh Gorjian;
                          URDF conversion by the JPL RSVP team.
  Camera models         : the 27 CAHV / CAHVOR / CAHVORE flight calibrations
                          shipped in the same repository, converted to MuJoCo
                          cameras (position, boresight and vertical FOV).
  Generated by          : build_m2020.py

  FRAME
    The JPL rover frame is X-forward, Y-right, Z-down.  This model is rotated
    180 deg about +X into the MuJoCo/ROS convention: X forward, Y left, Z up.
    The chassis body frame is the M2020 Rover Navigation (RNAV) frame, whose
    origin sits on the ground plane beneath the rover centre -- so spawning
    the chassis at z = 0 places all six wheels exactly on flat ground.

  PUBLISHED NUMBERS USED (NASA)
    total mass 1025 kg | wheel dia 52.5 cm (1.65 m per revolution)
    6 wheels, 4 steerable | top speed 152 m/h | tilt tolerance 45 deg
    ground clearance 60 cm | arm 2.1 m, 5 DOF | MMRTG 45 kg
    Mars surface gravity 3.721 m/s^2

  ESTIMATED (marked EST): the per-link mass split, link inertias, joint
    ranges, actuator torques and gains, and terrain friction.  JPL has not
    published a mass/inertia breakdown; the split below is an engineering
    estimate that sums exactly to the published 1025 kg.
  ===========================================================================
  -->''')

    w('''
  <compiler angle="radian" meshdir="assets" texturedir="assets"
            autolimits="true" balanceinertia="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"
          gravity="0 0 -3.721"/>
  <!-- gravity above is Mars (3.721 m/s^2).  For Earth testing use -9.81. -->

  <size memory="20M"/>

  <visual>
    <global azimuth="130" elevation="-20" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.02" zfar="60"/>
    <headlight ambient="0.22 0.21 0.20" diffuse="0.45 0.43 0.41" specular="0.12 0.12 0.12"/>
  </visual>

  <statistic center="0 0 0.8" extent="3.5"/>
''')

    # ---- assets ---------------------------------------------------------
    w("  <asset>")
    w('''    <texture type="skybox" builtin="gradient" width="512" height="3072"
             rgb1="0.72 0.53 0.37" rgb2="0.35 0.24 0.18"/>
    <texture name="regolith" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.62 0.44 0.31" rgb2="0.55 0.38 0.26" mark="edge" markrgb="0.5 0.35 0.24"/>
    <material name="regolith" texture="regolith" texrepeat="24 24" texuniform="true"
              specular="0.05" shininess="0.05" reflectance="0.0"/>

    <!-- Flight hardware finishes.  The released glTF carries PBR textures that
         MuJoCo does not import; these approximate the flight appearance. -->
    <material name="mli_gold"   rgba="0.78 0.62 0.24 1" specular="0.6" shininess="0.55"/>
    <material name="white_paint" rgba="0.87 0.86 0.84 1" specular="0.25" shininess="0.25"/>
    <material name="alu"        rgba="0.68 0.68 0.70 1" specular="0.7" shininess="0.6"/>
    <material name="dark_alu"   rgba="0.30 0.30 0.32 1" specular="0.5" shininess="0.4"/>
    <material name="ti_wheel"   rgba="0.74 0.73 0.71 1" specular="0.6" shininess="0.5"/>
    <material name="black_ano"  rgba="0.14 0.14 0.15 1" specular="0.35" shininess="0.3"/>
<!--ROVER_TEXTURE-->
''')
    if USE_TEXTURE:
        # One shared atlas; every JPL mesh is UV-mapped into it. This is the
        # actual flight paint scheme, not an approximation of it.
        L[-1] = L[-1].replace("<!--ROVER_TEXTURE-->",
            f'    <texture name="rover_atlas" type="2d" file="{TEXTURE_FILE}"/>\n'
            f'    <material name="rover" texture="rover_atlas" texuniform="false"\n'
            f'              specular="0.25" shininess="0.3" reflectance="0.0"/>')
    else:
        L[-1] = L[-1].replace("<!--ROVER_TEXTURE-->", "")
    for name in sorted(props):
        w(f'    <mesh name="{name}" file="{name}.obj"/>')
    w("  </asset>\n")

    # ---- defaults -------------------------------------------------------
    w('''  <default>
    <default class="m2020">
      <joint limited="false" damping="1" armature="0.01"/>
      <geom friction="0.9 0.01 0.001" solref="0.008 1" solimp="0.95 0.99 0.001"/>

      <!-- visual: the JPL meshes.  No collision (contype/conaffinity = 0). -->
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"
              density="0" material="white_paint"/>
      </default>

      <!-- collision: primitives.  contype=1/conaffinity=2 pairs only with the
           terrain (contype=2/conaffinity=1), so the rover never self-collides.
           For self-collision set conaffinity="3" on this class. -->
      <default class="collision">
        <geom contype="1" conaffinity="2" group="3" density="0" rgba="0.8 0.2 0.2 0.35"/>
        <default class="wheel">
          <geom type="cylinder" size="0.2625 0.1653" condim="4"
                friction="0.9 0.02 0.001" solref="0.006 1" priority="2"/>
        </default>
        <!-- grousers (wheel cleats); only emitted with --grousers -->
        <default class="grouser">
          <geom type="box" size="''' + f"{GROUSER_HW} {WHEEL_HALFWIDTH} {GROUSER_H}" + '''" condim="4"
                friction="1.2 0.02 0.001" solref="0.006 1" priority="2"
                rgba="0.8 0.4 0.2 0.5"/>
        </default>
      </default>

      <!-- actuator/joint classes -->
      <default class="drive">
        <joint axis="0 1 0" damping="2.0" armature="0.35" frictionloss="1.5"/>
        <general ctrlrange="''' + f"{-MAX_WHEEL_RATE} {MAX_WHEEL_RATE}" + '''"
                 forcerange="''' + f"{-DRIVE_TORQUE} {DRIVE_TORQUE}" + '''"
                 gainprm="6000 0 0" biasprm="0 0 -6000" biastype="affine"
                 dyntype="none"/>
      </default>
      <default class="steer">
        <joint axis="0 0 1" damping="12" armature="0.25" frictionloss="3"
               range="-1.745 1.745"/>
        <position kp="6000" kv="600" forcerange="-120 120" ctrlrange="-1.745 1.745"/>
      </default>
      <default class="arm">
        <joint damping="25" armature="0.6" frictionloss="4"/>
        <position kp="8000" kv="800"/>
      </default>
      <default class="mast">
        <joint damping="8" armature="0.15" frictionloss="1.5"/>
        <position kp="900" kv="90"/>
      </default>
    </default>
  </default>
''')

    # ---- worldbody ------------------------------------------------------
    w("  <worldbody>")
    w('''    <light name="sun" directional="true" pos="6 -4 12" dir="-0.45 0.3 -1"
           diffuse="1.05 0.98 0.88" specular="0.3 0.3 0.3" castshadow="true"/>
    <geom name="terrain" type="plane" size="60 60 0.5" material="regolith"
          contype="2" conaffinity="1" condim="4" friction="0.9 0.02 0.001"/>
''')

    def inertial(link):
        m = mass.get(link, 0.0)
        if m <= 0:
            return None
        if link == "Body_Chassis":
            com = flip_p([CHASSIS_COM_RNAV[0], -CHASSIS_COM_RNAV[1],
                          -CHASSIS_COM_RNAV[2]])
            # CHASSIS_COM_RNAV is already given Z-up, so use it directly:
            com = np.array(CHASSIS_COM_RNAV, float)
            diag = box_inertia(m, CHASSIS_BOX)
            return (f'      <inertial pos="{f3(com)}" mass="{m:.3f}" '
                    f'diaginertia="{f3(diag)}"/>'
                    f'   <!-- EST: uniform box {CHASSIS_BOX[0]}x{CHASSIS_BOX[1]}'
                    f'x{CHASSIS_BOX[2]} m -->')
        vs = visuals.get(link) or []
        if not vs:
            return f'      <inertial pos="0 0 0" mass="{m:.3f}" diaginertia="0.01 0.01 0.01"/>'
        # combine all meshes of this link (shell model), in link coordinates
        coms, Is, ws = [], [], []
        for v in vs:
            p = props.get(v["mesh"])
            if p is None:
                continue
            off = flip_p(v["pos"])
            coms.append(p["com"] + off)
            Is.append(p["I"])
            ws.append(1.0)
        if not coms:
            return f'      <inertial pos="0 0 0" mass="{m:.3f}" diaginertia="0.01 0.01 0.01"/>'
        ws = np.array(ws) / sum(ws)
        com = sum(wi * c for wi, c in zip(ws, coms))
        I = np.zeros((3, 3))
        for wi, Ii, ci in zip(ws, Is, coms):
            d = ci - com
            I += wi * (Ii + np.eye(3) * (d @ d) - np.outer(d, d))
        I *= m
        vals, q = diagonalize(I)
        vals = np.maximum(vals, 1e-5)
        # enforce triangle inequality (MuJoCo requirement)
        for _ in range(8):
            a, b, c = vals
            if a + b >= c and a + c >= b and b + c >= a:
                break
            vals = vals + 0.02 * vals.max()
        return (f'      <inertial pos="{f3(com)}" quat="{f3(q)}" mass="{m:.3f}" '
                f'diaginertia="{f3(vals)}"/>'
                f'   <!-- EST mass; inertia from mesh (thin-shell) -->')

    MAT = {
        "CHASSIS": "white_paint", "CenterDifferential": "alu",
        "Left_Rocker": "white_paint", "Right_Rocker": "white_paint",
        "Left_Bogie": "white_paint", "Right_Bogie": "white_paint",
        "Turret": "dark_alu", "RSM": "white_paint", "RSM_Head": "white_paint",
        "HGA_AZ": "alu", "HGA_EL": "mli_gold",
    }
    for k in ["Wheel_LeftFront", "Wheel_LeftMiddle", "Wheel_LeftRear",
              "Wheel_RightFront", "Wheel_RightMiddle", "Wheel_RightRear"]:
        MAT[k] = "ti_wheel"
    for k in ["Steer_LeftFront", "Steer_LeftRear", "Steer_RightFront", "Steer_RightRear"]:
        MAT[k] = "alu"
    for k in ["Arm_LINK_1", "Arm_LINK_2", "Arm_LINK_3", "Arm_LINK_4", "Arm_LINK_5"]:
        MAT[k] = "white_paint"

    # ------------------------------------------------------------------
    # Collision primitives.  These are DERIVED from the parsed joint offsets
    # and mesh bounding boxes rather than hand-written, so they cannot drift
    # out of the frame convention.  (Hand-typing them in JPL coordinates is
    # an easy way to get the Y/Z signs wrong.)
    # ------------------------------------------------------------------
    def joff(child):
        """Child joint origin expressed in its parent body's (Z-up) frame."""
        return flip_p(joints[child]["pos"])

    def mesh_box(name):
        """Axis-aligned bbox (lo, hi) of a converted mesh, in link coords."""
        import trimesh
        p = props[name]
        return p["bbox"]

    def clamp(p, link):
        """Clamp a point into the union bbox of a link's visual meshes.
        The URDF places some pivots outside the tube they belong to (the
        front steer pivot sits at the wheel axle, below the rocker tube,
        because the steering knuckle spans the gap)."""
        vs = visuals.get(link) or []
        if not vs:
            return p
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for v in vs:
            blo, bhi = props[v["mesh"]]["bbox"]
            off = flip_p(v["pos"])
            lo = np.minimum(lo, np.array(blo) + off)
            hi = np.maximum(hi, np.array(bhi) + off)
        return np.clip(p, lo, hi)

    def cap(nm, a, b, r):
        return (f'<geom name="{nm}" class="collision" type="capsule" '
                f'fromto="{f3(a)} {f3(b)}" size="{r}"/>')

    EXTRA_COL = defaultdict(list)
    EXTRA_COL["Body_Chassis"] = [
        # belly pan: bottom face at z = 0.60 m == published 60 cm ground clearance
        '<geom name="belly" class="collision" type="box" size="0.95 0.70 0.26" pos="-0.05 0 0.86"/>',
        '<geom name="deck_fwd" class="collision" type="box" size="0.30 0.55 0.10" pos="0.95 0 1.10"/>',
        # MMRTG (published 64 cm dia x 66 cm), canted off the aft deck
        '<geom name="rtg" class="collision" type="cylinder" size="0.32 0.33" '
        'pos="-1.25 0 1.25" euler="0 1.05 0"/>',
    ]
    # rocker tube: bogie pivot -> front steer pivot
    for side, rl, bg, st in [("l", "Body_RockerLeft", "Body_BogieLeft", "Body_SteerLeftFront"),
                             ("r", "Body_RockerRight", "Body_BogieRight", "Body_SteerRightFront")]:
        EXTRA_COL[rl].append(cap(f"rocker_{side}_col",
                                 clamp(joff(bg), rl), clamp(joff(st), rl), 0.065))
    # bogie tube: rear steer pivot -> middle wheel axle
    for side, bg, st, mw in [("l", "Body_BogieLeft", "Body_SteerLeftRear", "Body_WheelLeftMiddle"),
                             ("r", "Body_BogieRight", "Body_SteerRightRear", "Body_WheelRightMiddle")]:
        EXTRA_COL[bg].append(cap(f"bogie_{side}_col",
                                 clamp(joff(st), bg), clamp(joff(mw), bg), 0.055))
    # arm links: link origin -> next joint origin
    for ln, nxt, rad in [("Body_RA_Link2", "Body_RA_Link3", 0.07),
                         ("Body_RA_Link3", "Body_RA_Link4", 0.06),
                         ("Body_RA_Link4", "Body_RA_Link5", 0.055)]:
        EXTRA_COL[ln].append(cap(f"{short(ln)}_col", np.zeros(3), joff(nxt), rad))
    # turret: a puck centred on the turret mesh
    tlo, thi = props["Turret"]["bbox"]
    tc = 0.5 * (np.array(tlo) + np.array(thi))
    EXTRA_COL["Body_Turret"].append(
        f'<geom name="turret_col" class="collision" type="cylinder" '
        f'size="0.27 0.16" pos="{f3(tc)}"/>')
    WHEEL_COL = {
        "Body_WheelLeftFront": "wheel_lf", "Body_WheelLeftMiddle": "wheel_lm",
        "Body_WheelLeftRear": "wheel_lr", "Body_WheelRightFront": "wheel_rf",
        "Body_WheelRightMiddle": "wheel_rm", "Body_WheelRightRear": "wheel_rr",
    }
    # Sites.  The drill tip is taken from the actual coring-bit mesh, so it
    # tracks the DRILL_FEED prismatic joint.
    # The bit mesh sits at the CorerFeed link's visual origin; the tip is its
    # far extremity along the DRILL_FEED axis (+x in the JPL turret frame).
    _bv = (visuals["Body_CorerFeed"] or [{}])[0]
    blo, bhi = props[_bv["mesh"]]["bbox"]
    _boff = flip_p(_bv["pos"])
    bit_tip = np.array([bhi[0], 0.5 * (blo[1] + bhi[1]),
                        0.5 * (blo[2] + bhi[2])]) + _boff
    SITES = {
        "Body_Chassis": [
            '<site name="imu" pos="-0.05 0 0.90" size="0.02" rgba="0 1 0 0.6"/>',
            '<site name="rnav_origin" pos="0 0 0" size="0.03" rgba="1 1 0 0.5"/>',
        ],
        "Body_CorerFeed": [f'<site name="drill_tip" pos="{f3(bit_tip)}" size="0.02" '
                           f'rgba="1 0 0 0.8"/>'],
    }

    def body(link, depth):
        ind = "  " * depth
        jd = joints.get(link)
        nm = short(link)
        pos = flip_p(jd["pos"]) if jd else np.zeros(3)
        R = flip_R(rpy_to_R(jd["rpy"])) if jd else np.eye(3)
        q = R_to_quat(R)
        qa = "" if np.allclose(q, [1, 0, 0, 0], atol=1e-9) else f' quat="{f3(q)}"'
        extra = ' childclass="m2020"' if link == "Body_Chassis" else ""
        w(f'{ind}<body name="{nm}" pos="{f3(pos)}"{qa}{extra}>')

        if link == "Body_Chassis":
            w(f'{ind}  <freejoint name="root"/>')

        ii = inertial(link)
        if ii:
            w(ind + ii.strip().replace("      ", "  ", 1) if False else
              f'{ind}  ' + ii.strip())

        # joint
        if jd and jd["type"] in ("revolute", "continuous", "prismatic"):
            jn = jd["name"]
            JOINT_ORDER.append(jn)
            ax = flip_p(jd["axis"])
            jtype = "slide" if jd["type"] == "prismatic" else "hinge"
            if jn in DRIVE_JOINTS:
                w(f'{ind}  <joint name="{jn}" class="drive" axis="{f3(ax)}"/>')
            elif jn in STEER_JOINTS:
                w(f'{ind}  <joint name="{jn}" class="steer" axis="{f3(ax)}"/>')
            elif jn in PASSIVE:
                p = PASSIVE[jn]
                w(f'{ind}  <joint name="{jn}" type="hinge" axis="{f3(ax)}" '
                  f'damping="{p["damping"]}" armature="{p["armature"]}" '
                  f'frictionloss="{p["frictionloss"]}"/>'
                  f'   <!-- passive rocker-bogie DOF, no spring -->')
            elif jn in ARM_JOINTS:
                lo, hi, _ = ARM_JOINTS[jn]
                w(f'{ind}  <joint name="{jn}" class="arm" type="hinge" axis="{f3(ax)}" '
                  f'range="{lo} {hi}"/>   <!-- EST range -->')
            elif jn in MISC_JOINTS:
                lo, hi, _ = MISC_JOINTS[jn]
                cls = "mast" if "RSM" in jn or "HGA" in jn else "arm"
                w(f'{ind}  <joint name="{jn}" class="{cls}" type="hinge" axis="{f3(ax)}" '
                  f'range="{lo} {hi}"/>   <!-- EST range -->')
            elif jn in PRISM_JOINTS:
                lo, hi, _ = PRISM_JOINTS[jn]
                w(f'{ind}  <joint name="{jn}" type="slide" axis="{f3(ax)}" '
                  f'range="{lo} {hi}" damping="600" armature="0.4"/>   <!-- EST stroke -->')
            else:
                w(f'{ind}  <joint name="{jn}" type="{jtype}" axis="{f3(ax)}"/>')

        # visual geoms
        for i, v in enumerate(visuals.get(link) or []):
            p = flip_p(v["pos"])
            Rv = flip_R(rpy_to_R(v["rpy"]))
            qv = R_to_quat(Rv)
            qs = "" if np.allclose(qv, [1, 0, 0, 0], atol=1e-9) else f' quat="{f3(qv)}"'
            mat = "rover" if USE_TEXTURE else MAT.get(v["mesh"], "alu")
            gn = f"{nm}_vis" + (f"{i}" if i else "")
            w(f'{ind}  <geom name="{gn}" class="visual" mesh="{v["mesh"]}" '
              f'pos="{f3(p)}"{qs} material="{mat}"/>')

        # collision
        if link in WHEEL_COL:
            wn = WHEEL_COL[link]
            w(f'{ind}  <geom name="{wn}" class="wheel" euler="1.5708 0 0"/>'
              f'   <!-- r=0.2625 m from published 1.65 m/rev -->')
            if N_GROUSERS:
                w(f'{ind}  <!-- {N_GROUSERS} grousers (cleats).  The flight wheels have 48;'
                  f' they are what lets the rover hook and climb a step taller'
                  f' than a smooth cylinder ever could. -->')
                for gi in range(N_GROUSERS):
                    th = 2 * np.pi * gi / N_GROUSERS
                    # grouser bar on the rim, aligned with the wheel axis (local y)
                    p = np.array([np.cos(th) * (WHEEL_RADIUS - 0.004), 0.0,
                                  np.sin(th) * (WHEEL_RADIUS - 0.004)])
                    w(f'{ind}  <geom name="{wn}_g{gi:02d}" class="grouser" pos="{f3(p)}" '
                      f'euler="0 {-th:.5f} 0"/>')
        for g in EXTRA_COL.get(link, []):
            w(f'{ind}  {g}')
        for s in SITES.get(link, []):
            w(f'{ind}  {s}')

        # cameras attached to this body
        for cname, cpos, cquat, fovy in cams_by_body.get(nm, []):
            w(f'{ind}  <camera name="{cname}" pos="{f3(cpos)}" quat="{f3(cquat)}" '
              f'fovy="{fovy:.3f}"/>   <!-- JPL CAHV(ORE) calibration -->')

        for ch in children.get(link, []):
            if ch in keep:
                body(ch, depth + 1)
        w(f"{ind}</body>")

    w('    <!-- chassis frame == M2020 Rover Navigation (RNAV) frame: origin on\n'
      '         the ground plane under the rover centre, so z=0 spawns on flat\n'
      '         ground with all six wheels in contact. -->')
    body("Body_Chassis", 2)
    w("  </worldbody>\n")

    # ---- equality: the rocker-bogie differential ------------------------
    w('''  <equality>
    <!-- The differential bar on the front deck constrains the two rockers to
         equal and opposite pitch relative to the chassis, so the chassis
         pitch is the average of the two rocker pitches.  This is the defining
         behaviour of a rocker-bogie suspension. -->
    <joint name="differential" joint1="LEFT_DIFFERENTIAL" joint2="RIGHT_DIFFERENTIAL"
           polycoef="0 -1 0 0 0" solref="0.004 1" solimp="0.98 0.999 0.001"/>
    <!-- Drive the visible differential bar from the rocker difference.
         Linkage ratio is EST (cosmetic; carries no load). -->
    <joint name="diff_bar_vis" joint1="CENTER_DIFFERENTIAL" joint2="LEFT_DIFFERENTIAL"
           polycoef="0 -1 0 0 0" solref="0.02 1"/>
  </equality>
''')

    # ---- actuators ------------------------------------------------------
    w("  <actuator>")
    w("    <!-- Drive: velocity servos, rad/s.  +-0.161 rad/s == the published")
    w("         152 m/h top speed at the 0.2625 m wheel radius. -->")
    for j in DRIVE_JOINTS:
        w(f'    <general name="{j.lower()}" joint="{j}" class="drive"/>')
    w("    <!-- Steer: position servos, rad.  EST range +-100 deg. -->")
    for j in STEER_JOINTS:
        w(f'    <position name="{j.lower()}" joint="{j}" class="steer"/>')
    w("    <!-- 5-DOF robotic arm (2.1 m).  EST ranges and torques. -->")
    for j, (lo, hi, t) in ARM_JOINTS.items():
        w(f'    <position name="{j.lower()}" joint="{j}" class="arm" '
          f'ctrlrange="{lo} {hi}" forcerange="{-t} {t}"/>')
    w("    <!-- Mast (RSM), HGA and turret caps.  EST ranges and torques. -->")
    for j, (lo, hi, t) in MISC_JOINTS.items():
        cls = "mast" if ("RSM" in j or "HGA" in j) else "arm"
        w(f'    <position name="{j.lower()}" joint="{j}" class="{cls}" '
          f'ctrlrange="{lo} {hi}" forcerange="{-t} {t}"/>')
    w("    <!-- Corer feed and stabilizers (prismatic).  EST strokes. -->")
    for j, (lo, hi, t) in PRISM_JOINTS.items():
        w(f'    <position name="{j.lower()}" joint="{j}" kp="60000" kv="3000" '
          f'ctrlrange="{lo} {hi}" forcerange="{-t} {t}"/>')
    w("  </actuator>\n")

    # ---- sensors --------------------------------------------------------
    w("  <sensor>")
    w('    <framequat name="imu_quat" objtype="site" objname="imu"/>')
    w('    <gyro name="imu_gyro" site="imu"/>')
    w('    <accelerometer name="imu_acc" site="imu"/>')
    w('    <framepos name="rover_pos" objtype="site" objname="rnav_origin"/>')
    w('    <velocimeter name="rover_vel" site="imu"/>')
    for j in DRIVE_JOINTS:
        w(f'    <jointpos name="{j.lower()}_pos" joint="{j}"/>')
        w(f'    <jointvel name="{j.lower()}_vel" joint="{j}"/>')
    for j in STEER_JOINTS:
        w(f'    <jointpos name="{j.lower()}_pos" joint="{j}"/>')
    for j in ["LEFT_DIFFERENTIAL", "RIGHT_DIFFERENTIAL", "LEFT_BOGIE", "RIGHT_BOGIE"]:
        w(f'    <jointpos name="{j.lower()}_pos" joint="{j}"/>')
    for j in ARM_JOINTS:
        w(f'    <jointpos name="{j.lower()}_pos" joint="{j}"/>')
    w('    <framepos name="drill_tip_pos" objtype="site" objname="drill_tip"/>')
    w("  </sensor>\n")

    # ---- keyframes ------------------------------------------------------
    # ---- keyframes ------------------------------------------------------
    # The URDF zero pose has the arm straight out in front, which is not a
    # pose the rover is ever actually in.  These are useful working poses.
    # nq is laid out as: freejoint(7) then the joints in tree order.
    order = [j for j in JOINT_ORDER]
    def kf(name, vals):
        q = [0.0] * len(order)
        for k, v in vals.items():
            q[order.index(k)] = v
        return (f'    <key name="{name}" qpos="0 0 0 1 0 0 0 '
                + " ".join(f"{x:g}" for x in q) + '"/>')

    w("  <keyframe>")
    w("    <!-- deployed: the URDF zero pose, arm fully extended forward -->")
    w(kf("deployed", {}))
    w("    <!-- stowed: arm folded back over the front deck, drive configuration -->")
    w(kf("stowed", STOW_POSE))
    w("    <!-- arm down at the workspace in front of the rover -->")
    w(kf("arm_work", WORK_POSE))
    w("  </keyframe>\n")

    w("</mujoco>")
    return "\n".join(L)


if __name__ == "__main__":
    main()
