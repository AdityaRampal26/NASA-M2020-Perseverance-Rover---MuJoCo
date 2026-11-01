# Technical notes — Mars 2020 Perseverance in MuJoCo

A physically validated MuJoCo (MJCF) model of NASA/JPL's Mars 2020 Perseverance
rover, built from JPL's own released flight geometry rather than from scratch.

![poses](../images/render_poses.png)

---

## Provenance

Everything geometric comes from official NASA/JPL open-source releases:

| What | Source |
|---|---|
| Kinematic tree, link/joint frames, meshes | [`nasa-jpl/m2020-urdf-models`](https://github.com/nasa-jpl/m2020-urdf-models) — the URDF the JPL RSVP team uses for operations visualization. Credit NASA/JPL-Caltech, release URS307049. Rover modeling and texturing by Zareh Gorjian. |
| Camera placement and fields of view | The 27 CAHV / CAHVOR / CAHVORE flight camera calibrations shipped in the same repository. |
| Masses, dimensions, speeds | Published NASA/JPL figures (see *Published numbers*, below). |

This means the rocker-bogie geometry, the 5-DOF arm link lengths, the mast and
HGA mounting, and the turret are **the real flight values**, not approximations.

License note: the JPL meshes and URDF are released by NASA/JPL-Caltech; observe
their terms. The build script and helper code in this repository are MIT-licensed (see `LICENSE`).

---

## Files

```
models/m2020_perseverance.xml          the model (48 grousers per wheel)  <- use this
models/m2020_perseverance_smooth.xml   smooth cylinder wheels, ~1.5x faster
models/course_jezero.xml               REAL HiRISE Jezero terrain
models/course_jezero_boulder.xml       Jezero terrain + a boulder, for the autonomy demo
models/course_procedural.xml           procedural terrain course
models/assets/                         terrain heightfields; setup.sh adds the 36 JPL meshes + texture

scripts/teleop.py                      drive it from the keyboard  <- start here
scripts/demo.py                        drive, obstacle, arm, cameras, validate
scripts/fetch_jezero.py                pull the real HiRISE Jezero DTM patch
scripts/terrain_probe.py               terrain queries; settle a pose, measure tilt/clearance
scripts/autonav.py                     hazard map + A* + safety check + autonomous drive
scripts/plot_drive.py                  figure of the autonav result
scripts/make_gifs.py                   route + rocker-bogie animations for the README
scripts/make_course.py                 generate terrain courses
scripts/m2020_drive.py                 rocker-bogie steering/drive kinematics
scripts/build_m2020.py                 regenerates the MJCF from the JPL URDF

setup.sh                               one-time setup: venv, deps, course, validation
```

Generated results (drive logs, plots, videos) are written to `out/`.

Quick start:

```bash
bash setup.sh
source .venv/bin/activate
mjpython scripts/teleop.py           # macOS needs mjpython; Linux use python
```

To rebuild from the URDF:

```bash
git clone https://github.com/nasa-jpl/m2020-urdf-models m2020-urdf-models
python scripts/build_m2020.py --grousers 48
```

---

## Frame convention

The JPL rover frame is **X-forward, Y-right, Z-down**. This model is rotated
180° about +X into the MuJoCo/ROS convention: **X-forward, Y-left, Z-up**. That
rotation has determinant +1, so mesh winding and normals are preserved.

Two consequences worth knowing:

* **Joint angles are numerically identical to JPL's URDF.** Rotating the frame
  and the joint axis together leaves the joint coordinate unchanged. So a steer
  angle here means what it means in JPL's data — including the fact that a
  **positive steer angle turns the wheel clockwise seen from above**, because
  the frame it was defined in points Z down. `m2020_drive.py` hides this;
  if you drive the actuators directly, mind the sign.
* **The chassis body frame is the M2020 Rover Navigation (RNAV) frame**, whose
  origin sits on the ground plane under the rover centre. Spawning the chassis
  at `z = 0` puts all six wheels exactly on flat ground (verified: every wheel
  bottom lands within 0.4 mm of the plane).

---

## Published numbers used

| Quantity | Value | Model |
|---|---|---|
| Total mass | 1025 kg | 1025.00 kg exactly |
| Length × width × height | 3.0 × 2.7 × 2.2 m | 3.02 × 2.70 × 2.08 m (measured from meshes) |
| Ground clearance | 60 cm | 60.0 cm (belly pan) |
| Wheel diameter | 52.5 cm (JPL: 52.6), 1.65 m per revolution | r = 0.2625 m |
| Grousers per wheel | 48 (Curiosity had 24) | 48, optional |
| Top speed | 152 m/h | 152.6 m/h measured |
| Tilt tolerance | 45° | 52.5° static tip angle |
| Robotic arm | 2.1 m, 5 rotary joints | from the JPL URDF |
| MMRTG | 45 kg, 64 × 66 cm | inside the chassis mass |
| Mars gravity | 3.721 m/s² | `option gravity` |

---

## What is estimated

JPL has not published a mass/inertia breakdown or joint limits, so these are
engineering estimates. Every one is tagged `EST` in the XML.

* **Mass split.** Mobility ≈ 132 kg, arm + turret ≈ 80 kg, mast ≈ 25 kg,
  HGA ≈ 15 kg, chassis = the remainder (773.6 kg, which includes the MMRTG,
  the warm electronics box, instruments and the sample caching assembly).
  The split sums to exactly the published 1025 kg.
* **Link inertias** are computed from the actual mesh geometry, treating each
  surface as a uniform thin shell — a reasonable model for thin-walled
  aerospace structure, and robust for the non-watertight JPL meshes. The
  chassis instead uses a uniform 2.2 × 1.6 × 1.0 m box, with its CoM placed at
  0.80 m; that CoM height is the dominant stability parameter and the easiest
  thing to retune.
* **Joint ranges.** Steer ±100°, arm ranges, prismatic strokes.
* **Drive torque: 250 N·m at the wheel.** Chosen on principle rather than
  guessed — see below.
* **Terrain friction** (µ = 0.9 nominal) and all actuator gains.

### Why 250 N·m

At 100 N·m the model could not climb even a 15 cm step, which contradicts the
published 40 cm obstacle capability. The diagnosis was that the model was
*torque-limited* — the actuators saturated before the wheels ever slipped —
whereas a real Mars rover on regolith is *traction-limited*. At 250 N·m the
tangential force at the contact is 250/0.2625 = 952 N, comfortably above the
friction limit µN = 0.9 × (1025 × 3.721 / 6) = 572 N, so the wheel now slips
before the actuator stalls. That is the correct regime. For reference, merely
holding station on a 30° slope needs only 83 N·m per wheel.

---

## Grousers matter, and here is the evidence

The flight wheels have 48 curved cleats. Modelling them is not cosmetic — it is
the difference between a rover that can climb and one that cannot:

| Vertical step | Smooth cylinder wheels | 48 grousers |
|---|---|---|
| 15 cm | blocked | climbs (+0.150 m) |
| 20 cm | blocked | climbs (+0.200 m) |
| 25 cm | blocked | climbs (+0.250 m) |

This is real physics, not a modelling artefact. Against a step edge the contact
normal tilts far from vertical; for a smooth wheel the force direction needed to
lift it falls outside the friction cone, so it can only slip. The grousers give
the wheel something to hook on. It is exactly why the wheels have them.

A vertical wall taller than the wheel *radius* (26.25 cm) is unclimbable by any
single wheel, since the contact point then sits above the axle. The published
"40 cm obstacle" figure refers to rocks, where the rocker-bogie geometry and the
grousers work together — not to a smooth vertical wall.

---

## Validation results

All produced by `python scripts/demo.py validate` and the drive tests.

**Statics**
* Total mass 1025.00 kg; CoM at 0.814 m; static tip angle 52.5° (spec: 45°).
* Settles flat, and contact normal forces sum to 3814.0 N against m·g = 3814.0 N.

**Mobility**
* Flat top speed 152.6 m/h against the published 152 m/h, at 0.6 % wheel slip.
* Straight-line: 1.7 mm lateral drift over 2.4 m; yaw drift −0.03°.
* Commanded arcs: 4 m → 4.09 m measured; 10 m → 10.27 m. Symmetric left/right.
* Point turn: ±1.53 °/s against an ideal 1.72 °/s (the shortfall is middle-wheel
  scrub, which is real), with only 15 mm of centre drift.
* Crab: tracks commanded heading within 3°.

**Gradeability** (tilted ground plane, µ = 0.9)

| Slope | 0° | 10° | 20° | 25° | 30° | 35° |
|---|---|---|---|---|---|---|
| Speed (m/h) | 152.6 | 147.7 | 142.3 | 135.7 | 126.3 | 116.1 |

Chassis pitch tracks the slope to within 0.1° throughout.

**Suspension.** Driving the left-front wheel onto a 25 cm block gives rocker
angles of +4.44° left and −4.45° right — the differential constraint holding to
5 significant figures — with the bogie taking −9.09°, while the chassis rolls
only 2.88°. Averaging the terrain instead of following it is the whole point of
a rocker-bogie, and the model does it.

**Performance.** 25× real time with grousers (337 geoms), 37× without (49).

---

## Model contents

33 bodies, 30 joints, 35 DOF, 24 actuators, 13 cameras, 31 sensors.

**Mobility.** Six drive joints and four corner steer joints, on the JPL
rocker-bogie tree: chassis → rocker → bogie → steer → wheel. The rocker pivots
are passive with no springs, as on the real vehicle.

**The differential** is a MuJoCo equality constraint forcing the two rocker
angles equal and opposite, so chassis pitch is the average of the two rocker
pitches. That is the defining behaviour of a rocker-bogie, and it is what the
bar across the front deck does mechanically. A second, cosmetic constraint
animates that bar.

**Arm.** The 5-DOF arm with the sample caching turret, the corer feed and both
stabilizer prismatic joints, plus the WATSON and SHERLOC dust covers.

**Actuators.** Drive joints are velocity servos in rad/s, clamped to ±0.161 rad/s
(the published top speed at the real wheel radius). Everything else is a
position servo in radians.

**Cameras.** 13 MuJoCo cameras converted from the JPL CAHV/CAHVOR/CAHVORE
calibrations — position, boresight and vertical FOV all derived from flight
calibration data. The derived FOVs check out independently: Mastcam-Z at 110 mm
comes out at 4.64° vertical against a published 4.6°. Pass `--all-cameras` to
emit all 27, including every Mastcam-Z zoom stop.

Note that the front Hazcam positions sit *inside* the simplified chassis hull,
so their views include some of the rover's own structure. That is faithful to
where the cameras actually are — real front Hazcam images always have the arm
and body in frame.

**Keyframes.** `deployed` (the URDF zero pose, arm straight out), `stowed`
(arm folded over the front deck — the drive configuration), `arm_work` (turret
down at the surface 2.26 m ahead, solved by IK).

**Sensors.** IMU (quaternion, gyro, accelerometer), rover position and velocity,
per-wheel drive position and velocity, steer positions, rocker and bogie angles,
arm joint positions, and drill tip position.

---

## Mast pointing gotcha

The RSM elevation encoder does **not** zero at the horizon — zero points the
mast straight down. Measured from the model, the horizon is at
`RSM_EL_ENC = −1.5446 rad` and forward azimuth is `RSM_AZ_ENC = +0.0117 rad`.

Use the helper, which takes the sane convention (az 0 = ahead, positive left;
el 0 = horizon, positive up) and is accurate to about 0.2°:

```python
from m2020_drive import mast_pose
mast_pose(model, data.ctrl, az=math.radians(45), el=math.radians(20))
```

---

## Driving it

```python
import mujoco, math
from m2020_drive import Mobility

m = mujoco.MjModel.from_xml_path("m2020_perseverance.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 1)      # stowed
mob = Mobility(m)

mob.arc(d.ctrl, 0.04, None)               # straight, 0.04 m/s
mob.arc(d.ctrl, 0.04, 4.0)                # 4 m radius arc, left
mob.point_turn(d.ctrl, 0.03)              # turn in place, left
mob.crab(d.ctrl, 0.04, math.pi/2)         # translate sideways
```

`Mobility` resolves all six wheels onto one instantaneous centre of rotation,
reading the wheel positions out of the compiled model rather than hard-coding
them. Note that the four corner wheels sit further from the rover centre than
the two middle wheels, so a point turn requires them to spin proportionally
faster — commanding all six at the same rate just scrubs. Like the real rover,
steer to the target angles with the drives stopped before rolling.

---

## Things you may want to change

* **Gravity** is Mars (3.721 m/s²). For Earth-comparison tests set `-9.81`.
* **Self-collision is off.** Rover parts have `contype="1" conaffinity="2"` and
  the terrain `2`/`1`, so the rover collides with the world but never itself.
  Set `conaffinity="3"` on the `collision` class to enable it.
* **Chassis CoM** (`CHASSIS_COM_RNAV` in the build script) drives tip-over
  behaviour more than anything else.
* **Textures.** The released glTF carries PBR textures that MuJoCo does not
  import; the model uses approximate flight finishes instead.
* **Mesh resolution.** 570k faces at full resolution. `--decimate-above N` if
  you need it lighter.

---

## Real Jezero Crater terrain

The default course is not procedural noise — it is the **actual HiRISE stereo
DTM of Perseverance's landing site**.

```bash
python scripts/fetch_jezero.py        # pull the terrain patch (~11 MB)
python scripts/make_course.py --jezero
mjpython scripts/teleop.py            # drives course_jezero.xml
```

| | |
|---|---|
| Source | `DTEEC_045994_1985_046060_1985_U01.IMG`, HiRISE stereo DTM, "Candidate landing site for 2020 mission in Jezero Crater" |
| Produced by | USGS-Astrogeology / University of Arizona, ISIS + SOCET SET |
| Native resolution | 1.0097 m/post, 14266 × 6673 |
| Site | Octavia E. Butler Landing, 18.4447° N 77.4508° E |
| Patch | 400 × 400 m at 0.78 m/post |
| Elevation | -2572.6 to -2564.3 m, Mars 2000 datum |
| Relief | 8.35 m |
| Slope | mean 3.1°, p95 8.2°, max 31.4° |

The full product is 363 MB. `fetch_jezero.py` reads the PDS3 label, solves the
equirectangular projection for the pixel containing the site, and HTTP
range-requests only the rows it needs — about 11 MB for a 400 m patch. The
projection math is checked against the label's own corner coordinates: the
west edge solves to sample 1.0 and the south edge to line 14266, both exact.

That the landing site comes out at −2572.6 to −2564.3 m with a mean slope of
3.1° is itself a sanity check — Jezero's floor sits near −2570 m, and the
ellipse was chosen precisely because it is this flat.

```bash
python scripts/fetch_jezero.py --size 600           # bigger patch
python scripts/fetch_jezero.py --lat 18.46 --lon 77.41
python scripts/fetch_jezero.py --survey             # find the high-relief terrain
```

**Rocks are added statistically.** A 1 m/post DTM cannot resolve boulders, so
the relief is measured Mars but the rocks scattered on it are synthetic —
placed by ray-casting onto the real surface so they sit on it correctly.

**Two things that bite when putting a rover on a real heightfield**, both fixed
here and worth knowing if you build your own:

1. *Spawn height must clear the highest wheel, not the origin.* Terrain under
   the six wheels varied by 0.29 m at this site. Dropping the rover using the
   height at its centre buried the left-rear wheel 5.2 cm in the hfield, and
   MuJoCo resolved that penetration with a **5.8 MN** impulse that flipped the
   rover on step one. `make_course.py` now ray-probes a grid under all six
   wheels and clears the maximum.
2. *Keyframes override body position.* `mj_resetDataKeyframe` restores the free
   joint from `qpos`, so a keyframe written for flat ground drops the rover
   back to the datum — several metres underground. The keyframe `qpos` gets
   lifted with the spawn.

Heights come from ray-casting the compiled model, not from indexing the PNG:
MuJoCo stores hfield rows flipped relative to the image and re-normalises the
data, so index arithmetic is an easy way to put the rover underground.

### Courses

| file | terrain |
|---|---|
| `course_jezero.xml` | real HiRISE DTM of the landing site (default) |
| `course_procedural.xml` | procedural fractal terrain, crater rim, rocks |
| `m2020_perseverance.xml` | flat ground |


---

## Flight texture

The rover wears its real paint. Every JPL mesh is UV-mapped into one shared
texture atlas (`Textures/M2020_Rover_Texture_*.jpg` in the URDF repo), so the
model uses that atlas directly rather than approximating it with flat colours.

```bash
python scripts/build_m2020.py --grousers 48 --texture 4k    # sharper
python scripts/build_m2020.py --grousers 48 --texture none  # flat materials
```

Default is 2k (5.1 MB PNG). Three things that bite here:

* `trimesh` drops UVs when exporting with `include_texture=False`, and writes
  a redundant copy of the atlas per mesh when it is on. The build exports with
  textures, strips the per-mesh `mtllib`/`usemtl` lines, and keeps one atlas.
* **MuJoCo reads PNG textures, not JPEG.** The atlas is converted on build.
* `meshdir` does not cover textures — MuJoCo resolves those against
  `texturedir`, which has to be set separately.

---

## ACE and ENav

The two algorithms Perseverance navigates with, implemented from the
published papers. The write-up lives with the code, in
[`ace/README.md`](../ace/README.md): how ACE turns terrain uncertainty into
guaranteed bounds, how ENav spends it, the full published-vs-implemented
comparison, what is estimated, where this deviates, and the validation
against settled physics.

The sections below cover the earlier A* route planner, which is a separate
thing and still in the repo.

## Autonomous navigation (the AI-planned-drive loop)

In December 2025 Perseverance drove 689 ft and 807 ft on routes planned by AI
rather than by human route planners: vision-language models read HiRISE orbital
imagery and elevation models, picked out hazards, emitted a waypoint path, and
JPL validated it in a digital twin before uplink.

`autonav.py` runs that loop here, at rover scale:

```
real HiRISE DTM  ->  hazard map  ->  A* waypoint route
                 ->  safety check in the digital twin  ->  drive it
```

```bash
python scripts/autonav.py --model course_jezero_boulder.xml --goal 18 6 --span 40  # the boulder demo
python scripts/autonav.py --goal 25 10             # plan, validate, drive (headless)
mjpython scripts/autonav.py --goal 25 10 --viewer  # watch it live
python scripts/autonav.py --goal 25 10 --record    # render the drive to autonav_drive.mp4
python scripts/autonav.py --goal 25 10 --plan-only
python scripts/autonav.py --survey                 # hazard map -> hazard_map.npz
python scripts/plot_drive.py                       # figure of the result
```

`--viewer` draws the planned route, the driven track and the goal into the
viewer's overlay scene, then executes the traverse while following the rover.
macOS needs `mjpython`, same as `teleop.py`. `--record` does the same offscreen
to a video, and takes `--realtime` to set the speed multiplier (viewer 8x,
record 60x -- at true speed a 23 m traverse takes 10 minutes).

One trap if you extend the drawing: the viewer's `user_scn` is a dedicated
overlay, so zeroing `ngeom` before adding markers is correct there. A
`Renderer`'s scene is not -- `update_scene()` has already filled it with the
model, and zeroing `ngeom` deletes the rover, leaving a path drawn across an
empty landscape.

**The hazard map** follows the ENav costmap definition in Abcouwer et al. — a
weighted sum of tilt, roughness ("the average squared distance between a fitted
plane and the points in the plane") and obstacle height, with infinite cost
where a limit is exceeded or the plane fit fails, then dilated by the rover
radius so a cell is only free if the whole vehicle fits.

**Validation** samples poses every 2 m along the route and settles the full
rocker-bogie on the terrain at each, checking tilt, suspension angles and belly
clearance. A route with any failing pose is rejected.

**Driving** uses the two primitives a real rover has: turn in place to the
heading, then drive an arc. No holonomic cheating.

### Result on the real terrain, with a boulder in the way

| | |
|---|---|
| straight line | passes **0.00 m** from the boulder centre — straight through it |
| planned route | passes **4.75 m** from it (needs 3.0 m: rover 1.6 + boulder 1.4) |
| cost of the detour | 22.8 m planned vs 19.0 m straight |
| safety check on the route | 16 poses, 0 failures, worst tilt 5.07°, min clearance 0.579 m |
| driven | 21.5 m odometry, 9.7 min, 133 m/h average, reached goal ±0.90 m |
| max tilt driving | **5.08°** |

The pre-drive check predicted a worst-case tilt of 5.07° and the drive hit 5.08°.

### Keeping the arm stowed

Two bugs conspired to make the arm deploy itself mid-drive, and both are worth
knowing if you script poses:

* **`ctrl = 0` is not "no command" for a position servo — it is a command to go
  to zero**, and zero is the URDF pose with the arm straight out. Loading the
  `stowed` keyframe and then stepping made the arm swing out and sag, keyframe
  or not. `m2020_drive.hold_pose(model, data)` latches every position actuator
  onto the joint angle the keyframe just set; call it after any reset. Drive
  joints are velocity servos, so they stay at zero.
* **`Mobility.__init__` used to reset the data it was handed**, to measure the
  wheel geometry — silently wiping the caller's keyframe and ctrl. It now
  measures on a private `MjData`. `teleop.py` masked this by re-commanding the
  arm every frame; `autonav.py` did not, so the arm drifted 2.56 rad.

This matters for the safety check, not just looks: the arm swinging out moves the CoM
forward and changes the settled attitude. The real rover drives stowed, so the
pose the route is checked against should be stowed too.


### Lessons from the hazard map

**Costmap resolution has to resolve the obstacle.** At 1.0 m and 0.5 m cell
size, *zero* of the rocks in a 80 × 80 m window were detected — the grid steps
straight over a 0.4 m rock. At 0.25 m, all five were caught. The default is now
0.25 m. A coarser heightmap is not conservative — it is blind.

**Raw max−min over a window is not an obstacle metric.** Used that way it just
re-measures slope: 3° across a 3.2 m rover footprint is already 0.17 m, so a
0.30 m threshold rejects open ground. It rejected 61% of the benign landing
ellipse. Both roughness and obstacle height must be residuals to a *fitted
plane*; after that change the same terrain came out 100% traversable and the
metrics separate cleanly — ground at cm scale (step p90 = 0.07 m), rocks at
metre scale.

A third one, specific to MuJoCo: **a ray that lands exactly on a heightfield
triangle edge can return the full ray length**, i.e. an elevation of zero. Nine
cells in 6561 on a 513-post hfield. Each punches a fake hole that corrupts a
wheel box or costmap cell — and they masquerade convincingly as obstacles.
The terrain sampler now samples 1.7 mm off-grid and cross-checks.


---

## Soft soil (optional terramechanics)

MuJoCo's ground is rigid, so out of the box a wheel either grips up to its
friction coefficient or slides. `scripts/terramechanics.py` adds the classic
soft-soil terms on top of the live contacts, without touching the model
files. Each wheel gets Bekker sinkage and compaction resistance from its
measured normal load, and a Janosi-Hanamoto traction limit that builds with
slip, applied by setting that wheel's friction coefficient to H/W every step.

```python
from terramechanics import SoftSoil
soil = SoftSoil(model, "lunar_bekker")   # or "dry_sand", or your own n, kc, kphi, c, phi, K
soil.apply(data)                          # before every mj_step
```

`python scripts/terramechanics.py` drives straight uphill at full speed for
60 s per slope (gravity rotated to make the slope), on rigid ground and on
the `lunar_bekker` soil, with the smooth-wheel model:

| Slope | Rigid m/h | Soft m/h | Travel reduction | Wheel slip | Sinkage |
|---|---|---|---|---|---|
| 0° | 150.9 | 133.6 | 0.11 | 0.09 | 2.9 cm |
| 5° | 148.6 | 92.9 | 0.38 | 0.29 | 2.8 cm |
| 10° | 146.3 | 58.9 | 0.61 | 0.48 | 2.6 cm |
| 15° | 144.0 | 29.0 | 0.81 | 0.61 | 2.6 cm |
| 20° | 141.8 | 7.0 | 0.95 | 0.67 | 2.4 cm |
| 25° | 139.7 | -8.5 | 1.06 | 0.73 | 2.4 cm |

On rigid ground the rover barely notices the slope; only the drive motors
sag a little under load. On loose soil, slip grows steeply with slope, the
rover is down to 7 m/h at 20° and slides backwards at 25°. Mars rovers see
the same thing on sand, which is why their drive planners treat sandy slopes
far more conservatively than rock. The cleated model gives the same picture
(29 m/h at 15° vs 29 m/h smooth).

**What it is not:** the rut is not geometric (wheels still ride on the rigid
surface; only its resistance is modelled), there is no bulldozing,
multi-pass or grouser-specific term, lateral grip uses the same coefficient
as longitudinal, and the soil values are textbook numbers (n, kc, kphi, c,
phi as tabulated by J. Y. Wong, *Theory of Ground Vehicles*; K is EST), not
measurements from Jezero. Treat the table as the right shape, not the right
numbers for a particular dune.

## Running it on your machine

```bash
git clone https://github.com/AdityaRampal26/NASA-M2020-Perseverance-Rover---MuJoCo.git
cd NASA-M2020-Perseverance-Rover---MuJoCo
bash setup.sh                    # venv + deps + terrain + validation
source .venv/bin/activate
mjpython scripts/teleop.py               # drive it
```

**On macOS the viewer must run under `mjpython`, not `python`.** MuJoCo's viewer
needs the Cocoa main thread, and `mjpython` (installed with the `mujoco` wheel)
is the launcher that provides it. `teleop.py` will tell you if you forget.
Headless scripts — `demo.py`, `make_course.py` — run under plain `python`.

### Driving

| key | action |
|---|---|
| `W` / `S` | forward / reverse (tap to build speed) |
| `A` / `D` | steer left / right through 5 arc radii (20 m down to 2.5 m) |
| `Q` / `E` | turn in place |
| `Z` / `C` | crab left / right |
| `X` | all stop |
| `R` | reset |
| `1` `2` `3` | arm: stowed / mid-deploy / work pose |
| `I` `K` `J` `L` | mast up / down / left / right, `M` to centre |
| `TAB` | MuJoCo control panel |
| `[` `]` | cycle through the 13 flight cameras |

The real rover tops out at 152 m/h — about 4 cm/s — so `teleop.py` paces the
wall clock at 6× by default (`--realtime 1` for true speed, `--realtime 20` to
cover ground). The physics is identical either way; only the pacing changes.

### The terrain course

`make_course.py` builds a procedural Jezero-flavoured heightfield — fractal
noise with a smoothstep interpolant, a flat blended pad at the origin so the
rover always spawns clean, a shallow crater rim to climb, and ~90 scattered
rocks with a few at wheel height. `--seed` for a different landscape,
`--relief` for how dramatic it is, `--flat` for rocks on flat ground.

Driving it: 10 m traversed in 240 s, climbing 0.65 m, pitching to 10.5°,
suspension articulating throughout.
