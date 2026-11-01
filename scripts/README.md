# scripts/

Run these from the repo root after `bash setup.sh && source .venv/bin/activate`. On macOS, use `mjpython` instead of `python` for anything that opens a window.

**Drive it yourself**

| Script | What it does |
|---|---|
| `teleop.py` | Drive the rover with the keyboard. Start here. `--course` puts it on the Jezero terrain |
| `demo.py` | Quick demos: `drive`, `obstacle` (rock field), `arm`, `cameras`, `viewer`, and `validate` (checks the model against NASA's numbers) |

**Autonomy**

| Script | What it does |
|---|---|
| `autonav.py` | Builds a hazard map from the terrain, plans a route with A\*, checks it, then drives it. `--viewer` to watch, `--plan-only` to skip the drive |
| `terrain_probe.py` | Terrain height lookups, and a check that drops the rover at a pose and measures its tilt, suspension angles and belly clearance |
| `m2020_drive.py` | Rocker-bogie steering: arcs, turn in place, and crab (sideways) driving |
| `gym_env.py` | A [Gymnasium](https://gymnasium.farama.org) environment (`Perseverance-Nav-v0`) for training driving policies with reinforcement learning: drive to a random goal without tipping. Needs `pip install "gymnasium>=1.0"`; run it directly for a self-check |
| `terramechanics.py` | Optional soft-soil wheel model (Bekker sinkage and compaction, slip-dependent traction). Run it for a rigid vs soft-soil slope sweep |

**Flight-style autonomy (ACE and ENav)**

The algorithms themselves live in [`../ace/`](../ace/) — pure kinematics, numpy
only, no simulator. These are the tools that run and check them.

| Script | What it does |
|---|---|
| `run_enav.py` | Drives to a goal with ENav on the Jezero terrain, replanning every metre |
| `ace_eval.py` | The physics oracle ACE gets checked against: drops the full rover onto the terrain, lets the suspension settle, and measures what really happened |
| `validate_ace.py` | The conservatism study: samples poses, compares ACE's ranges against the settled truth, and reports both containment and how tight the bounds are |

**Terrain**

| Script | What it does |
|---|---|
| `fetch_jezero.py` | Downloads a real HiRISE elevation map and turns it into MuJoCo terrain: Jezero Crater by default, `--site gale` for Curiosity's landing site, or `--url` for any other HiRISE DTM. `make_course.py --dtm NAME` then builds the course |
| `make_course.py` | Generates a made-up Mars-like course with rocks and a crater rim |

**Figures and animations**

| Script | What it does |
|---|---|
| `plot_drive.py` | The route plot on the main page (hazard map, planned vs. driven route) |
| `make_enav_3d.py` | The whole ENav + ACE loop in MuJoCo with the real rover: the reachable set, the paths ACE checked, which it rejected, the chosen one and the driven track, all drawn on the terrain. `--viewer` to watch it live (macOS: `mjpython`), `--gif` to record, `--dry-run` to drive without rendering |
| `make_enav_explainer.py` | The whole loop step by step: the plan beside what ACE predicted for belly clearance and tilt along the chosen path, against the published limits |
| `make_gifs.py` | The top-down route animation, the rocker-bogie animations (2D and MuJoCo) and the steering diagram |

**Model builder**

| Script | What it does |
|---|---|
| `build_m2020.py` | Rebuilds `models/m2020_perseverance.xml` from JPL's released rover files. Only needed if you want to change the model |
