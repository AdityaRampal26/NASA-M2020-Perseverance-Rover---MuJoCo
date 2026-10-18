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

**Terrain**

| Script | What it does |
|---|---|
| `fetch_jezero.py` | Downloads a real HiRISE elevation map and turns it into MuJoCo terrain: Jezero Crater by default, `--site gale` for Curiosity's landing site, or `--url` for any other HiRISE DTM. `make_course.py --dtm NAME` then builds the course |
| `make_course.py` | Generates a made-up Mars-like course with rocks and a crater rim |

**Figures and animations**

| Script | What it does |
|---|---|
| `plot_drive.py` | The route plot on the main page (hazard map, planned vs. driven route) |
| `make_gifs.py` | The top-down route animation, the rocker-bogie animations (2D and MuJoCo) and the steering diagram |

**Model builder**

| Script | What it does |
|---|---|
| `build_m2020.py` | Rebuilds `models/m2020_perseverance.xml` from JPL's released rover files. Only needed if you want to change the model |
