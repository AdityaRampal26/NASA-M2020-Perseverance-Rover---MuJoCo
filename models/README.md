# models/

The simulated rover and the ground it drives on, in MuJoCo's XML format (MJCF).

| File | What it is |
|---|---|
| `m2020_perseverance.xml` | **The rover.** Every body, joint, motor, camera and sensor, on flat ground. Wheels have 48 grousers (cleats) each, like the real ones |
| `m2020_perseverance_smooth.xml` | Same rover with smooth cylinder wheels. About 1.5× faster to simulate, but can't climb as well |
| `course_jezero.xml` | The rover on **real Mars terrain**: the HiRISE elevation map of the Jezero Crater landing site, plus scattered rocks |
| `course_jezero_boulder.xml` | Same, with a 2.8 m boulder placed across the straight-line route, for the autonomy demo |
| `course_gale.xml` | Real Mars terrain at **Gale Crater**, beside Curiosity's Bradbury Landing, from HiRISE. Build other sites with `fetch_jezero.py --site` / `--url` and `make_course.py --dtm` |
| `course_procedural.xml` | A made-up, Mars-like course from `scripts/make_course.py` |
| `assets/` | The terrain height images. `setup.sh` also downloads JPL's 36 part meshes and the rover texture into here (they belong to NASA/JPL, so they aren't stored in the repo) |

**What's in the rover model:** 33 bodies, 30 joints, 24 motors, 13 cameras and 31 sensors. The rocker-bogie suspension, 5-joint robotic arm, mast and sample turret all come from JPL's released flight geometry. The full breakdown is in [`docs/TECHNICAL_NOTES.md`](../docs/TECHNICAL_NOTES.md).

**Want to change it?** Don't edit `m2020_perseverance.xml` by hand. It's generated from JPL's files by `scripts/build_m2020.py`, so change the builder and rebuild.
