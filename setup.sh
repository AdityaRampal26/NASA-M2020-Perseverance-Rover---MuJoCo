#!/usr/bin/env bash
# setup.sh -- one-time setup for the M2020 Perseverance MuJoCo project.
#
#   bash setup.sh
#
# Creates a local virtualenv in .venv, installs MuJoCo, downloads and converts
# JPL's rover meshes, generates the terrain course, and runs the validation
# suite so you know it works before you drive.

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "==> using $($PY -V 2>&1) at $(command -v "$PY")"

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

python - <<'PYEOF'
import mujoco, sys
print(f"==> mujoco {mujoco.__version__} on {sys.platform}")
PYEOF

# The rover meshes and texture belong to NASA/JPL and aren't stored in this repo.
# Fetch JPL's release and convert it into models/assets (about a minute, once).
if [ ! -f models/assets/CHASSIS.obj ]; then
  if [ ! -d m2020-urdf-models ]; then
    echo "==> downloading the JPL rover models (nasa-jpl/m2020-urdf-models)"
    git clone --depth 1 https://github.com/nasa-jpl/m2020-urdf-models m2020-urdf-models
  fi
  echo "==> converting the JPL meshes into models/assets"
  python scripts/build_m2020.py --assets-only
fi

echo "==> generating the terrain course"
python scripts/make_course.py

echo "==> validating the model"
python scripts/demo.py validate

cat <<'EOF'

==> setup complete.

  Drive it (macOS needs mjpython, which pip just installed):

      source .venv/bin/activate
      mjpython scripts/teleop.py                 # Jezero terrain course
      mjpython scripts/teleop.py --flat          # flat ground

  Other entry points:

      python scripts/demo.py validate            # physical validation suite
      python scripts/demo.py drive               # scripted manoeuvres
      python scripts/demo.py obstacle            # rock field, suspension trace
      python scripts/demo.py cameras             # render all 13 flight cameras
      python scripts/make_course.py --seed 12    # a different terrain

EOF
