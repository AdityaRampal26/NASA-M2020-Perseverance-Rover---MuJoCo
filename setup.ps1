# setup.ps1 -- one-time setup for the M2020 Perseverance MuJoCo project (Windows).
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# The Windows counterpart of setup.sh: creates a local virtualenv in .venv,
# installs MuJoCo, downloads and converts JPL's rover meshes, generates the
# terrain course, and runs the validation suite so you know it works before
# you drive. Works in Windows PowerShell 5.1 and PowerShell 7.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Stop on a failed native command too (PowerShell does not do this by itself).
function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE"
    }
}

$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Write-Host "==> using $(& $Py -V 2>&1) at $((Get-Command $Py).Source)"

if (-not (Test-Path .venv)) {
    Write-Host "==> creating .venv"
    Invoke-Step $Py @("-m", "venv", ".venv")
}
$VenvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "==> installing dependencies"
Invoke-Step $VenvPy @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
Invoke-Step $VenvPy @("-m", "pip", "install", "--quiet", "-r", "requirements.txt")

Invoke-Step $VenvPy @("-c", "import mujoco, sys; print(f'==> mujoco {mujoco.__version__} on {sys.platform}')")

# The rover meshes and texture belong to NASA/JPL and aren't stored in this repo.
# Fetch JPL's release and convert it into models/assets (about a minute, once).
if (-not (Test-Path models\assets\CHASSIS.obj)) {
    if (-not (Test-Path m2020-urdf-models)) {
        Write-Host "==> downloading the JPL rover models (nasa-jpl/m2020-urdf-models)"
        Invoke-Step git @("clone", "--quiet", "--depth", "1", "https://github.com/nasa-jpl/m2020-urdf-models", "m2020-urdf-models")
    }
    Write-Host "==> converting the JPL meshes into models/assets"
    Invoke-Step $VenvPy @("scripts\build_m2020.py", "--assets-only")
}

Write-Host "==> generating the terrain course"
Invoke-Step $VenvPy @("scripts\make_course.py")

Write-Host "==> validating the model"
Invoke-Step $VenvPy @("scripts\demo.py", "validate")

Write-Host @"

==> setup complete.

  Drive it:

      .venv\Scripts\Activate.ps1
      python scripts\teleop.py                   # Jezero terrain course
      python scripts\teleop.py --flat            # flat ground

  Other entry points:

      python scripts\demo.py validate            # physical validation suite
      python scripts\demo.py drive               # scripted manoeuvres
      python scripts\demo.py obstacle            # rock field, suspension trace
      python scripts\demo.py cameras             # render all 13 flight cameras
      python scripts\make_course.py --seed 12    # a different terrain

"@
