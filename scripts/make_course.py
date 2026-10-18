#!/usr/bin/env python3
"""
make_course.py -- build a Jezero-flavoured driving course for the M2020 model.

Generates a procedural heightfield (fractal noise, gentle Mars-like relief),
scatters rocks and a crater rim, and writes a scene XML that swaps the flat
terrain plane of the rover model for that terrain.

    python scripts/make_course.py                 # default course
    python scripts/make_course.py --flat          # flat ground + rocks only
    python scripts/make_course.py --seed 7 --relief 0.9
"""

import argparse
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")   # MJCF + assets
OUT = os.path.join(ROOT, "out")         # generated results
HERE = MODELS


def fractal_noise(n, seed, octaves=6, persistence=0.52):
    """Value-noise fractal heightfield in [0, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        res = 2 ** (o + 2)
        if res > n:
            break
        grid = rng.random((res + 1, res + 1))
        # bilinear upsample to n x n
        yi = np.linspace(0, res, n)
        xi = np.linspace(0, res, n)
        y0 = np.clip(np.floor(yi).astype(int), 0, res - 1)
        x0 = np.clip(np.floor(xi).astype(int), 0, res - 1)
        fy = (yi - y0)[:, None]
        fx = (xi - x0)[None, :]
        g00 = grid[np.ix_(y0, x0)]
        g10 = grid[np.ix_(y0 + 1, x0)]
        g01 = grid[np.ix_(y0, x0 + 1)]
        g11 = grid[np.ix_(y0 + 1, x0 + 1)]
        # smoothstep for C1 continuity -- avoids faceted terrain
        sy, sx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        out += amp * ((g00 * (1 - sy) + g10 * sy) * (1 - sx)
                      + (g01 * (1 - sy) + g11 * sy) * sx)
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / (np.ptp(out) + 1e-9)   # np.ptp: numpy 2 removed the method


def build_heightfield(n, seed, half_extent, relief):
    h = fractal_noise(n, seed)

    ax = np.linspace(-half_extent, half_extent, n)
    X, Y = np.meshgrid(ax, ax)
    r = np.hypot(X, Y)

    # Flat, level pad around the origin so the rover always spawns clean,
    # blending out over a few metres.
    pad = np.clip((r - 5.0) / 6.0, 0.0, 1.0)
    h = h * pad

    # A shallow crater rim off to one side, for something to drive up.
    cr = np.hypot(X - 26.0, Y - 8.0)
    rim = np.exp(-((cr - 13.0) ** 2) / (2 * 4.0 ** 2)) * 0.85
    bowl = -np.exp(-(cr ** 2) / (2 * 9.0 ** 2)) * 0.5
    h = h + (rim + bowl) * pad

    h -= h.min()
    h /= (h.max() + 1e-9)
    return h, relief


def rocks_xml(seed, half_extent, n_rocks):
    """Scatter ellipsoid rocks, keeping the spawn pad clear."""
    rng = np.random.default_rng(seed + 991)
    parts = []
    placed = []
    tries = 0
    while len(placed) < n_rocks and tries < n_rocks * 60:
        tries += 1
        x = rng.uniform(-half_extent * 0.55, half_extent * 0.55)
        y = rng.uniform(-half_extent * 0.55, half_extent * 0.55)
        if np.hypot(x, y) < 4.5:                     # keep the pad clear
            continue
        if any(np.hypot(x - px, y - py) < 1.4 for px, py in placed):
            continue
        placed.append((x, y))
        i = len(placed)
        # Most rocks are small; a few are wheel-height obstacles.
        big = rng.random() < 0.18
        sx = rng.uniform(0.28, 0.55) if big else rng.uniform(0.10, 0.26)
        sy = sx * rng.uniform(0.7, 1.3)
        sz = sx * rng.uniform(0.45, 0.85)
        parts.append(
            f'    <geom name="rock{i:03d}" type="ellipsoid" '
            f'size="{sx:.3f} {sy:.3f} {sz:.3f}" '
            f'pos="{x:.2f} {y:.2f} {sz * 0.55:.3f}" '
            f'euler="0 0 {rng.uniform(0, 3.14):.2f}" '
            f'contype="2" conaffinity="1" condim="4" '
            f'friction="1.2 0.02 0.001" '
            f'rgba="{rng.uniform(0.34, 0.46):.2f} {rng.uniform(0.25, 0.33):.2f} '
            f'{rng.uniform(0.19, 0.26):.2f} 1"/>')
    return "\n".join(parts)


def build_jezero(src, args):
    """Swap the flat plane for a real HiRISE heightfield (Jezero by default;
    any site fetch_jezero.py has written to assets/<args.dtm>.png + .json).

    Terrain heights are found by ray-casting against the *compiled* model
    rather than by indexing the PNG. MuJoCo stores hfield rows flipped
    relative to the image and re-normalises the data, so index arithmetic on
    the PNG is a good way to drop the rover several metres underground.
    A ray cast is ground truth and matches exactly what collision will do.
    """
    import json
    import mujoco

    name = args.dtm
    meta_path = os.path.join(HERE, "assets", f"{name}.json")
    png = os.path.join(HERE, "assets", f"{name}.png")
    if not (os.path.exists(png) and os.path.exists(meta_path)):
        fetch = "" if name == "jezero" else (
            f" --site {name}" if name == "gale" else f" --url ... --name {name}")
        raise SystemExit(f"assets/{name}.png not found -- run:  "
                         f"python scripts/fetch_jezero.py{fetch}")
    meta = json.load(open(meta_path))
    rx, ry, zt, zb = meta["hfield_size"]
    title = meta.get("site_title", "Jezero Crater")   # older jezero.json has none

    hf = (f'    <hfield name="{name}" file="{name}.png" '
          f'size="{rx} {ry} {zt:.4f} {zb}"/>')
    old_plane = ('<geom name="terrain" type="plane" size="60 60 0.5" material="regolith"\n'
                 '          contype="2" conaffinity="1" condim="4" friction="0.9 0.02 0.001"/>')
    if old_plane not in src:
        raise SystemExit("terrain plane not found in the model XML")
    new_terrain = (
        f'<geom name="terrain" type="hfield" hfield="{name}" material="regolith"\n'
        '          pos="0 0 0" contype="2" conaffinity="1" condim="4"\n'
        '          friction="0.9 0.02 0.001"/>')

    base = src.replace("  </asset>", hf + "\n  </asset>").replace(old_plane, new_terrain)

    # --- pass 1: compile terrain only, then probe it -------------------
    probe_path = os.path.join(HERE, "_probe.xml")
    open(probe_path, "w").write(base)
    pm = mujoco.MjModel.from_xml_path(probe_path)
    pd = mujoco.MjData(pm)
    mujoco.mj_forward(pm, pd)
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # terrain is group 0
    top = zt + 50.0

    def ground(x, y):
        pnt = np.array([float(x), float(y), top])
        vec = np.array([0.0, 0.0, -1.0])
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(pm, pd, pnt, vec, group, 1, -1, gid)
        if dist < 0:
            return None
        return top - dist

    # --- rocks, sitting on the measured surface ------------------------
    rng = np.random.default_rng(args.seed + 17)
    parts, placed = [], []
    reach = min(rx, ry) * 0.8
    tries = 0
    while len(placed) < args.rocks and tries < args.rocks * 80:
        tries += 1
        x = rng.uniform(-reach, reach)
        y = rng.uniform(-reach, reach)
        if math.hypot(x, y) < 6.0:                     # keep the spawn clear
            continue
        if any(math.hypot(x - px, y - py) < 2.0 for px, py in placed):
            continue
        g = ground(x, y)
        if g is None:
            continue
        placed.append((x, y))
        i = len(placed)
        big = rng.random() < 0.15
        sx = rng.uniform(0.30, 0.55) if big else rng.uniform(0.10, 0.26)
        sy = sx * rng.uniform(0.7, 1.3)
        sz = sx * rng.uniform(0.45, 0.85)
        parts.append(
            f'    <geom name="rock{i:03d}" type="ellipsoid" '
            f'size="{sx:.3f} {sy:.3f} {sz:.3f}" '
            f'pos="{x:.2f} {y:.2f} {g + sz * 0.55:.3f}" '
            f'euler="0 0 {rng.uniform(0, 3.14):.2f}" '
            f'contype="2" conaffinity="1" condim="4" '
            f'friction="1.2 0.02 0.001" '
            f'rgba="{rng.uniform(0.34,0.46):.2f} {rng.uniform(0.25,0.33):.2f} '
            f'{rng.uniform(0.19,0.26):.2f} 1"/>')

    # Spawn height must clear the HIGHEST terrain under the wheel footprint,
    # not the height at the origin. On real terrain the ground under the six
    # wheels varies by ~0.15 m, and a wheel spawned even a few cm inside the
    # hfield gets resolved with a mega-newton impulse that launches the rover.
    WHEELS = [(1.185, 1.0625), (0.0, 1.1845), (-1.075, 1.0625),
              (1.185, -1.0625), (0.0, -1.1845), (-1.075, -1.0625)]
    hits = []
    for wx, wy in WHEELS:
        for dx in (-0.2, 0.0, 0.2):                  # sample across the tread
            for dy in (-0.16, 0.0, 0.16):
                g = ground(wx + dx, wy + dy)
                if g is not None:
                    hits.append(g)
    if not hits:
        raise SystemExit("could not probe the terrain under the rover")
    spawn = max(hits) + 0.12                          # clear the highest wheel
    centre = ground(0.0, 0.0)
    print(f"  terrain under the wheels: {min(hits):.2f} to {max(hits):.2f} m"
          + (f" (origin {centre:.2f} m)" if centre is not None else ""))
    os.remove(probe_path)

    scene = base.replace(new_terrain, new_terrain + "\n" + "\n".join(parts))
    scene = scene.replace('<body name="chassis" pos="0 0 0"',
                          f'<body name="chassis" pos="0 0 {spawn:.3f}"')
    # The keyframes pin the free joint explicitly, so they override the body
    # pos on reset -- they have to be lifted onto the terrain too, otherwise
    # mj_resetDataKeyframe drops the rover back to the datum, several metres
    # underground.
    scene = scene.replace('qpos="0 0 0 1 0 0 0',
                          f'qpos="0 0 {spawn:.3f} 1 0 0 0')
    # the free-camera default should look at the rover, not the datum
    scene = scene.replace('<statistic center="0 0 0.8" extent="3.5"/>',
                          f'<statistic center="0 0 {spawn + 0.8:.2f}" extent="3.5"/>')

    lat = meta["site_lat"]
    ns = f'{abs(lat):.4f} {"N" if lat >= 0 else "S"}'
    site = meta.get("site_name") or f'{ns} {meta["site_lon"]:.4f}E'
    banner = (f'\n  <!-- TERRAIN: real HiRISE DTM of {title}.\n'
              f'       site      {site} ({ns}, {meta["site_lon"]:.4f} E)\n'
              f'       patch     {meta["patch_m"]:.0f} x {meta["patch_m"]:.0f} m at '
              f'{meta["post_m"]:.2f} m/post (DTM native {meta["dtm_post_m"]:.4f} m)\n'
              f'       elevation {meta["elev_min_m"]:.1f} to {meta["elev_max_m"]:.1f} m '
              f'(Mars 2000 datum), relief {meta["relief_m"]:.2f} m\n'
              f'       slope     mean {meta["slope_mean_deg"]:.1f} deg, '
              f'p95 {meta["slope_p95_deg"]:.1f} deg, max {meta["slope_max_deg"]:.1f} deg\n'
              f'       source    {meta["source"]}\n'
              f'                 {meta["product_page"]}\n'
              f'       The relief is measured Mars. Rocks are added\n'
              f'       statistically -- a 1 m/post DTM cannot resolve them. -->\n')
    scene = scene.replace('<mujoco model="M2020 Perseverance">',
                          '<mujoco model="M2020 Perseverance">' + banner, 1)
    print(f"wrote {args.out}  [real {title} DTM: {site}]")
    print(f"  {meta['patch_m']:.0f} m patch, {meta['relief_m']:.2f} m relief, "
          f"{len(parts)} rocks, spawn z={spawn:.2f} m")
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m2020_perseverance.xml")
    ap.add_argument("--out", default=None,
                    help="default: course_<dtm>.xml with --jezero/--dtm, "
                         "course_procedural.xml otherwise")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--relief", type=float, default=1.1,
                    help="vertical scale of the heightfield, metres")
    ap.add_argument("--extent", type=float, default=45.0,
                    help="half-width of the terrain, metres")
    ap.add_argument("--res", type=int, default=257)
    ap.add_argument("--rocks", type=int, default=90)
    ap.add_argument("--flat", action="store_true",
                    help="keep the flat plane, just add rocks")
    ap.add_argument("--jezero", action="store_true",
                    help="use the real HiRISE Jezero DTM patch built by "
                         "fetch_jezero.py (assets/jezero.png); same as --dtm jezero")
    ap.add_argument("--dtm", default=None, metavar="NAME",
                    help="use any DTM patch fetch_jezero.py wrote to "
                         "assets/NAME.png, e.g. --dtm gale")
    args = ap.parse_args()
    if args.jezero and args.dtm is None:
        args.dtm = "jezero"
    if args.out is None:
        args.out = f"course_{args.dtm}.xml" if args.dtm else "course_procedural.xml"

    src = open(os.path.join(HERE, args.model)).read()
    rocks = rocks_xml(args.seed, args.extent, args.rocks)

    if args.dtm:
        scene = build_jezero(src, args)
        open(os.path.join(HERE, args.out), "w").write(scene)
        return

    if args.flat:
        scene = src.replace('<geom name="terrain"', rocks + '\n    <geom name="terrain"')
    else:
        try:
            from PIL import Image
            h, relief = build_heightfield(args.res, args.seed, args.extent, args.relief)
            img = (h * 65535).astype(np.uint16)
            path = os.path.join(HERE, "assets", "terrain.png")
            Image.fromarray(img).save(path)
            hf_asset = (f'    <hfield name="mars" file="terrain.png" '
                        f'size="{args.extent} {args.extent} {relief} 0.5"/>')
        except ImportError:
            print("Pillow not installed -- falling back to a flat course "
                  "(pip install pillow for terrain)")
            scene = src.replace('<geom name="terrain"', rocks + '\n    <geom name="terrain"')
            open(os.path.join(HERE, args.out), "w").write(scene)
            print(f"wrote {args.out} (flat)")
            return

        # add the hfield asset, then swap the plane for an hfield geom
        scene = src.replace("  </asset>", hf_asset + "\n  </asset>")
        old_plane = ('<geom name="terrain" type="plane" size="60 60 0.5" material="regolith"\n'
                     '          contype="2" conaffinity="1" condim="4" friction="0.9 0.02 0.001"/>')
        new_terrain = (
            f'<geom name="terrain" type="hfield" hfield="mars" material="regolith"\n'
            f'          pos="0 0 0" contype="2" conaffinity="1" condim="4"\n'
            f'          friction="0.9 0.02 0.001"/>\n'
            f'{rocks}')
        if old_plane not in scene:
            raise SystemExit("terrain plane not found in the model XML -- "
                             "did the model change?")
        scene = scene.replace(old_plane, new_terrain)
        # spawn a little higher so the rover settles onto the terrain
        scene = scene.replace('<body name="chassis" pos="0 0 0"',
                              '<body name="chassis" pos="0 0 0.05"')

    open(os.path.join(HERE, args.out), "w").write(scene)
    print(f"wrote {args.out}"
          + ("" if args.flat else f" + assets/terrain.png "
                                  f"({args.res}x{args.res}, {args.relief} m relief)"))


if __name__ == "__main__":
    main()
