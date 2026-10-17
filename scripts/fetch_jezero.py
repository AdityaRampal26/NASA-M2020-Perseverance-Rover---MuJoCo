#!/usr/bin/env python3
"""
fetch_jezero.py -- build a MuJoCo heightfield from the real HiRISE DTM of
Jezero Crater, so the rover drives on actual Mars terrain.

Source
    DTEEC_045994_1985_046060_1985_U01.IMG  (363 MB, PDS3)
    HiRISE stereo DTM, "Candidate landing site for 2020 mission in Jezero
    Crater", produced by USGS-Astrogeology / University of Arizona with ISIS
    and SOCET SET.  1.0097 m/post, elevations in metres relative to the
    Mars 2000 datum.
    https://www.uahirise.org/dtm/dtm.php?ID=ESP_045994_1985

The full product is 363 MB but a rover-scale patch is a few hundred metres
across, so this script reads the PDS label, solves for the pixel containing
the site you asked for, and HTTP range-requests only the rows it needs --
typically 10-20 MB instead of 363.

    python scripts/fetch_jezero.py                    # Octavia E. Butler landing site
    python scripts/fetch_jezero.py --size 600         # a bigger patch
    python scripts/fetch_jezero.py --lat 18.46 --lon 77.41
    python scripts/fetch_jezero.py --survey           # find the interesting terrain

Then:
    python scripts/make_course.py --jezero
    mjpython scripts/teleop.py --course
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")   # MJCF + assets
OUT = os.path.join(ROOT, "out")         # generated results
HERE = MODELS
URL = ("https://www.uahirise.org/PDS/DTM/ESP/ORB_045900_045999/"
       "ESP_045994_1985_ESP_046060_1985/"
       "DTEEC_045994_1985_046060_1985_U01.IMG")

# Perseverance touched down at Octavia E. Butler Landing, 18.4447 N 77.4508 E.
LANDING_SITE = (18.4447, 77.4508)

MISSING = np.frombuffer(bytes.fromhex("FF7FFFFB")[::-1], dtype="<f4")[0]


def get(url, start=None, end=None, timeout=120):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class PdsDtm:
    """Just enough PDS3 to read an equirectangular HiRISE DTM by range request."""

    def __init__(self, url, label_bytes=32768):
        self.url = url
        t = get(url, 0, label_bytes - 1).decode("latin-1")
        self.label = t

        def num(k):
            m = re.search(k + r"\s*=\s*([-\d.eE+]+)", t)
            if not m:
                raise KeyError(k)
            return float(m.group(1))

        self.record_bytes = int(num("RECORD_BYTES"))
        self.lines = int(num("LINES"))
        self.samples = int(num("LINE_SAMPLES"))
        self.scale = num("MAP_SCALE")               # metres per pixel
        self.res = num("MAP_RESOLUTION")            # pixels per degree
        self.clat = num("CENTER_LATITUDE")
        self.clon = num("CENTER_LONGITUDE")
        self.line_off = num("LINE_PROJECTION_OFFSET")
        self.samp_off = num("SAMPLE_PROJECTION_OFFSET")
        self.lat_min = num("MINIMUM_LATITUDE")
        self.lat_max = num("MAXIMUM_LATITUDE")
        self.lon_min = num("WESTERNMOST_LONGITUDE")
        self.lon_max = num("EASTERNMOST_LONGITUDE")
        img_rec = int(num(r"\^IMAGE"))
        self.data_start = (img_rec - 1) * self.record_bytes
        if self.samples * 4 != self.record_bytes:
            raise ValueError("unexpected record layout")

    # -- projection ----------------------------------------------------
    # Equirectangular, PDS convention. Verified against the label's own
    # corner coordinates: the west edge solves to sample 1.0 and the south
    # edge to line 14266, both exact.
    def lonlat_to_pixel(self, lat, lon):
        line = self.line_off - lat * self.res
        samp = self.samp_off + ((lon - self.clon) * self.res
                                * math.cos(math.radians(self.clat)))
        return line, samp

    def check_bounds(self, lat, lon):
        return (self.lat_min <= lat <= self.lat_max
                and self.lon_min <= lon <= self.lon_max)

    # -- data ----------------------------------------------------------
    def read_rows(self, l0, l1):
        """Read image lines [l0, l1] inclusive, 1-based, as a float array."""
        l0 = max(1, int(l0))
        l1 = min(self.lines, int(l1))
        b0 = self.data_start + (l0 - 1) * self.record_bytes
        b1 = self.data_start + l1 * self.record_bytes - 1
        raw = get(self.url, b0, b1)
        a = np.frombuffer(raw, dtype="<f4").reshape(l1 - l0 + 1, self.samples)
        return a.astype(np.float64), l0

    def patch(self, lat, lon, half_px):
        line, samp = self.lonlat_to_pixel(lat, lon)
        l0, l1 = int(line - half_px), int(line + half_px)
        rows, got0 = self.read_rows(l0, l1)
        s0 = max(0, int(samp - half_px))
        s1 = min(self.samples, int(samp + half_px))
        sub = rows[:, s0:s1]
        return sub, (line, samp)


def clean(z):
    """Mask PDS missing values and fill them by nearest-valid interpolation."""
    bad = (z <= MISSING * 0.5) | ~np.isfinite(z) | (z < -1e30)
    if bad.all():
        raise SystemExit("patch is entirely no-data -- pick another location")
    if bad.any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(bad, return_distances=False,
                                     return_indices=True)
        z = z[tuple(idx)]
    return z, bad.mean()


def resample(z, n):
    """Bilinear resample to n x n."""
    from scipy.ndimage import map_coordinates
    yy = np.linspace(0, z.shape[0] - 1, n)
    xx = np.linspace(0, z.shape[1] - 1, n)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(z, [Y, X], order=1, mode="nearest")


def slope_stats(z, post):
    gy, gx = np.gradient(z, post)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    return s


def survey(dtm, step=400):
    """Report relief across the DTM so you can pick an interesting patch."""
    print("sampling the DTM for relief (this reads a few strips)...")
    best = []
    for frac in np.linspace(0.08, 0.92, 9):
        line = 1 + frac * (dtm.lines - 1)
        rows, _ = dtm.read_rows(line - 40, line + 40)
        lat = (dtm.line_off - line) / dtm.res
        for sfrac in np.linspace(0.1, 0.9, 5):
            s = int(sfrac * dtm.samples)
            sub = rows[:, max(0, s - 80):s + 80]
            v = sub[(sub > -1e30) & np.isfinite(sub)]
            if v.size < 500:
                continue
            lon = (dtm.clon + (s - dtm.samp_off)
                   / (dtm.res * math.cos(math.radians(dtm.clat))))
            relief = float(v.max() - v.min())
            best.append((relief, lat, lon, float(v.mean())))
    best.sort(reverse=True)
    print(f"\n  {'relief':>8} {'lat':>10} {'lon':>10} {'mean elev':>11}")
    for r, la, lo, me in best[:10]:
        print(f"  {r:7.1f}m {la:10.4f} {lo:10.4f} {me:10.1f}m")
    print("\n  pick one with:  python scripts/fetch_jezero.py --lat LAT --lon LON")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--size", type=float, default=400.0,
                    help="patch width in metres (default 400)")
    ap.add_argument("--res", type=int, default=513,
                    help="heightfield grid resolution (default 513)")
    ap.add_argument("--out", default="assets/jezero.png")
    ap.add_argument("--survey", action="store_true")
    args = ap.parse_args()

    print(f"reading PDS label from\n  {URL}")
    dtm = PdsDtm(URL)
    print(f"  {dtm.lines} x {dtm.samples} posts at {dtm.scale:.4f} m/post")
    print(f"  covers lat {dtm.lat_min:.4f}..{dtm.lat_max:.4f}, "
          f"lon {dtm.lon_min:.4f}..{dtm.lon_max:.4f}")

    if args.survey:
        survey(dtm)
        return

    lat = args.lat if args.lat is not None else LANDING_SITE[0]
    lon = args.lon if args.lon is not None else LANDING_SITE[1]
    named = (args.lat is None and args.lon is None)
    if not dtm.check_bounds(lat, lon):
        raise SystemExit(f"{lat}, {lon} is outside this DTM's coverage")

    half_px = args.size / 2.0 / dtm.scale
    mb = (2 * half_px + 1) * dtm.record_bytes / 1e6
    print(f"\nsite: {lat:.4f} N, {lon:.4f} E"
          + ("  (Octavia E. Butler Landing)" if named else ""))
    print(f"patch: {args.size:.0f} x {args.size:.0f} m  "
          f"-> range-requesting ~{mb:.0f} MB of the 363 MB product")

    z, (line, samp) = dtm.patch(lat, lon, half_px)
    print(f"  DTM pixel (line {line:.1f}, sample {samp:.1f}), raw {z.shape}")

    z, frac_bad = clean(z)
    if frac_bad:
        print(f"  filled {frac_bad*100:.2f}% no-data posts")

    zr = resample(z, args.res)
    post = args.size / (args.res - 1)
    sl = slope_stats(zr, post)
    relief = float(zr.max() - zr.min())

    print(f"\n  elevation  {zr.min():.1f} to {zr.max():.1f} m "
          f"(Mars 2000 datum), relief {relief:.2f} m")
    print(f"  slope      mean {sl.mean():.1f} deg, "
          f"p95 {np.percentile(sl,95):.1f} deg, max {sl.max():.1f} deg")

    # 16-bit PNG, normalised; the true vertical scale goes in the hfield size
    from PIL import Image
    norm = (zr - zr.min()) / (relief if relief > 1e-6 else 1.0)
    path = os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((norm * 65535).astype(np.uint16)).save(path)

    meta = dict(
        source="HiRISE DTEEC_045994_1985_046060_1985_U01 (USGS/UA, PDS3)",
        source_url=URL,
        product_page="https://www.uahirise.org/dtm/dtm.php?ID=ESP_045994_1985",
        site_lat=lat, site_lon=lon,
        site_name="Octavia E. Butler Landing" if named else None,
        patch_m=args.size, grid=args.res, post_m=post,
        elev_min_m=float(zr.min()), elev_max_m=float(zr.max()),
        relief_m=relief,
        slope_mean_deg=float(sl.mean()),
        slope_p95_deg=float(np.percentile(sl, 95)),
        slope_max_deg=float(sl.max()),
        dtm_post_m=dtm.scale,
        # MuJoCo hfield size: radius_x radius_y z_top z_bottom
        hfield_size=[args.size / 2, args.size / 2, relief, 1.0],
    )
    mpath = os.path.join(HERE, "assets", "jezero.json")
    json.dump(meta, open(mpath, "w"), indent=1)

    print(f"\nwrote {args.out} ({args.res}x{args.res}, {post:.2f} m/post)")
    print(f"      assets/jezero.json")
    print(f"\nnow:  python scripts/make_course.py --jezero")


if __name__ == "__main__":
    main()
