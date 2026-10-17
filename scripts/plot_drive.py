#!/usr/bin/env python3
"""
plot_drive.py -- figure of the autonav result: terrain, hazard map, the
planned route and the track the rover actually drove.

    python scripts/autonav.py --model course_jezero_boulder.xml --goal 18 6 --span 40
    python scripts/plot_drive.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")   # MJCF + assets
OUT = os.path.join(ROOT, "out")         # generated results
HERE = OUT

# Validated palette. Sequential = one blue hue, light -> dark (magnitude).
# Two categorical hues for the two routes; both direct-labelled, which is the
# relief the validator requires for aqua's sub-3:1 contrast.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
SEQ_WARM = ["#fde8dd", "#f8c4a5", "#f39b6f", "#eb6834", "#c14f23", "#8c3715"]
PLANNED = "#eb6834"      # categorical slot 2, orange
DRIVEN = "#1baf7a"       # categorical slot 3, aqua
BLOCKED = "#e34948"      # status: critical -- ships with a label, never alone
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    hz = np.load(os.path.join(HERE, "hazard_map.npz"), allow_pickle=True) \
        if os.path.exists(os.path.join(HERE, "hazard_map.npz")) else None
    dl = np.load(os.path.join(HERE, "drive_log.npz"), allow_pickle=True)
    log, path = dl["log"], dl["path"]

    if hz is None:
        raise SystemExit("run:  python scripts/autonav.py --survey --span 40 --res 0.25")

    ax_m = hz["axis"]
    ext = [ax_m[0], ax_m[-1], ax_m[0], ax_m[-1]]
    Z, cost, blocked = hz["Z"], hz["cost"], hz["blocked"]
    # arrays are [i=x, j=y]; imshow wants [row=y, col=x]
    Zp, Cp, Bp = Z.T, cost.T, blocked.T

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=SURFACE)
    fig.suptitle("Autonomous traverse on the real Jezero DTM",
                 color=INK, fontsize=14, fontweight="600", x=0.055, ha="left",
                 y=0.975)
    fig.text(0.055, 0.925,
             "HiRISE terrain → hazard map → A* route → "
             "safety check → driven in the digital twin",
             color=INK2, fontsize=10, ha="left")

    # -- panel 1: terrain + routes -------------------------------------
    a = axes[0]
    style(a)
    cm_seq = LinearSegmentedColormap.from_list("elev", SEQ_BLUE)
    im = a.imshow(Zp, origin="lower", extent=ext, cmap=cm_seq,
                  interpolation="bilinear")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("elevation above hfield base (m)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)

    a.plot(path[:, 0], path[:, 1], color=PLANNED, lw=2.0, zorder=5)
    a.plot(path[:, 0], path[:, 1], "o", color=PLANNED, ms=8,
           mec=SURFACE, mew=1.6, zorder=6)
    a.plot(log[:, 1], log[:, 2], color=DRIVEN, lw=2.0, zorder=7)
    a.plot([log[0, 1]], [log[0, 2]], "o", color=INK, ms=9, mec=SURFACE,
           mew=1.6, zorder=8)

    # direct labels rather than relying on colour alone
    mid = len(path) // 2
    a.annotate("planned", (path[mid, 0], path[mid, 1]),
               textcoords="offset points", xytext=(6, 12),
               color=PLANNED, fontsize=10, fontweight="600", zorder=9)
    k = int(len(log) * 0.62)
    a.annotate("driven", (log[k, 1], log[k, 2]),
               textcoords="offset points", xytext=(6, -18),
               color=DRIVEN, fontsize=10, fontweight="600", zorder=9)
    a.annotate("start", (log[0, 1], log[0, 2]), textcoords="offset points",
               xytext=(-30, -6), color=INK2, fontsize=9)
    a.set_title("terrain and route", color=INK, fontsize=11, loc="left",
                pad=8)
    a.set_xlabel("x, rover frame (m)", color=INK2, fontsize=9)
    a.set_ylabel("y (m)", color=INK2, fontsize=9)
    a.set_aspect("equal")

    # -- panel 2: hazard cost ------------------------------------------
    b = axes[1]
    style(b)
    finite = np.isfinite(Cp)
    vmax = np.percentile(Cp[finite], 99.5) if finite.any() else 1.0
    cm_cost = LinearSegmentedColormap.from_list("cost", SEQ_WARM)
    im2 = b.imshow(np.where(finite, Cp, np.nan), origin="lower", extent=ext,
                   cmap=cm_cost, vmin=0, vmax=vmax, interpolation="nearest")
    cb2 = fig.colorbar(im2, ax=b, fraction=0.046, pad=0.02)
    cb2.set_label("traversal cost (tilt + roughness + obstacle)",
                  color=INK2, fontsize=9)
    cb2.ax.tick_params(colors=INK2, labelsize=8)
    cb2.outline.set_visible(False)

    b.imshow(np.ma.masked_where(~Bp, Bp), origin="lower", extent=ext,
             cmap=ListedColormap([BLOCKED]), alpha=0.85, interpolation="nearest")
    b.plot(path[:, 0], path[:, 1], color=PLANNED, lw=2.0, zorder=5)
    b.plot(path[:, 0], path[:, 1], "o", color=PLANNED, ms=8, mec=SURFACE,
           mew=1.6, zorder=6)
    b.annotate("blocked: hazard +\nrover-radius dilation",
               (0.03, 0.965), xycoords="axes fraction", va="top",
               color=BLOCKED, fontsize=9.5, fontweight="600", zorder=9)
    b.annotate("planned route", (0.03, 0.06), xycoords="axes fraction",
               color=PLANNED, fontsize=10, fontweight="600", zorder=9)
    b.set_title("hazard map the planner searched", color=INK, fontsize=11,
                loc="left", pad=8)
    b.set_xlabel("x, rover frame (m)", color=INK2, fontsize=9)
    b.set_aspect("equal")

    planned_m = float(dl["planned_m"])
    driven_m = float(dl["driven_m"])
    fig.text(0.055, 0.035,
             f"planned {planned_m:.1f} m  ·  driven {driven_m:.1f} m  "
             f"·  max tilt {log[:,4].max():.2f}°  ·  "
             f"{log[-1,0]/60:.1f} min sim",
             color=INK2, fontsize=9.5, ha="left")

    fig.tight_layout(rect=[0, 0.055, 1, 0.91])
    out = os.path.join(HERE, "autonav_result.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
