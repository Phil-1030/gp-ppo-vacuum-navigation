"""
Renders the 20x20 room floor plan as a high-resolution static image.
Output: room_layout.png (300 dpi)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from environment import (
    _build_obstacle_map,
    FREE, WALL, CHARGER, TABLE, CABINET, BED,
    HOTSPOTS,
    GRID_H, GRID_W,
)

# ── Colour scheme ──────────────────────────────────────────────────────────────
TERRAIN = [
    (FREE,    "#ECECEC", "Free space"),
    (WALL,    "#2E2E2E", "Wall"),
    (CHARGER, "#3DBE5A", "Charger"),
    (TABLE,   "#C49A52", "Table"),
    (CABINET, "#6A7FB5", "Cabinet / Shelf"),
    (BED,     "#A882C0", "Bed"),
]

CODES   = [t[0] for t in TERRAIN]
COLOURS = [t[1] for t in TERRAIN]
LABELS  = [t[2] for t in TERRAIN]

cmap = ListedColormap(COLOURS)
norm = BoundaryNorm(boundaries=[c - 0.5 for c in CODES] + [CODES[-1] + 0.5],
                    ncolors=len(CODES))

# ── Build map ──────────────────────────────────────────────────────────────────
obs_map = _build_obstacle_map()

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7.4))
fig.patch.set_facecolor("white")

im = ax.imshow(
    obs_map, cmap=cmap, norm=norm,
    origin="upper", interpolation="nearest",
    extent=[-0.5, GRID_W - 0.5, GRID_H - 0.5, -0.5],
)

# ── Grid lines ─────────────────────────────────────────────────────────────────
ax.set_xticks(np.arange(-0.5, GRID_W, 1), minor=True)
ax.set_yticks(np.arange(-0.5, GRID_H, 1), minor=True)
ax.grid(which="minor", color="#AAAAAA", linewidth=0.35, alpha=0.6)
ax.tick_params(which="minor", length=0)

ax.set_xticks(np.arange(0, GRID_W, 2))
ax.set_yticks(np.arange(0, GRID_H, 2))
ax.tick_params(labelsize=8, colors="#333333")
ax.set_xlabel("Column", fontsize=10, labelpad=6)
ax.set_ylabel("Row", fontsize=10, labelpad=6)

# ── Hotspot markers ────────────────────────────────────────────────────────────
for h in HOTSPOTS:
    r0, c0 = h["center"]
    ax.plot(c0, r0, marker="x", color="#FF4444", markersize=6,
            linewidth=1.4, zorder=3)

# ── Charger label ──────────────────────────────────────────────────────────────
ax.text(1, 18.15, "PWR", fontsize=5.5, color="white",
        ha="center", va="center", fontweight="bold", zorder=4)

# ── Room zone annotations ──────────────────────────────────────────────────────
zone_style = dict(fontsize=8, color="#555555", ha="center", va="center",
                  style="italic", zorder=4)
ax.text(5,  4,  "Bedroom A", **zone_style)
ax.text(14, 4,  "Bedroom B", **zone_style)
ax.text(9.5, 13, "Living Room", **zone_style)

# ── Doorway arrows ─────────────────────────────────────────────────────────────
arrow_kw = dict(arrowstyle="<->", color="#888888", lw=0.9)
ax.annotate("", xy=(5, 8.5), xytext=(5, 7.5),
            arrowprops=arrow_kw)
ax.annotate("", xy=(15, 8.5), xytext=(15, 7.5),
            arrowprops=arrow_kw)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=c, edgecolor="#888888", linewidth=0.6, label=l)
    for c, l in zip(COLOURS, LABELS)
]
hotspot_marker = plt.Line2D([0], [0], marker="x", color="#FF4444",
                             linewidth=0, markersize=7, label="Dirt hotspot centre")
legend_patches.append(hotspot_marker)

ax.legend(
    handles=legend_patches,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    fontsize=8.5,
    frameon=True,
    edgecolor="#CCCCCC",
    title="Legend",
    title_fontsize=9,
)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.set_title("Robot Vacuum Environment — Room Floor Plan (20 × 20 Grid)",
             fontsize=11, fontweight="bold", pad=10)

# ── Export ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
fig.savefig("room_layout.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: room_layout.png")
