"""Shared matplotlib styling so notebook figures match the site theme.

Usage: import at the top of a notebook before plotting.

    from plotstyle import PALETTE, CMAP
"""

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PALETTE = ["#4C4BA8", "#B07D2B", "#2F7D74", "#A34A6B"]
SEQUENTIAL = ["#EDEDF7", "#C9C8E8", "#9C9AD6", "#6E6BC2", "#4C4BA8"]

CMAP = LinearSegmentedColormap.from_list("project", SEQUENTIAL)

_INK = "#1C1C1A"
_TICK = "#6B6B66"
_SPINE = "#DEDEDA"
_GRID = "#DEDEDA"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": _INK,
        "axes.labelcolor": _INK,
        "xtick.color": _TICK,
        "ytick.color": _TICK,
        "axes.edgecolor": _SPINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.prop_cycle": cycler(color=PALETTE),
    }
)
