"""Shared figure styling and palette for the project."""
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

PALETTE = ["#4C4BA8", "#B07D2B", "#2F7D74", "#A34A6B"]
SEQUENTIAL = ["#EDEDF7", "#C9C8E8", "#9C9AD6", "#6E6BC2", "#4C4BA8"]
EXPOSED, CONTROL = "#4C4BA8", "#B07D2B"
CMAP = LinearSegmentedColormap.from_list("project", SEQUENTIAL)

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
THUMBDIR = FIGDIR / "thumbs"


def apply_style():
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "#1C1C1A",
        "axes.labelcolor": "#1C1C1A",
        "axes.titlecolor": "#1C1C1A",
        "xtick.color": "#6B6B66",
        "ytick.color": "#6B6B66",
        "axes.edgecolor": "#DEDEDA",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#EDECE8",
        "axes.grid": True,
        "axes.axisbelow": True,
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.dpi": 150,
    })


def save(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    THUMBDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", dpi=150)
    fig.savefig(THUMBDIR / f"{name}.png", bbox_inches="tight", dpi=50)
