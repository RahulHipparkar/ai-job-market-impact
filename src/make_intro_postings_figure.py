"""Draw the software development postings chart for the Introduction page of the website.

Input: data/processed/indeed_sector.csv. Output: figures/intro_softdev_postings.png and its
thumbnail.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from make_intro_figure import EVENTS, MUTED
from plotstyle import PALETTE, apply_style, save

SECTOR_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "indeed_sector.csv"
SECTOR = "Software Development"


def load_postings() -> pd.Series:
    """Return the monthly mean of Indeed's daily software development postings index."""
    df = pd.read_csv(SECTOR_CSV, parse_dates=["date"])
    df = df[(df.display_name == SECTOR) & (df.variable == "total postings")]
    return df.set_index("date")["indeed_job_postings_index"].resample("MS").mean()


def main() -> None:
    """Build and save the software development postings chart."""
    apply_style()
    postings = load_postings()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(postings.index, postings.values, color=PALETTE[0], lw=2.2)
    ax.axhline(100, color="#DEDEDA", lw=1)
    top = postings.max() * 1.12
    for day, label, align in EVENTS:
        x = pd.Timestamp(day)
        ax.axvline(x, color=MUTED, ls="--", lw=1)
        nudge = pd.Timedelta(days=-30 if align == "right" else 30)
        ax.text(x + nudge, top * 0.98, label, ha=align, va="top", fontsize=9, color=MUTED)

    ax.set_ylim(0, top)
    ax.set_xlim(postings.index.min(), postings.index.max())
    ax.set_title("Software development job postings on Indeed, 2020 to 2026")
    ax.set_xlabel("")
    ax.set_ylabel("Index (1 February 2020 = 100)")
    save(fig, "intro_softdev_postings")
    print(f"saved intro_softdev_postings.png ({postings.index.min():%b %Y} to "
          f"{postings.index.max():%b %Y}; peak {postings.max():.0f} in {postings.idxmax():%b %Y}; "
          f"last {postings.iloc[-1]:.0f})")


if __name__ == "__main__":
    main()
