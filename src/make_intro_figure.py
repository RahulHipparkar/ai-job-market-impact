"""Draw the US job openings chart for the Introduction page of the website.

Input: data/processed/bls_jolts.csv. Output: figures/intro_job_openings.png and its thumbnail.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plotstyle import PALETTE, apply_style, save

JOLTS_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "bls_jolts.csv"
SERIES = "Total nonfarm: job openings"
MUTED = "#6B6B66"
# (date, label, which side of the line the label sits on)
EVENTS = [
    ("2022-03-16", "First interest rate rise", "right"),
    ("2022-11-30", "ChatGPT released", "left"),
]


def load_openings() -> pd.Series:
    """Return monthly US job openings in millions."""
    df = pd.read_csv(JOLTS_CSV, parse_dates=["date"])
    return df[df.series_name == SERIES].set_index("date")["value"].sort_index() / 1000


def main() -> None:
    """Build and save the job openings chart."""
    apply_style()
    openings = load_openings()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(openings.index, openings.values, color=PALETTE[0], lw=2.2)
    top = openings.max() * 1.12
    for day, label, align in EVENTS:
        x = pd.Timestamp(day)
        ax.axvline(x, color=MUTED, ls="--", lw=1)
        nudge = pd.Timedelta(days=-30 if align == "right" else 30)
        ax.text(x + nudge, top * 0.98, label, ha=align, va="top", fontsize=9, color=MUTED)

    ax.set_ylim(0, top)
    ax.set_xlim(openings.index.min(), openings.index.max())
    ax.set_title("Job openings in the United States, 2015 to 2026")
    ax.set_xlabel("")
    ax.set_ylabel("Job openings (millions)")
    save(fig, "intro_job_openings")
    print(f"saved intro_job_openings.png ({openings.index.min():%b %Y} to {openings.index.max():%b %Y})")


if __name__ == "__main__":
    main()
