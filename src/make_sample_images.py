"""Draw the raw and cleaned sample files as styled PNG tables for the website.

Input: data/samples/. Output: ten PNG tables in figures/samples/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plotstyle import FIGDIR, PALETTE, SEQUENTIAL, apply_style

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
OUT_DIR = FIGDIR / "samples"
MAX_ROWS, MAX_COLS, MAX_CHARS = 6, 6, 40
TEXT, EDGE = "#1C1C1A", "#DEDEDA"
FONT_SIZE, DPI = 9, 150
PAD_IN, ROW_IN, TITLE_IN = 0.08, 0.3, 0.5  # sizes in inches

# (output name, title, columns) per image; rows are matched on the id column
HN_RAW = ("hn_raw", "Hacker News: raw", ["id", "by", "time", "thread_month", "parent", "text"])
HN_CLEAN = ("hn_cleaned", "Hacker News: cleaned",
            ["company_clean", "role", "location_clean", "work_mode", "posted_date", "is_repost"])
ATS_RAW = ("ats_raw", "ATS job postings: raw",
           ["companyName", "title", "location", "salaryMin", "salaryInterval", "postedAt"])
ATS_CLEAN = ("ats_cleaned", "ATS job postings: cleaned",
             ["company_clean", "title", "seniority", "role_family", "salary_annual_min",
              "location_clean"])

# The three numeric sources arrived tidy, so each cleaned table repeats the
# raw numbers and adds the derived and flag columns.
INDEED_RAW = ("indeed_raw", "Indeed Hiring Lab: raw",
              ["date", "jobcountry", "indeed_job_postings_index", "variable", "display_name"])
INDEED_CLEAN = ("indeed_cleaned", "Indeed Hiring Lab: cleaned",
                ["date", "indeed_job_postings_index", "variable", "display_name", "sector_group",
                 "is_outlier_indeed_job_postings_index"])
BLS_RAW = ("bls_raw", "BLS JOLTS: raw", ["seriesID", "year", "period", "periodName", "value"])
BLS_CLEAN = ("bls_cleaned", "BLS JOLTS: cleaned",
             ["series_id", "series_name", "date", "value", "is_outlier", "value_suspect"])
OEWS_RAW = ("oews_raw", "OEWS annual files: raw",
            ["source_file", "occ_code", "occ_title", "tot_emp", "a_median", "a_mean"])
OEWS_CLEAN = ("oews_cleaned", "OEWS annual files: cleaned",
              ["occ_code", "occ_title", "year", "tot_emp", "a_median", "wage_topcoded"])


def read_csv(name: str) -> pd.DataFrame:
    """Read a sample CSV with every value kept as the text stored in the file."""
    return pd.read_csv(SAMPLES_DIR / name, dtype=str, keep_default_na=False)


def read_json(name: str) -> pd.DataFrame:
    """Read a sample JSON list of records with every value kept as text."""
    return pd.DataFrame(json.loads((SAMPLES_DIR / name).read_text())).astype(str)


def take(df: pd.DataFrame, key: str, ids: list[str]) -> pd.DataFrame:
    """Return the rows whose key is in ids, in the order given."""
    return df.drop_duplicates(key).set_index(key).loc[ids].reset_index()


def matched(raw: pd.DataFrame, clean: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the same first records from the raw and cleaned tables."""
    in_clean = set(clean[key])
    ids = [i for i in dict.fromkeys(raw[key]) if i in in_clean][:MAX_ROWS]
    return take(raw, key, ids), take(clean, key, ids)


def truncate(value: str) -> str:
    """Collapse whitespace, show blanks as NA, and cut text to 40 characters."""
    text = " ".join(str(value).split()) or "NA"
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


def text_inches(fig: plt.Figure, text: str, bold: bool) -> float:
    """Measure how wide a piece of text is, in inches, at the table font size."""
    artist = fig.text(0, 0, text, fontsize=FONT_SIZE, fontweight="bold" if bold else "normal")
    width = artist.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    artist.remove()
    return width


def render(df: pd.DataFrame, name: str, title: str, columns: list[str]) -> Path:
    """Draw one styled table and save it as a PNG in figures/samples/."""
    columns = columns[:MAX_COLS]
    cells = [[truncate(v) for v in row] for row in df[columns].head(MAX_ROWS).itertuples(index=False)]

    fig = plt.figure(dpi=DPI)
    widths = [max([text_inches(fig, col, bold=True)]
                  + [text_inches(fig, row[i], bold=False) for row in cells]) + 2 * PAD_IN
              for i, col in enumerate(columns)]
    total_w, body_h = sum(widths), ROW_IN * (len(cells) + 1)
    fig.set_size_inches(total_w, body_h + TITLE_IN)

    ax = fig.add_axes((0, 0, 1, body_h / (body_h + TITLE_IN)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, colWidths=[w / total_w for w in widths],
                     cellLoc="left", colLoc="left", bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.PAD = PAD_IN / widths[col]
        if row == 0:
            cell.set_facecolor(PALETTE[0])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(SEQUENTIAL[0] if row % 2 == 0 else "white")
            cell.get_text().set_color(TEXT)
    fig.text(0, 1 - 0.45 * TITLE_IN / (body_h + TITLE_IN), title, fontsize=13, color=TEXT,
             va="center")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def main() -> None:
    """Build the ten sample table images."""
    apply_style()
    hn_raw, hn_clean = matched(read_json("hn_raw_sample.json"),
                               read_csv("hn_cleaned_sample.csv"), "id")
    ats_raw, ats_clean = matched(read_csv("ats_raw_sample.csv"),
                                 read_csv("ats_final_sample.csv"), "jobId")
    # make_samples.py writes the Indeed, BLS and OEWS pairs row for row, so
    # those tables line up without matching on a key here.
    pairs = [(hn_raw, HN_RAW), (hn_clean, HN_CLEAN), (ats_raw, ATS_RAW), (ats_clean, ATS_CLEAN),
             (read_csv("indeed_sector_raw_sample.csv"), INDEED_RAW),
             (read_csv("indeed_sector_clean_sample.csv"), INDEED_CLEAN),
             (read_csv("bls_jolts_raw_sample.csv"), BLS_RAW),
             (read_csv("bls_jolts_clean_sample.csv"), BLS_CLEAN),
             (read_csv("oews_raw_sample.csv"), OEWS_RAW),
             (read_csv("oews_clean_sample.csv"), OEWS_CLEAN)]
    for df, spec in pairs:
        print(f"saved {render(df, *spec)}")


if __name__ == "__main__":
    main()
