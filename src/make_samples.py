"""Write raw and cleaned sample CSVs for the three tidy numeric sources.

Input: data/raw/ and data/processed/. Output: six 20-row CSVs in
data/samples/, a raw and a cleaned one per source. Each pair holds the
same records in the same order, so make_sample_images.py can draw the two
tables side by side without matching keys.

The Hacker News and job board samples are written by their own cleaning
scripts. These three sources have none, because clean_indeed.py,
clean_bls.py and clean_oews_files.py write only processed tables.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLES_DIR = ROOT / "data" / "samples"
N_ROWS = 20

# One series per source, matching the figures on the site.
INDEED_SECTOR, INDEED_VARIABLE = "Software Development", "total postings"
JOLTS_SERIES = "JTS510000000000000JOL"  # Information: job openings


def latest_bls_file(pattern: str) -> Path:
    """Return the most recently modified raw BLS file matching a glob pattern."""
    return max((RAW_DIR / "bls").glob(pattern), key=lambda p: p.stat().st_mtime)


def indeed_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One sector series: the cleaned rows and the raw rows they came from."""
    clean = pd.read_csv(PROCESSED_DIR / "indeed_sector.csv")
    clean = clean[(clean["display_name"] == INDEED_SECTOR)
                  & (clean["variable"] == INDEED_VARIABLE)].head(N_ROWS).reset_index(drop=True)

    raw = pd.read_csv(RAW_DIR / "indeed" / "job_postings_tracker" / "US" / "job_postings_by_sector_US.csv")
    keys = ["date", "display_name", "variable"]
    matched = clean[keys].merge(raw, on=keys, how="left")[raw.columns]
    return matched, clean


def bls_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One JOLTS series: the cleaned rows and the raw API records behind them."""
    clean = pd.read_csv(PROCESSED_DIR / "bls_jolts.csv")
    clean = clean[clean["series_id"] == JOLTS_SERIES].head(N_ROWS).reset_index(drop=True)

    # Flatten series[].data[] without renaming, so the sample keeps the API's own fields.
    payload = json.loads(latest_bls_file("jolts_*.json").read_text())[0]
    raw = pd.DataFrame([{"seriesID": s["seriesID"], "year": d["year"], "period": d["period"],
                         "periodName": d["periodName"], "latest": d.get("latest", ""),
                         "value": d["value"], "footnotes": json.dumps(d["footnotes"])}
                        for s in payload["Results"]["series"] for d in s["data"]])

    date = pd.to_datetime(clean["date"])
    keys = pd.DataFrame({"seriesID": clean["series_id"], "year": date.dt.year.astype(str),
                         "period": "M" + date.dt.month.map("{:02d}".format)})
    matched = keys.merge(raw, on=["seriesID", "year", "period"], how="left")[raw.columns]
    return matched, clean


def oews_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trend rows and the annual workbook rows they were read from.

    The workbooks carry no year column -- the year is in the file name -- so
    the raw sample keeps source_file to show which year each row came from.
    """
    clean = pd.read_csv(PROCESSED_DIR / "oews_trend.csv").head(N_ROWS).reset_index(drop=True)

    frames = []
    for year in sorted(clean["year"].unique()):
        path = RAW_DIR / "bls" / "oews" / f"national_M{year}_dl.xlsx"
        df = pd.read_excel(path, sheet_name=path.stem)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[df["occ_code"].isin(clean["occ_code"].unique())].copy()
        df.insert(0, "source_file", path.name)
        df["_year"] = year
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    columns = [c for c in raw.columns if c != "_year"]
    matched = clean[["occ_code", "year"]].merge(
        raw, left_on=["occ_code", "year"], right_on=["occ_code", "_year"], how="left")[columns]
    return matched, clean


def write(df: pd.DataFrame, name: str, value_col: str) -> None:
    """Save one sample CSV, first checking every row found its match."""
    assert df[value_col].notna().all(), f"{name}: {int(df[value_col].isna().sum())} rows without a match"
    path = SAMPLES_DIR / name
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df)} rows, {len(df.columns)} columns)")


def main() -> None:
    """Write the six sample CSVs for the Indeed, BLS and OEWS sources."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    indeed_raw, indeed_clean = indeed_samples()
    bls_raw, bls_clean = bls_samples()
    oews_raw, oews_clean = oews_samples()

    write(indeed_raw, "indeed_sector_raw_sample.csv", "indeed_job_postings_index")
    write(indeed_clean, "indeed_sector_clean_sample.csv", "indeed_job_postings_index")
    write(bls_raw, "bls_jolts_raw_sample.csv", "value")
    write(bls_clean, "bls_jolts_clean_sample.csv", "value")
    write(oews_raw, "oews_raw_sample.csv", "tot_emp")
    write(oews_clean, "oews_clean_sample.csv", "tot_emp")


if __name__ == "__main__":
    main()
