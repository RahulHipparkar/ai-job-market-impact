"""Reshape and clean the Indeed Hiring Lab trackers into tidy per-series CSVs.

Both source repos are already tidy, long-format data, so this is mostly
reshaping -- but every output still gets a full missing/duplicate/outlier/
range pass. Nothing is imputed or silently dropped; issues are flagged.
"""

import re
from pathlib import Path

import pandas as pd

AI_TRACKER = Path(__file__).resolve().parent.parent / "data" / "raw" / "indeed" / "ai-tracker" / "AI_posting.csv"
US_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "indeed" / "job_postings_tracker" / "US"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "indeed_clean_report.txt"

# The brief names 4 ai_exposed sectors; "Information Design & Documentation"
# is not present in the current US sector file (see report) -- everything
# not listed here defaults to "control".
SECTOR_GROUP = {
    "Software Development": "ai_exposed",
    "Data & Analytics": "ai_exposed",
    "IT Systems & Solutions": "ai_exposed",
    "Information Design & Documentation": "ai_exposed",
}
SNAKE_RE = re.compile(r"(?<!^)(?<![A-Z_])(?=[A-Z])")

def classify_sector(name: str) -> str:
    """Return 'ai_exposed' for the 4 named sectors, else 'control'."""
    return SECTOR_GROUP.get(name, "control")

def to_snake(name: str) -> str:
    """Mechanically convert one column name to snake_case."""
    return SNAKE_RE.sub("_", name).lower()

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Snake_case every column and strip whitespace from string columns."""
    df = df.rename(columns={c: to_snake(c) for c in df.columns})
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    return df

def load(path: Path) -> pd.DataFrame:
    """Read a tracker CSV, snake_case + strip it, and sort by date."""
    return normalize(pd.read_csv(path, parse_dates=["date"])).sort_values("date").reset_index(drop=True)

def missing_report(df: pd.DataFrame, label: str, report: list[str]) -> None:
    """Print a missing-count/pct table for every column."""
    report.append("  missing values:")
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n:
            report.append(f"    {c}: {n} ({n / len(df):.2%})")
    if not df.isna().any().any():
        report.append("    none")

def date_gap_report(df: pd.DataFrame, report: list[str], group_cols: list[str], freq: str) -> None:
    """Report missing steps in the daily/monthly date index, per group."""
    n_series = df[group_cols].drop_duplicates().shape[0]
    gap_series, total_gaps = 0, 0
    for _, grp in df.groupby(group_cols):
        full = pd.date_range(grp["date"].min(), grp["date"].max(), freq=freq)
        n_missing = len(full.difference(grp["date"]))
        if n_missing:
            gap_series += 1
            total_gaps += n_missing
    report.append(f"  date gaps: {total_gaps} missing {freq} steps across {gap_series} of {n_series} series "
                   "(a missing month/day here is distinct from a null value in a present row)")

def dedupe(df: pd.DataFrame, report: list[str], key_cols: list[str]) -> pd.DataFrame:
    """Drop exact duplicate rows; report (never drop) rows sharing a key with conflicting values."""
    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    key_dupes = df[df.duplicated(subset=key_cols, keep=False)]
    n_conflict_keys = key_dupes[key_cols].drop_duplicates().shape[0]
    report.append(f"  duplicates: {n_exact} exact duplicate rows dropped; "
                   f"{n_conflict_keys} keys have conflicting values across rows (NOT dropped, needs a decision)")
    if n_conflict_keys:
        report.append(f"    example conflicting rows:\n{key_dupes.head(4).to_string()}")
    return df

def flag_outliers(df: pd.DataFrame, col: str, group_cols: list[str], report: list[str], top_n: int = 10) -> pd.DataFrame:
    """Flag |z-score| > 3 within each group (not pooled); report per-series stats and the most extreme rows."""
    g = df.groupby(group_cols)[col]
    z = (df[col] - g.transform("mean")) / g.transform("std").replace(0, pd.NA)
    out_col = f"is_outlier_{col}"
    df[out_col] = z.abs().gt(3).fillna(False)
    stats = df.groupby(group_cols)[col].agg(min="min", p1=lambda s: s.quantile(.01),
                                              p99=lambda s: s.quantile(.99), max="max")
    report.append(f"  outliers ({col}): {int(df[out_col].sum())} rows flagged across "
                   f"{stats.shape[0]} series (|z|>3 computed per series, not pooled)")
    report.append(f"    per-series min/p1/p99/max:\n{stats.to_string()}")
    order = z.abs().sort_values(ascending=False).head(top_n).index
    top = df.loc[order, ["date", *group_cols, col]].copy()
    top["z"] = z.loc[order].round(1)
    report.append(f"    {top_n} most extreme {col} values:\n{top.to_string(index=False)}")
    return df

def flag_range(df: pd.DataFrame, col: str, low: float, high: float, report: list[str], note: str = "") -> pd.DataFrame:
    """Flag values outside [low, high] as suspect; report the count."""
    out_col = f"{col}_suspect"
    df[out_col] = (df[col] < low) | (df[col] > high)
    report.append(f"  range check ({col} outside [{low}, {high}]{note}): {int(df[out_col].sum())} rows flagged")
    return df

def add_year_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Add year and quarter columns derived from the date column."""
    df["year"], df["quarter"] = df["date"].dt.year, df["date"].dt.quarter
    return df

def clean(df: pd.DataFrame, label: str, key_cols: list[str], group_cols: list[str], value_col: str,
          report: list[str], freq: str = "D") -> pd.DataFrame:
    """Run the full missing/gap/duplicate/outlier pass shared by every output file."""
    report.append(f"\n--- cleaning pass: {label} ---")
    missing_report(df, label, report)
    date_gap_report(df, report, group_cols, freq)
    df = dedupe(df, report, key_cols)
    df = flag_outliers(df, value_col, group_cols, report)
    df = add_year_quarter(df)
    return df

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    # AI share of postings, 9 countries
    ai = load(AI_TRACKER)
    report.append("=== indeed_ai_share.csv (AI_posting.csv) ===")
    report.append(f"rows: {len(ai)}, countries: {sorted(ai['jobcountry'].unique())}")
    ai = clean(ai, "ai_share", ["date", "jobcountry"], ["jobcountry"], "ai_share_postings", report)
    ai = flag_range(ai, "ai_share_postings", 0, 100, report, " -- this is a % share, not the Feb2020=100 index")

    # National US aggregate -- swept in by the US/*.csv glob but not in the
    # requested output list; written to indeed_national.csv rather than
    # silently dropped. Flag for review.
    nat = load(US_DIR / "aggregate_job_postings_US.csv")
    report.append("\n=== indeed_national.csv (aggregate_job_postings_US.csv) -- NOT in requested outputs, kept anyway ===")
    nat = clean(nat, "national", ["date", "variable"], ["variable"], "indeed_job_postings_index_sa", report)
    nat = flag_outliers(nat, "indeed_job_postings_index_nsa", ["variable"], report)
    nat = flag_range(nat, "indeed_job_postings_index_sa", 0, 500, report, " -- Feb2020=100 index")
    nat = flag_range(nat, "indeed_job_postings_index_nsa", 0, 500, report, " -- Feb2020=100 index")

    # Sector index: total postings + new postings, both kept
    sector = load(US_DIR / "job_postings_by_sector_US.csv")
    sector["sector_group"] = sector["display_name"].map(classify_sector)
    report.append("\n=== indeed_sector.csv (job_postings_by_sector_US.csv) ===")
    report.append("variable breakdown (both kept): " +
                   ", ".join(f"{v!r}={c}" for v, c in sector["variable"].value_counts().items()))
    sectors_found = sorted(sector["display_name"].unique())
    report.append(f"sectors found: {len(sectors_found)} (brief specifies 44)")
    missing_named = [s for s in SECTOR_GROUP if s not in sectors_found]
    if missing_named:
        report.append(f"  NOTE: named ai_exposed in the brief but absent from this data: {missing_named}")
    report.append("full sector -> group assignment (edit SECTOR_GROUP to move any):")
    for name in sectors_found:
        report.append(f"  [{classify_sector(name):>10}] {name}")
    sector = clean(sector, "sector", ["date", "display_name", "variable"], ["display_name", "variable"],
                    "indeed_job_postings_index", report)
    sector = flag_range(sector, "indeed_job_postings_index", 0, 500, report, " -- Feb2020=100 index")
    report.append("NOTE: index is relative to Feb 2020 = 100 -- comparable across sectors/geographies "
                   "over time, but NOT a count of postings.")

    # Metro
    metro = load(US_DIR / "metro_job_postings_us.csv")
    metro[["metro_name", "metro_state"]] = metro["metro"].str.split(", ", n=1, expand=True)
    report.append("\n=== indeed_metro.csv (metro_job_postings_us.csv) ===")
    report.append(f"metro split into metro_name/metro_state, e.g. {metro['metro'].iloc[0]!r} -> "
                   f"({metro['metro_name'].iloc[0]!r}, {metro['metro_state'].iloc[0]!r})")
    metro = clean(metro, "metro", ["date", "cbsa_code"], ["cbsa_code"], "indeed_job_postings_index", report)
    metro = flag_range(metro, "indeed_job_postings_index", 0, 500, report, " -- Feb2020=100 index")

    # State
    state = load(US_DIR / "state_job_postings_us.csv")
    report.append("\n=== indeed_state.csv (state_job_postings_us.csv) ===")
    state = clean(state, "state", ["date", "state"], ["state"], "indeed_job_postings_index", report)
    state = flag_range(state, "indeed_job_postings_index", 0, 500, report, " -- Feb2020=100 index")

    report.append("\nNORMALISATION applied to every file above: snake_case columns, stripped string "
                   "whitespace, 'date' as the sole date column name/dtype (datetime64[ns] before CSV write), "
                   "year/quarter added for later aggregation.")

    report_text = "\n".join(report)
    print(report_text)
    REPORT_PATH.write_text(report_text + "\n")

    ai.to_csv(PROCESSED_DIR / "indeed_ai_share.csv", index=False)
    nat.to_csv(PROCESSED_DIR / "indeed_national.csv", index=False)
    sector.to_csv(PROCESSED_DIR / "indeed_sector.csv", index=False)
    metro.to_csv(PROCESSED_DIR / "indeed_metro.csv", index=False)
    state.to_csv(PROCESSED_DIR / "indeed_state.csv", index=False)

if __name__ == "__main__":
    main()
