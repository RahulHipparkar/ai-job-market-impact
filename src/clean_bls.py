"""Flatten the BLS API pulls (JOLTS, CES, OEWS) into tidy long-format CSVs.

The API responses are already clean -- this is mostly reshaping (nested
JSON to columns), not repair -- but every survey still gets a full
missing/duplicate/outlier/range pass. Nothing imputed, only flagged.
"""

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bls"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "bls_clean_report.txt"
CATALOG = json.loads((RAW_DIR / "_series_catalog.json").read_text())
UNITS = {
    "bls_jolts": "levels in thousands (BLS JOLTS national convention)",
    "bls_ces": "all-employee counts in thousands",
    "bls_oews": "mixed within the survey: 'employment' series = raw headcount, "
                "'median annual wage' series = USD/year -- see series_name",
}

def latest_file(pattern: str) -> Path:
    """Return the most recently modified raw file matching a glob pattern."""
    return max(RAW_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)

def series_name(series_id: str) -> str:
    """Look up a readable series name from the local catalog."""
    return CATALOG.get(series_id, {}).get("name", series_id)

def fmt_footnotes(footnotes: list[dict]) -> str:
    """Join non-empty footnote entries into a single 'code: text' string."""
    return "; ".join(f"{fn['code']}: {fn['text']}" for fn in footnotes if fn)

def period_to_date(year: str, period: str) -> pd.Timestamp | None:
    """Convert BLS year+period to a first-of-month/year Timestamp; M13 (annual avg) -> None."""
    if period == "M13":
        return None
    if period.startswith("M"):
        return pd.Timestamp(int(year), int(period[1:]), 1)
    return pd.Timestamp(int(year), 1, 1)  # OEWS uses "A01" for its single annual figure

def flatten(path: Path) -> pd.DataFrame:
    """Flatten one BLS API response's Results.series[].data[] into a long dataframe."""
    payload = json.loads(path.read_text())[0]
    rows = []
    for s in payload["Results"]["series"]:
        for d in s["data"]:
            rows.append({
                "series_id": s["seriesID"].strip(),
                "series_name": series_name(s["seriesID"]).strip(),
                "year": d["year"], "period": d["period"],
                "value_raw": d["value"],
                "footnotes": fmt_footnotes(d["footnotes"]),
            })
    return pd.DataFrame(rows)

def missing_report(df: pd.DataFrame, report: list[str]) -> None:
    """Print a missing-count/pct table for every column (empty-string footnotes are not missing)."""
    report.append("  missing values:")
    any_missing = False
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n:
            any_missing = True
            report.append(f"    {c}: {n} ({n / len(df):.2%})")
    if not any_missing:
        report.append("    none")

def date_gap_report(df: pd.DataFrame, report: list[str]) -> None:
    """Report missing months in each series' date index (monthly except OEWS's single annual point)."""
    n_series, gap_series, total_gaps = df["series_id"].nunique(), 0, 0
    for sid, grp in df.groupby("series_id"):
        if len(grp) < 2:
            continue  # OEWS: a single annual observation has no internal gaps to check
        full = pd.date_range(grp["date"].min(), grp["date"].max(), freq="MS")
        n_missing = len(full.difference(grp["date"]))
        if n_missing:
            gap_series += 1
            total_gaps += n_missing
            report.append(f"    {sid} missing months: {n_missing}")
    report.append(f"  date gaps: {total_gaps} missing monthly steps across {gap_series} of {n_series} series")

def dedupe(df: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """Drop exact duplicate rows; report (never drop) rows sharing (series_id, date) with conflicting values."""
    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    key_dupes = df[df.duplicated(subset=["series_id", "date"], keep=False)]
    n_conflict_keys = key_dupes[["series_id", "date"]].drop_duplicates().shape[0]
    report.append(f"  duplicates: {n_exact} exact duplicate rows dropped; "
                   f"{n_conflict_keys} (series_id, date) keys have conflicting values (NOT dropped, needs a decision)")
    if n_conflict_keys:
        report.append(f"    example conflicting rows:\n{key_dupes.head(4).to_string()}")
    return df

def flag_outliers(df: pd.DataFrame, report: list[str], top_n: int = 10) -> pd.DataFrame:
    """Flag |z-score| > 3 within each series (not pooled); report per-series stats and the most extreme rows."""
    g = df.groupby("series_id")["value"]
    z = (df["value"] - g.transform("mean")) / g.transform("std").replace(0, pd.NA)
    df["is_outlier"] = z.abs().gt(3).fillna(False)
    stats = df.groupby("series_id")["value"].agg(min="min", p1=lambda s: s.quantile(.01),
                                                   p99=lambda s: s.quantile(.99), max="max")
    report.append(f"  outliers: {int(df['is_outlier'].sum())} rows flagged across "
                   f"{stats.shape[0]} series (|z|>3 computed per series, not pooled)")
    report.append(f"    per-series min/p1/p99/max:\n{stats.to_string()}")
    order = z.abs().sort_values(ascending=False).head(top_n).index
    top = df.loc[order, ["series_id", "date", "value"]].copy()
    top["z"] = z.loc[order].round(1)
    report.append(f"    {top_n} most extreme values:\n{top.to_string(index=False)}")
    return df

def flag_negative(df: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """Flag negative values as impossible (levels, headcounts, and wages can't be negative)."""
    df["value_suspect"] = df["value"] < 0
    report.append(f"  range check (value < 0, impossible for a level/headcount/wage): "
                   f"{int(df['value_suspect'].sum())} rows flagged")
    return df

def process(name: str, pattern: str, report: list[str]) -> pd.DataFrame:
    """Load, flatten, date-convert, numeric-convert, clean, and report one survey."""
    path = latest_file(pattern)
    df = flatten(path)
    report.append(f"\n=== {name} ({path.name}) ===")
    report.append(f"units: {UNITS[name]}")
    report.append(f"series: {df['series_id'].nunique()}, raw rows: {len(df)}")

    df["value"] = pd.to_numeric(df["value_raw"], errors="coerce")
    failed = df[df["value"].isna() & df["value_raw"].notna()]
    report.append(f"value conversion failures: {len(failed)}"
                   + (f" -- {failed[['series_id', 'year', 'period', 'value_raw']].to_dict('records')}" if len(failed) else ""))

    df["date"] = [period_to_date(y, p) for y, p in zip(df["year"], df["period"])]
    n_annual_avg = df["date"].isna().sum()
    report.append(f"M13 (annual average) rows flagged and excluded: {n_annual_avg}")
    df = df.dropna(subset=["date"]).drop(columns=["year", "period", "value_raw"])
    df = df[["series_id", "series_name", "date", "value", "footnotes"]].sort_values(["series_id", "date"]).reset_index(drop=True)
    report.append(f"final rows: {len(df)}, date range: {df['date'].min().date()} to {df['date'].max().date()}")

    missing_report(df, report)
    date_gap_report(df, report)
    df = dedupe(df, report)
    df = flag_outliers(df, report)
    df = flag_negative(df, report)
    df["year"], df["quarter"] = df["date"].dt.year, df["date"].dt.quarter
    return df

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    jolts = process("bls_jolts", "jolts_*.json", report)
    report.append("NOTE: JOLTS API returns 'Unable to get Catalog Data' for every series -- "
                   "that's a BLS-side metadata gap, not a data problem. Names above come from "
                   "_series_catalog.json, not the API response.")

    ces = process("bls_ces", "ces_*.json", report)

    oews = process("bls_oews", "oews_*.json", report)
    report.append("NOTE: OEWS carries only the 2025 reference year by design -- the API exposes "
                   "one year at a time for this survey. A single row per series is expected, not an error.")

    report.append("\nNORMALISATION applied to every survey above: columns already snake_case "
                   "(series_id, series_name, date, value, footnotes), string columns stripped, "
                   "'date' as the sole date column name/dtype (datetime64[ns] before CSV write), "
                   "year/quarter added for later aggregation.")

    report_text = "\n".join(report)
    print(report_text)
    REPORT_PATH.write_text(report_text + "\n")

    jolts.to_csv(PROCESSED_DIR / "bls_jolts.csv", index=False)
    ces.to_csv(PROCESSED_DIR / "bls_ces.csv", index=False)
    oews.to_csv(PROCESSED_DIR / "bls_oews.csv", index=False)

if __name__ == "__main__":
    main()
