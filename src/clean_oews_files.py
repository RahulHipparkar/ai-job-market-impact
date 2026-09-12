"""Build a multi-year OEWS trend series from the annual national XLSX downloads.

Input: data/raw/bls/oews/national_M*_dl.xlsx (2019-2025). These are
separate from the single-year API pull in clean_bls.py and are the
only source of OEWS history.
"""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bls" / "oews"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "oews_trend_report.txt"

# The task's label for 15-1245 doesn't match BLS's own title for it (see
# report) -- BLS's occ_title is kept as the source of truth either way.
TARGET_CODES = {
    "15-1252": "Software Developers",
    "15-2051": "Data Scientists",
    "15-1211": "Computer Systems Analysts",
    "15-1221": "Computer and Information Research Scientists",
    "15-1299": "Computer Occupations, All Other",
    "15-1245": "Database and Network Administrators",
}
KEEP = ["occ_code", "occ_title", "tot_emp", "a_median", "a_mean", "a_pct10", "a_pct90", "year"]
NUMERIC = ["tot_emp", "a_median", "a_mean", "a_pct10", "a_pct90"]
TOPCODE_WAGE = 239_200
# YoY moves in this data run single digits to low teens; 20% comfortably
# separates normal growth from a definitional break or a first-year series.
YOY_FLAG_PCT = 20

def load_year(path: Path) -> pd.DataFrame:
    """Read one annual OEWS workbook, snake_case its columns, and tag it with its year."""
    year = int(re.search(r"M(\d{4})", path.name).group(1))
    df = pd.read_excel(path, sheet_name=f"national_M{year}_dl")
    df.columns = [c.strip().lower() for c in df.columns]
    df["year"] = year
    return df

def clean_numeric(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strip commas, convert '*'/'#' suppression markers to null, coerce to numeric."""
    raw = s.astype(str).str.strip()
    is_star, is_hash = raw.eq("*"), raw.eq("#")
    numeric = pd.to_numeric(raw.str.replace(",", "", regex=False).mask(is_star | is_hash), errors="coerce")
    return numeric, is_star, is_hash

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    # 1. load and stack
    files = sorted(RAW_DIR.glob("national_M*_dl.xlsx"))
    frames = []
    report.append(f"=== load ({len(files)} files) ===")
    for f in files:
        df = load_year(f)
        report.append(f"  {f.name}: {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    report.append("row counts per year: " + ", ".join(f"{y}={n}" for y, n in all_df.groupby("year").size().items()))

    # 2. filter to target occupations, report gaps (2018 SOC revision)
    report.append("\n=== target occupation coverage (2018 SOC revision breaks some codes) ===")
    for yr, grp in all_df.groupby("year"):
        found = sorted(set(grp["occ_code"]) & set(TARGET_CODES))
        missing = sorted(set(TARGET_CODES) - set(found))
        report.append(f"  {yr}: found {found}" + (f", MISSING {missing}" if missing else ""))
    df = all_df[all_df["occ_code"].isin(TARGET_CODES)].copy()
    for code, expected_title in TARGET_CODES.items():
        actual = sorted(df.loc[df["occ_code"] == code, "occ_title"].unique())
        if actual and actual != [expected_title]:
            report.append(f"  NOTE: {code} BLS title {actual} != brief's label {expected_title!r} -- BLS title kept")

    # 3. keep fields
    missing_fields = [c for c in KEEP if c not in df.columns]
    report.append(f"\n=== fields ===\nmissing expected fields: {missing_fields or 'none'}")
    df = df[[c for c in KEEP if c in df.columns]].reset_index(drop=True)

    # 4. cleaning
    report.append("\n=== suppression markers ('*' = withheld, '#' = wage above $239,200 topcode) ===")
    hash_any = pd.Series(False, index=df.index)
    for col in NUMERIC:
        numeric, is_star, is_hash = clean_numeric(df[col])
        failed = df[col].notna() & numeric.isna() & ~is_star & ~is_hash
        if is_star.sum() or is_hash.sum() or failed.sum():
            for yr, grp_idx in df.groupby("year").groups.items():
                s, h, fl = is_star[grp_idx].sum(), is_hash[grp_idx].sum(), failed[grp_idx].sum()
                if s or h or fl:
                    report.append(f"  {col} {yr}: {s} suppressed ('*'), {h} topcoded ('#'), {fl} other conversion failures")
        df[col] = numeric
        if col != "tot_emp":
            hash_any |= is_hash
    df["wage_topcoded"] = hash_any
    report.append(f"wage_topcoded rows (wage > ${TOPCODE_WAGE:,}, value nulled, flagged not lost): {int(hash_any.sum())}")
    if hash_any.any():
        report.append(f"  {df.loc[hash_any, ['occ_code', 'occ_title', 'year']].to_string(index=False)}")

    report.append("\n=== missing values ===")
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n:
            report.append(f"  {c}: {n} ({n / len(df):.1%})")

    report.append("\n=== duplicates on (occ_code, year) ===")
    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    key_dupes = df[df.duplicated(subset=["occ_code", "year"], keep=False)]
    report.append(f"{n_exact} exact duplicate rows dropped; "
                   f"{key_dupes[['occ_code', 'year']].drop_duplicates().shape[0]} keys conflict (NOT dropped)")

    report.append(f"\n=== year-over-year outliers (|change| > {YOY_FLAG_PCT}%, per occupation) ===")
    df = df.sort_values(["occ_code", "year"]).reset_index(drop=True)
    df["emp_yoy_pct"] = df.groupby("occ_code")["tot_emp"].pct_change() * 100
    df["wage_yoy_pct"] = df.groupby("occ_code")["a_median"].pct_change() * 100
    df["is_outlier_yoy"] = (df["emp_yoy_pct"].abs() > YOY_FLAG_PCT) | (df["wage_yoy_pct"].abs() > YOY_FLAG_PCT)
    flagged = df[df["is_outlier_yoy"]]
    report.append(f"{len(flagged)} rows flagged (first year of any series has no prior year, so is never flagged):")
    if len(flagged):
        report.append(flagged[["occ_code", "occ_title", "year", "emp_yoy_pct", "wage_yoy_pct"]].to_string(index=False))

    report_text = "\n".join(report)
    print(report_text)
    REPORT_PATH.write_text(report_text + "\n")

    df.to_csv(PROCESSED_DIR / "oews_trend.csv", index=False)

if __name__ == "__main__":
    main()
