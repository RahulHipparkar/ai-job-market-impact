"""Second-stage cleaning for ats_clean.csv: missing-value recovery,
location normalisation, near-duplicate flags, outlier flags, and
discretised bands.

Input: data/processed/ats_clean.csv.
Output: data/processed/ats_final.csv, plus a report and a before/after
sample in data/samples/.
"""

import re
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
IN_CSV = PROCESSED_DIR / "ats_clean.csv"
OUT_CSV = PROCESSED_DIR / "ats_final.csv"
REPORT_PATH = PROCESSED_DIR / "ats_final_report.txt"

PAREN_RE = re.compile(r"\([^)]*\)")
SPLIT_RE = re.compile(r"\s*(?:[;|•]|\bor\b)\s*", re.IGNORECASE)
PREFIX_RE = re.compile(r"^(hybrid|remote)\s+-\s+", re.IGNORECASE)
REGION_DASH_RE = re.compile(r"^[a-z\s]+-\s*(united states|usa|us|canada|india)$")
NON_CITY = {
    "united states", "remote", "north america", "south america", "europe", "asia",
    "emea", "apac", "uk", "usa", "us", "canada", "india", "worldwide", "global",
}
# Same mapping approach as clean_hn_stage2.py: first-match regex over the
# common hubs, raw stripped string as fallback for everything else.
LOCATION_MAP = [
    (re.compile(r"san francisco|sf bay area|\bbay area\b|mountain view|palo alto", re.I), "San Francisco, CA"),
    (re.compile(r"new york", re.I), "New York, NY"),
    (re.compile(r"\blondon\b", re.I), "London, UK"),
    (re.compile(r"\bsingapore\b", re.I), "Singapore"),
    (re.compile(r"bengaluru|bangalore", re.I), "Bengaluru, India"),
    (re.compile(r"washington", re.I), "Washington, DC"),
    (re.compile(r"\btokyo\b", re.I), "Tokyo, Japan"),
    (re.compile(r"seattle|bellevue", re.I), "Seattle, WA"),
    (re.compile(r"\btoronto\b|\bvancouver\b", re.I), "Toronto, Canada"),
    (re.compile(r"\bparis\b", re.I), "Paris, France"),
    (re.compile(r"\bmunich\b", re.I), "Munich, Germany"),
    (re.compile(r"\bberlin\b", re.I), "Berlin, Germany"),
    (re.compile(r"\bseoul\b", re.I), "Seoul, South Korea"),
    (re.compile(r"amsterdam", re.I), "Amsterdam, Netherlands"),
    (re.compile(r"\bchicago\b", re.I), "Chicago, IL"),
    (re.compile(r"\bdublin\b", re.I), "Dublin, Ireland"),
    (re.compile(r"\bsydney\b", re.I), "Sydney, Australia"),
    (re.compile(r"\bwarsaw\b", re.I), "Warsaw, Poland"),
    (re.compile(r"\baustin\b", re.I), "Austin, TX"),
    (re.compile(r"\bboston\b", re.I), "Boston, MA"),
    (re.compile(r"los angeles", re.I), "Los Angeles, CA"),
    (re.compile(r"remote", re.I), "Remote"),
]

def location_primary(loc: str) -> str | None:
    """Return the first location in a ;/|/bullet/or-delimited location string."""
    if not isinstance(loc, str) or not loc.strip():
        return None
    return SPLIT_RE.split(loc.strip())[0].strip()

def is_non_city(candidate: str) -> bool:
    """True if a location_primary segment reads as a region/remote descriptor, not a city."""
    c = candidate.lower()
    if c in NON_CITY or len(c) < 2:
        return True
    if "remote" in c or "hybrid" in c or "on-site" in c or "onsite" in c:
        return True
    return bool(REGION_DASH_RE.match(c))

def recover_city_region(primary: str) -> tuple[str | None, str | None]:
    """Split a location_primary into (city, region), filtering non-city descriptors."""
    if not isinstance(primary, str):
        return None, None
    s = PREFIX_RE.sub("", PAREN_RE.sub("", primary).strip()).strip()
    if not s:
        return None, None
    parts = [p.strip() for p in s.split(",")]
    if is_non_city(parts[0]):
        return None, None
    return parts[0], (parts[1] if len(parts) >= 2 else None)

def normalize_location(primary: str) -> str | None:
    """Map a location_primary to a canonical hub name via LOCATION_MAP."""
    if not isinstance(primary, str) or not primary.strip():
        return None
    for rx, canon in LOCATION_MAP:
        if rx.search(primary):
            return canon
    return primary.strip()

def normalize_title(title: str) -> str:
    """Lowercase and collapse whitespace for near-duplicate matching."""
    return re.sub(r"\s+", " ", title.strip().lower()) if isinstance(title, str) else ""

def main() -> None:
    df = pd.read_csv(IN_CSV)
    report = [f"input rows: {len(df)}", "\nmissing value audit (count, %):"]
    for c in df.columns:
        n = df[c].isna().sum()
        report.append(f"  {c}: {n} ({n / len(df):.1%})")

    # missing-value decisions
    df["has_salary"] = df["salary_annual_min"].notna()
    df["has_department"] = df["department"].notna()
    report.append(f"\nhas_salary: {df['has_salary'].sum()} True -- salary left null elsewhere, "
                   "Greenhouse exposes no compensation field at all")
    report.append(f"has_department: {df['has_department'].sum()} True -- left null elsewhere")
    report.append("employmentType fill rate by ats (left null; the gap is platform-driven, not random):")
    for name, rate in df.groupby("ats")["employmentType"].apply(lambda s: s.notna().mean()).items():
        report.append(f"  {name}: {rate:.1%}")

    df["location_primary"] = df["location"].map(location_primary)
    city_before, region_before = df["city"].notna().mean(), df["region"].notna().mean()
    missing_city, missing_region = df["city"].isna(), df["region"].isna()
    recovered = df["location_primary"].map(recover_city_region)
    df.loc[missing_city, "city"] = recovered[missing_city].map(lambda t: t[0])
    df.loc[missing_region, "region"] = recovered[missing_region].map(lambda t: t[1])
    report.append(f"\ncity fill rate: {city_before:.1%} -> {df['city'].notna().mean():.1%} (recovered from location)")
    report.append(f"region fill rate: {region_before:.1%} -> {df['region'].notna().mean():.1%} (recovered from location)")

    # location normalisation
    n_before = df["location"].nunique()
    df["location_count"] = df["location"].fillna("").map(lambda s: len(SPLIT_RE.split(s)) if s else 0)
    df["is_multi_location"] = df["location_count"] > 1
    df["location_clean"] = df["location_primary"].map(normalize_location)
    report.append(f"\ndistinct locations: {n_before} raw -> {df['location_clean'].nunique()} canonical")
    report.append(f"is_multi_location: {df['is_multi_location'].sum()} rows")
    report.append("top 20 location_clean:")
    for name, count in df["location_clean"].value_counts().head(20).items():
        report.append(f"  {name}: {count}")

    # near-duplicate flag (not dropped)
    title_norm = df["title"].map(normalize_title)
    desc_prefix = df["description_clean"].fillna("").str.slice(0, 300)
    group_size = df.groupby([df["company_clean"], title_norm, desc_prefix])["jobId"].transform("size")
    df["is_near_dup"] = group_size > 1
    report.append(f"\nis_near_dup: {df['is_near_dup'].sum()} rows "
                   "(company_clean + normalised title + first 300 desc chars repeat)")
    report.append("5 examples:")
    examples = df[df["is_near_dup"]].assign(title_norm=title_norm[df["is_near_dup"]])
    examples = examples.drop_duplicates(["company_clean", "title_norm"]).head(5)
    for _, r in examples.iterrows():
        report.append(f"  {r['company_clean']} -- {r['title']} (jobId={r['jobId']})")

    # description length outliers
    desc = df["desc_len"]
    report.append("\ndesc_len distribution:")
    report.extend(f"  {line}" for line in desc.describe().to_string().splitlines())
    p1, p99 = desc.quantile(0.01), desc.quantile(0.99)
    df["desc_too_short"] = desc < p1
    df["desc_too_long"] = desc > p99
    report.append(f"desc_too_short (< p1 = {p1:.0f} chars): {df['desc_too_short'].sum()}")
    for t in df.loc[df["desc_too_short"], "title"].head(3):
        report.append(f"  short: {t}")
    report.append(f"desc_too_long (> p99 = {p99:.0f} chars): {df['desc_too_long'].sum()}")
    for t in df.loc[df["desc_too_long"], "title"].head(3):
        report.append(f"  long: {t}")

    # discretisation
    df["desc_len_band"], desc_bins = pd.qcut(desc, 3, labels=["short", "medium", "long"], retbins=True)
    report.append(f"\ndesc_len_band boundaries: {[round(b) for b in desc_bins]}")
    report.append("  " + df["desc_len_band"].value_counts().to_string().replace("\n", "\n  "))
    has_sal = df["salary_annual_min"].notna()
    df["salary_band"] = pd.NA
    band, sal_bins = pd.qcut(df.loc[has_sal, "salary_annual_min"], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                              duplicates="drop", retbins=True)
    df.loc[has_sal, "salary_band"] = band
    report.append(f"salary_band boundaries: {[round(b) for b in sal_bins]}")
    report.append("  " + df["salary_band"].value_counts(dropna=True).to_string().replace("\n", "\n  "))

    # output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(report)
    print(report_text)
    REPORT_PATH.write_text(report_text + "\n")

    df.to_csv(OUT_CSV, index=False)
    sample_ids = pd.read_csv(SAMPLES_DIR / "ats_clean_sample.csv")["jobId"]
    df[df["jobId"].isin(sample_ids)].to_csv(SAMPLES_DIR / "ats_final_sample.csv", index=False)

if __name__ == "__main__":
    main()
