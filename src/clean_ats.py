"""Load, dedupe, and clean the four raw ATS job-posting exports.

Input: data/raw/ats/*.csv (Ashby/Greenhouse/Lever scrapes).
Output: ats_clean.csv in data/processed/, plus a report and raw/clean
samples in data/samples/.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ats"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
OUT_CSV = PROCESSED_DIR / "ats_clean.csv"
REPORT_PATH = PROCESSED_DIR / "ats_clean_report.txt"

QUOTE_MAP = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "}
)

# 19 companies span 3 ATS platforms with inconsistent casing ("openai",
# "Scale AI", "harvey"). A hand mapping is more reliable than title-casing
# rules, which would mangle "OpenAI" / "Scale AI".
COMPANY_MAP = {
    "airtable": "Airtable", "amplitude": "Amplitude", "anthropic": "Anthropic",
    "asana": "Asana", "databricks": "Databricks", "discord": "Discord",
    "figma": "Figma", "scale ai": "Scale AI", "stripe": "Stripe",
    "vercel": "Vercel", "webflow": "Webflow", "cohere": "Cohere",
    "harvey": "Harvey", "linear": "Linear", "openai": "OpenAI",
    "palantir": "Palantir", "perplexity": "Perplexity", "ramp": "Ramp",
    "sierra": "Sierra",
}

# Checked in order; later rules override earlier ones (last match wins), so
# a title matching two rules resolves to the more senior one -- e.g.
# "Senior Staff Engineer" matches senior then staff, and staff wins.
# senior/II/III use inline case flags so bare "I" doesn't match inside
# words like "AI" (word-boundary regex already excludes that).
SENIORITY_RULES = [
    ("intern", re.compile(r"(?i:intern(ship)?)")),
    ("new_grad", re.compile(r"(?i:new\s?grad(uate)?|university\s?grad|early career)")),
    ("junior", re.compile(r"(?i:junior|jr\.?|associate)|\bI\b")),
    ("senior", re.compile(r"(?i:senior|sr\.?)|\bII\b|\bIII\b")),
    ("staff", re.compile(r"(?i:staff|principal|distinguished|architect)")),
    ("lead", re.compile(r"(?i:lead|manager|head of|director|vp|chief)")),
]

# Checked in order; first match wins, so forward_deployed must precede
# solutions/software_eng or "Forward Deployed Software Engineer" would
# fall into software_eng instead.
ROLE_RULES = [
    ("forward_deployed", re.compile(r"forward.?deploy|\bFDE\b", re.I)),
    ("solutions", re.compile(r"solutions? (eng|arch|consult)|delivery|deployment strateg|technical account", re.I)),
    ("applied_ai", re.compile(r"applied (ai|ml|scien)|ai engineer|ml engineer|machine learning eng", re.I)),
    ("research", re.compile(r"research (scien|eng)", re.I)),
    ("software_eng", re.compile(r"software engineer|backend|frontend|infrastructure|platform eng|\bsre\b", re.I)),
    ("data", re.compile(r"data (scien|eng|analy)", re.I)),
    ("product", re.compile(r"product manager|product design", re.I)),
    ("sales_gtm", re.compile(r"account exec|sales|\bgtm\b|partnership|business development", re.I)),
    ("recruiting", re.compile(r"recruit|talent|people ops", re.I)),
]

# salaryInterval is blank on most salaried rows. Below this, a bare number
# reads as an hourly rate ($32-$150/hr); at or above, it reads as an
# annual salary (tech comp starts ~$100k). 1,000 sits well below the
# lowest observed annual figure and above the highest observed hourly one.
HOURLY_THRESHOLD = 1000
HOURS_PER_YEAR = 2080
SUSPECT_LOW, SUSPECT_HIGH = 20_000, 1_000_000

# Explicit output allowlist. ATS scraping plumbing (companySlug, tags/N,
# customFields/N, locations/1-15, requisitionId, scrapedAt, ...) is dropped
# for cleanliness -- see the report for the full drop list.
FINAL_COLUMNS = [
    "jobId", "ats", "company_clean", "companyName", "title", "seniority", "role_family",
    "description_clean", "desc_len",
    "salaryMin", "salaryMax", "salaryInterval", "salaryCurrency",
    "salary_interval_inferred", "salary_annual_min", "salary_annual_max",
    "salary_currency", "salary_is_usd", "salary_suspect",
    "isRemote", "workplaceType", "is_remote",
    "city", "region", "country", "location",
    "employmentType", "hasEquity", "department",
    "posted_date", "month", "year",
]

def clean_description(text: str) -> str:
    """Normalise nbsp/quotes/dashes and collapse whitespace in a description."""
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFC", text).translate(QUOTE_MAP)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

def classify_seniority(title: str) -> str:
    """Derive a seniority tier from a job title using SENIORITY_RULES."""
    if not isinstance(title, str):
        return "mid"
    result = "mid"
    for label, rx in SENIORITY_RULES:
        if rx.search(title):
            result = label
    return result

def classify_role_family(title: str) -> str:
    """Bucket a job title into a role family using ROLE_RULES."""
    if isinstance(title, str):
        for label, rx in ROLE_RULES:
            if rx.search(title):
                return label
    return "other"

def infer_interval(salary_min: float, interval: str) -> str | float:
    """Fill a blank salaryInterval from magnitude; pass explicit values through."""
    if pd.notna(interval):
        return interval
    if pd.isna(salary_min):
        return float("nan")
    return "hour" if salary_min < HOURLY_THRESHOLD else "year"

def annualize(value: float, interval: str) -> float:
    """Convert a salary figure to an annual basis given its interval."""
    if pd.isna(value) or pd.isna(interval):
        return float("nan")
    return value * HOURS_PER_YEAR if interval == "hour" else value

def load_all() -> tuple[pd.DataFrame, list[str]]:
    """Load the four raw CSVs, tagging each row with its source file."""
    lines = []
    frames = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(f, dtype={"jobId": str})
        df["__src"] = f.name
        frames.append(df)
        lines.append(f"  {f.name}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True), lines

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    df, load_lines = load_all()
    n_loaded = len(df)
    report.append(f"loaded: {n_loaded} rows from 4 files")
    report.extend(load_lines)

    # Duplicate jobIds are all Batch B2 re-scrapes of rows already in the
    # Companies B dataset; keep the freshest scrape per job.
    df = df.sort_values("scrapedAt").drop_duplicates(subset="jobId", keep="last")
    report.append(f"after dedup on jobId: {len(df)} unique (dropped {n_loaded - len(df)} duplicates)")
    report.append("rows retained per source file after dedup:")
    for name, count in df["__src"].value_counts().items():
        report.append(f"  {name}: {count}")

    raw_sample = df.sort_values("jobId").head(20).copy()

    df["description_clean"] = df["descriptionText"].apply(clean_description)
    df["desc_len"] = df["description_clean"].str.len()
    report.append("\ndesc_len distribution:")
    report.extend(f"  {line}" for line in df["desc_len"].describe().to_string().splitlines())

    df["company_clean"] = df["companyName"].str.strip().str.lower().map(COMPANY_MAP)
    unmapped = df["company_clean"].isna().sum()
    report.append(f"\ncompany_clean: {df['companyName'].str.strip().str.lower().nunique()} raw names -> "
                   f"{df['company_clean'].nunique()} canonical ({unmapped} unmapped)")

    df["seniority"] = df["title"].apply(classify_seniority)
    report.append("\nseniority distribution:")
    for label, count in df["seniority"].value_counts().items():
        report.append(f"  {label}: {count}")

    df["role_family"] = df["title"].apply(classify_role_family)
    report.append("\nrole_family distribution:")
    for label, count in df["role_family"].value_counts().items():
        report.append(f"  {label}: {count}")
    report.append("sample titles per family:")
    for label, _ in ROLE_RULES + [("other", None)]:
        examples = df.loc[df["role_family"] == label, "title"].head(5).tolist()
        report.append(f"  {label}:")
        for t in examples:
            report.append(f"    - {t}")

    has_salary = df["salaryMin"].notna()
    df["salary_interval_inferred"] = [infer_interval(m, i) for m, i in zip(df["salaryMin"], df["salaryInterval"])]
    df["salary_annual_min"] = [annualize(v, i) for v, i in zip(df["salaryMin"], df["salary_interval_inferred"])]
    df["salary_annual_max"] = [annualize(v, i) for v, i in zip(df["salaryMax"], df["salary_interval_inferred"])]
    df["salary_currency"] = df["salaryCurrency"]
    df["salary_is_usd"] = (df["salary_currency"] == "USD").where(has_salary, float("nan"))
    suspect = ((df["salary_annual_min"] < SUSPECT_LOW) | (df["salary_annual_min"] > SUSPECT_HIGH) |
               (df["salary_annual_max"] < SUSPECT_LOW) | (df["salary_annual_max"] > SUSPECT_HIGH))
    df["salary_suspect"] = suspect.where(has_salary, float("nan"))

    report.append(f"\nsalary: {has_salary.sum()} / {len(df)} rows have salaryMin")
    report.append(f"  salaryInterval given blank/year/hour: "
                   f"{df.loc[has_salary, 'salaryInterval'].isna().sum()} / "
                   f"{(df.loc[has_salary, 'salaryInterval'] == 'year').sum()} / "
                   f"{(df.loc[has_salary, 'salaryInterval'] == 'hour').sum()}")
    report.append(f"  interval inferred as hourly: {(df.loc[has_salary, 'salary_interval_inferred'] == 'hour').sum()}, "
                   f"annual: {(df.loc[has_salary, 'salary_interval_inferred'] == 'year').sum()}")
    report.append("  currency: " + ", ".join(f"{k}={v}" for k, v in df.loc[has_salary, "salary_currency"].value_counts().items()))
    report.append(f"  salary_is_usd True/False: {(df['salary_is_usd'] == True).sum()} / {(df['salary_is_usd'] == False).sum()}")
    report.append(f"  salary_suspect True/False: {(df['salary_suspect'] == True).sum()} / {(df['salary_suspect'] == False).sum()}")
    sub = df.loc[has_salary, ["title", "salaryMin", "salaryMax", "salary_interval_inferred", "salary_annual_min"]]
    report.append("  10 lowest salary_annual_min:")
    for _, r in sub.nsmallest(10, "salary_annual_min").iterrows():
        report.append(f"    {r['salary_annual_min']:.0f}  ({r['salary_interval_inferred']}, raw min={r['salaryMin']})  {r['title']}")
    report.append("  10 highest salary_annual_min:")
    for _, r in sub.nlargest(10, "salary_annual_min").iterrows():
        report.append(f"    {r['salary_annual_min']:.0f}  ({r['salary_interval_inferred']}, raw min={r['salaryMin']})  {r['title']}")

    dt = pd.to_datetime(df["postedAt"])
    df["posted_date"] = dt.dt.date.astype(str)
    df["month"] = dt.dt.strftime("%Y-%m")
    df["year"] = dt.dt.year
    report.append(f"\nposted_date range: {df['posted_date'].min()} to {df['posted_date'].max()}")

    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    report.append(f"\ndropped (0% populated): {empty_cols}")
    kept_extra = {"city", "region"}
    still_present = kept_extra & set(df.columns) - set(empty_cols)
    report.append(f"  NOTE: city/region are NOT 0% populated ({df['city'].notna().mean():.1%} / "
                   f"{df['region'].notna().mean():.1%}) -- kept despite the brief, since {still_present} have data")
    plumbing_dropped = sorted(set(df.columns) - set(FINAL_COLUMNS) - set(empty_cols) - {"descriptionText", "descriptionHtml", "postedAt", "salaryCurrency"})
    report.append(f"dropped (ATS scraping plumbing, redundant with kept fields): {plumbing_dropped}")

    is_remote = df["isRemote"]
    workplace_remote = df["workplaceType"] == "remote"
    df["is_remote"] = is_remote.where(is_remote.notna(), workplace_remote.where(df["workplaceType"].notna()))
    conflicts = ((df["isRemote"] == True) & (df["workplaceType"].notna()) & (df["workplaceType"] != "remote")).sum()
    report.append(f"\nis_remote: {df['is_remote'].notna().sum()} resolved, {df['is_remote'].isna().sum()} unknown "
                   f"({conflicts} rows where isRemote=True conflicted with a non-remote workplaceType -- isRemote wins)")

    out = df[FINAL_COLUMNS].reset_index(drop=True)
    report.insert(0, f"=== ats_clean report ===\nfinal rows written: {len(out)}, columns: {len(out.columns)}\n")
    report_text = "\n".join(report)
    print(report_text)
    REPORT_PATH.write_text(report_text + "\n")

    raw_sample.to_csv(SAMPLES_DIR / "ats_raw_sample.csv", index=False)
    df[df["jobId"].isin(raw_sample["jobId"])][FINAL_COLUMNS].to_csv(SAMPLES_DIR / "ats_clean_sample.csv", index=False)
    out.to_csv(OUT_CSV, index=False)

if __name__ == "__main__":
    main()
