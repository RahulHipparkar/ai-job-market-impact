"""Second-stage cleaning for parsed Hacker News postings.

Input: data/processed/hn_postings.csv.
Output: data/processed/hn_clean.csv, plus a report and before/after
samples in data/samples/.
"""

import difflib
import re
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
IN_CSV = PROCESSED_DIR / "hn_postings.csv"
OUT_CSV = PROCESSED_DIR / "hn_clean.csv"
REPORT_PATH = PROCESSED_DIR / "hn_clean_report.txt"

WORK_MODE_BODY_RE = re.compile(r"\b(remote|on-site|on site|onsite|hybrid|in office|in-office)\b", re.IGNORECASE)
STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
}
CITY_NAMES = [
    "San Francisco", "New York", "Los Angeles", "Seattle", "Austin", "Boston", "Denver",
    "Chicago", "London", "Berlin", "Toronto",
]
CITY_STATE_RE = re.compile(r"\b([A-Z][A-Za-z.\s]{1,30}),\s*([A-Z]{2})\b")
PAREN_RE = re.compile(r"\([^)]*\)")
PUNCT_RE = re.compile(r"[^\w\s]")
LEGAL_SUFFIX_RE = re.compile(r"\b(inc|llc|ltd|corp|corporation|gmbh|sa|bv|co)\b", re.IGNORECASE)
LOCATION_MAP = [
    (re.compile(r"\b(sf|san francisco|sf bay area|bay area)\b", re.IGNORECASE), "San Francisco, CA", "USA"),
    (re.compile(r"\b(nyc|new york|ny,?\s*ny)\b", re.IGNORECASE), "New York, NY", "USA"),
    (re.compile(r"\bseattle\b", re.IGNORECASE), "Seattle, WA", "USA"),
    (re.compile(r"\baustin\b", re.IGNORECASE), "Austin, TX", "USA"),
    (re.compile(r"\bboston\b", re.IGNORECASE), "Boston, MA", "USA"),
    (re.compile(r"\b(la|los angeles)\b", re.IGNORECASE), "Los Angeles, CA", "USA"),
    (re.compile(r"\bdenver\b", re.IGNORECASE), "Denver, CO", "USA"),
    (re.compile(r"\bchicago\b", re.IGNORECASE), "Chicago, IL", "USA"),
    (re.compile(r"\blondon\b", re.IGNORECASE), "London, UK", "United Kingdom"),
    (re.compile(r"\bberlin\b", re.IGNORECASE), "Berlin, Germany", "Germany"),
    (re.compile(r"\btoronto\b", re.IGNORECASE), "Toronto, Canada", "Canada"),
    (re.compile(r"\bremote\b", re.IGNORECASE), "Remote", None),
]
URL_ONLY_RE = re.compile(r"^\s*https?://\S+\s*$")
MODERATOR_RE = re.compile(r"^\s*\[(flagged|dead)\]\s*$", re.IGNORECASE)
URL_STRIP_RE = re.compile(r"https?://\S+")
EMAIL_STRIP_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
REPOST_SNIPPET_LEN = 400
REPOST_SIM_THRESHOLD = 0.90
REPOST_MIN_LEN = 20

def find_work_mode_in_body(text) -> str | None:
    """Search full body text for a work-mode keyword; return it verbatim."""
    m = WORK_MODE_BODY_RE.search(text if isinstance(text, str) else "")
    return m.group(0) if m else None

def find_location_in_body(text) -> str | None:
    """Search the first ~300 chars of the body for a city/state pattern."""
    snippet = (text if isinstance(text, str) else "")[:300]
    m = CITY_STATE_RE.search(snippet)
    if m and m.group(2).upper() in STATE_ABBR:
        return m.group(0).strip()
    for city in CITY_NAMES:
        if re.search(rf"\b{re.escape(city)}\b", snippet, re.IGNORECASE):
            return city
    return None

def normalize_company(name) -> str | None:
    """Lowercase, strip punctuation/legal suffixes/parentheticals, collapse whitespace."""
    if not isinstance(name, str) or not name.strip():
        return None
    s = PAREN_RE.sub(" ", name).lower()
    s = PUNCT_RE.sub(" ", s)
    s = LEGAL_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def normalize_location(loc) -> tuple:
    """Map known tech-hub variants to a canonical name and country."""
    if not isinstance(loc, str) or not loc.strip():
        return None, None
    for pattern, canon, country in LOCATION_MAP:
        if pattern.search(loc):
            return canon, country
    return loc.strip(), None

def make_text_model(text) -> str:
    if not isinstance(text, str):
        return ""
    t = URL_STRIP_RE.sub(" ", text.lower())
    t = EMAIL_STRIP_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def _uf_find(parent: dict, x):
    """Path-compressed find for the repost union-find structure."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def flag_reposts(df: pd.DataFrame) -> pd.DataFrame:
    """Flag postings as reposts of an earlier same-company posting when first-400-char snippets are >90% similar."""
    snippet = df["text_model"].fillna("").str.slice(0, REPOST_SNIPPET_LEN)
    is_repost = pd.Series(False, index=df.index)
    repost_of = pd.Series(pd.NA, index=df.index, dtype="Int64")
    repost_group = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for _, idx in df.groupby("company_clean").groups.items():
        members = [i for i in idx if len(snippet.loc[i]) >= REPOST_MIN_LEN]
        if len(members) < 2:
            continue
        parent = {i: i for i in members}
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ratio = difflib.SequenceMatcher(None, snippet.loc[members[a]], snippet.loc[members[b]]).ratio()
                if ratio > REPOST_SIM_THRESHOLD and _uf_find(parent, members[a]) != _uf_find(parent, members[b]):
                    parent[_uf_find(parent, members[a])] = _uf_find(parent, members[b])
        clusters: dict = {}
        for i in members:
            clusters.setdefault(_uf_find(parent, i), []).append(i)
        for cluster in clusters.values():
            if len(cluster) < 2:
                continue
            root = df.loc[cluster, ["month", "id"]].sort_values(["month", "id"]).index[0]
            root_id = df.loc[root, "id"]
            repost_group.loc[cluster] = root_id
            others = [c for c in cluster if c != root]
            is_repost.loc[others] = True
            repost_of.loc[others] = root_id
    df["is_repost"], df["repost_of"], df["repost_group"] = is_repost, repost_of, repost_group
    return df

def frequency_band(n_months) -> str | None:
    """Bucket a company's distinct-month posting count into one_off/occasional/regular/persistent."""
    if pd.isna(n_months):
        return None
    return ("one_off" if n_months <= 1 else "occasional" if n_months <= 5
            else "regular" if n_months <= 20 else "persistent")

def main() -> None:
    df = pd.read_csv(IN_CSV, keep_default_na=True)
    lines = []

    # recover missing work_mode / location from body text
    wm_before, loc_before = df["work_mode"].notna().mean(), df["location"].notna().mean()
    df["work_mode_source"] = df["work_mode"].notna().map({True: "header", False: None})
    df["location_source"] = df["location"].notna().map({True: "header", False: None})
    missing_wm = df["work_mode"].isna()
    recovered_wm = df.loc[missing_wm, "text_clean"].map(find_work_mode_in_body)
    df.loc[missing_wm, "work_mode"] = recovered_wm
    df.loc[missing_wm & recovered_wm.notna(), "work_mode_source"] = "body"
    missing_loc = df["location"].isna()
    recovered_loc = df.loc[missing_loc, "text_clean"].map(find_location_in_body)
    df.loc[missing_loc, "location"] = recovered_loc
    df.loc[missing_loc & recovered_loc.notna(), "location_source"] = "body"
    lines += [
        f"work_mode fill rate: {wm_before:.1%} -> {df['work_mode'].notna().mean():.1%}",
        f"location fill rate: {loc_before:.1%} -> {df['location'].notna().mean():.1%}",
    ]

    # normalise company names
    n_company_before = df["company"].nunique()
    df["company_clean"] = df["company"].map(normalize_company)
    lines.append(f"distinct companies: {n_company_before} -> {df['company_clean'].nunique()}")

    # normalise locations
    n_loc_before = df["location"].nunique()
    norm = df["location"].map(normalize_location)
    df["location_clean"] = norm.map(lambda t: t[0])
    df["country"] = norm.map(lambda t: t[1])
    lines.append(f"distinct locations: {n_loc_before} -> {df['location_clean'].nunique()}")
    lines.append("top 25 locations:")
    for name, count in df["location_clean"].value_counts().head(25).items():
        lines.append(f"  {name}: {count}")

    # junk and outliers
    is_url_only = df["text_clean"].fillna("").str.match(URL_ONLY_RE)
    is_mod = df["text_clean"].fillna("").str.match(MODERATOR_RE)
    is_empty = df["text_clean"].fillna("").str.len() == 0
    df["is_junk"] = ((df["text_len"] < 100) & ~df["parse_ok"]) | is_mod | is_empty | is_url_only
    p995 = df["text_len"].quantile(0.995)
    df["is_outlier_len"] = df["text_len"] > p995
    lines.append(f"is_junk rows: {df['is_junk'].sum()} ({df['is_junk'].mean():.1%})")
    lines.append("10 is_junk examples:")
    for t in df.loc[df["is_junk"], "text_clean"].fillna("").head(10):
        lines.append(f"  {t[:80]!r}")
    desc = df["text_len"].describe(percentiles=[0.5, 0.95, 0.99, 0.995])
    lines.append(f"text_len: min={desc['min']:.0f} median={desc['50%']:.0f} "
                  f"p95={desc['95%']:.0f} p99.5={desc['99.5%']:.0f} max={desc['max']:.0f}")
    lines.append(f"is_outlier_len rows (>p99.5): {df['is_outlier_len'].sum()}")

    # text for modelling
    df["text_model"] = df["text_clean"].map(make_text_model)

    # near-duplicate reposts (leakage risk: companies repost the same text across months)
    df = flag_reposts(df)
    lines.append(f"\nis_repost rows: {df['is_repost'].sum()} ({df['is_repost'].mean():.1%})")
    grp_sizes = df.loc[df["repost_group"].notna()].groupby("repost_group").size()
    lines.append(f"repost groups (2+ postings, excludes unmatched originals): {len(grp_sizes)}")
    lines.append("repost group size distribution:")
    for size, count in grp_sizes.value_counts().sort_index().items():
        lines.append(f"  {size}: {count}")
    lines.append("10 largest repost groups:")
    for root_id, size in grp_sizes.sort_values(ascending=False).head(10).items():
        rows = df.loc[df["repost_group"] == root_id]
        lines.append(f"  {rows['company_clean'].iloc[0]} (repost_group={root_id}): "
                      f"{size} postings across {rows['month'].nunique()} months")

    # discretisation
    non_junk = ~df["is_junk"]
    df["text_len_band"] = pd.NA
    band, len_bins = pd.qcut(df.loc[non_junk, "text_len"], 3, labels=["short", "medium", "long"], retbins=True)
    df.loc[non_junk, "text_len_band"] = band
    lines.append(f"\ntext_len_band boundaries (tercile, non-junk rows only): {[round(b) for b in len_bins]}")
    lines.append("  " + df["text_len_band"].value_counts(dropna=True).to_string().replace("\n", "\n  "))
    company_months = df.groupby("company_clean")["month"].transform("nunique")
    df["posting_frequency_band"] = company_months.map(frequency_band)
    lines.append("posting_frequency_band boundaries: one_off=1, occasional=2-5, regular=6-20, "
                  "persistent=21+ (distinct months posted per company_clean)")
    lines.append("  " + df["posting_frequency_band"].value_counts(dropna=True).to_string().replace("\n", "\n  "))

    # numeric scaling
    len_mean = df.loc[non_junk, "text_len"].mean()
    len_std = df.loc[non_junk, "text_len"].std(ddof=0)
    df["text_len_scaled"] = (df["text_len"] - len_mean) / len_std
    lines.append(f"\ntext_len_scaled: z-score fitted on non-junk rows only, mean={len_mean:.2f}, std={len_std:.2f}")

    # output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    parsed = pd.read_csv(IN_CSV).sort_values(["month", "id"]).head(20)
    parsed.to_csv(SAMPLES_DIR / "hn_parsed_sample.csv", index=False)
    df[df["id"].isin(parsed["id"])].to_csv(SAMPLES_DIR / "hn_cleaned_sample.csv", index=False)

    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n")


if __name__ == "__main__":
    main()
