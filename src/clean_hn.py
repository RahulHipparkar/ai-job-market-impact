"""Parse and clean the raw Hacker News "Who is hiring?" postings.

Input: data/raw/hn/*.json. Output: hn_postings.csv in data/processed/,
plus a report and raw/clean samples in data/samples/.
"""

import html
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "hn"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
OUT_CSV = PROCESSED_DIR / "hn_postings.csv"
REPORT_PATH = PROCESSED_DIR / "hn_parse_report.txt"

TAG_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_RE = re.compile(r"https?://")
SALARY_RE = re.compile(r"\$\s?\d|£\s?\d|€\s?\d|\b\d{2,3}\s?[kK]\b|\bUSD\b")
VISA_RE = re.compile(r"\bvisa\b|\bsponsor", re.IGNORECASE)
WORK_MODE_RE = re.compile(r"\b(remote|onsite|on-site|on site|hybrid)\b", re.IGNORECASE)
STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
}
CITY_HINTS = {
    "sf", "nyc", "la", "bay area", "san francisco", "new york", "los angeles", "chicago",
    "boston", "denver", "atlanta", "miami", "dallas", "houston", "seattle", "austin", "portland",
    "toronto", "vancouver", "montreal", "berlin", "munich", "london", "dublin", "amsterdam",
    "paris", "madrid", "barcelona", "zurich", "tel aviv", "bangalore", "sydney", "melbourne",
}
COUNTRY_HINTS = {
    "usa", "uk", "canada", "germany", "india", "poland", "spain", "france", "netherlands",
    "australia", "brazil", "mexico", "ireland", "portugal", "singapore", "japan", "sweden",
}
QUOTE_MAP = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "}
)

def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"<p>", "\n", raw)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text).translate(QUOTE_MAP)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

def looks_like_location(seg: str) -> bool:
    s = seg.lower()
    m = re.search(r",\s*([A-Za-z]{2})\b", seg)
    if m and m.group(1).upper() in STATE_ABBR:
        return True
    if any(re.search(rf"\b{re.escape(c)}\b", s) for c in CITY_HINTS | COUNTRY_HINTS):
        return True
    return bool(WORK_MODE_RE.search(s) and ("(" in s or re.search(r"\b(us|usa|worldwide|global)\b", s)))

def classify_segment(seg: str) -> str:
    if SALARY_RE.search(seg):
        return "salary_raw"
    if VISA_RE.search(seg):
        return "visa_note"
    if looks_like_location(seg):
        return "location"
    if WORK_MODE_RE.search(seg):
        return "work_mode"
    return "extra"

def parse_header(first_line: str) -> dict:
    segs = [s.strip() for s in first_line.split("|")]
    out = {"company": None, "role": None, "location": None, "work_mode": None,
           "salary_raw": None, "visa_note": None, "header_extra": None}
    out["company"] = segs[0] or None if segs else None
    out["role"] = (segs[1] or None) if len(segs) >= 2 else None
    extras = []
    for seg in segs[2:]:
        if not seg:
            continue
        cat = classify_segment(seg)
        if cat == "extra":
            extras.append(seg)
        elif out[cat] is None:
            out[cat] = seg
        else:
            out[cat] += "; " + seg
    out["header_extra"] = " | ".join(extras) or None
    out["parse_ok"] = bool(out["company"]) and bool(out["role"])
    return out

def load_all() -> list[dict]:
    records = []
    for f in sorted(RAW_DIR.glob("*.json")):
        if f.name == "_threads.json":
            continue
        records.extend(json.loads(f.read_text()))
    return records

def build_row(item: dict) -> dict:
    text_clean = clean_text(item.get("text"))
    first_line = text_clean.split("\n", 1)[0] if text_clean else ""
    header = parse_header(first_line)
    dt = datetime.fromtimestamp(item["time"], tz=UTC)
    return {
        "id": item["id"],
        "month": dt.strftime("%Y-%m"),
        "year": dt.year,
        "posted_date": dt.date().isoformat(),
        "company": header["company"],
        "role": header["role"],
        "location": header["location"],
        "work_mode": header["work_mode"],
        "salary_raw": header["salary_raw"],
        "visa_note": header["visa_note"],
        "header_extra": header["header_extra"],
        "parse_ok": header["parse_ok"],
        "text_len": len(text_clean),
        "has_email": bool(EMAIL_RE.search(text_clean)),
        "has_url": bool(URL_RE.search(text_clean)),
        "is_short": len(text_clean) < 100,
        "text_clean": text_clean,
        "text": item.get("text"),
    }

def main() -> None:
    records = load_all()
    n_loaded = len(records)
    dropped = [r for r in records if r.get("deleted") or r.get("dead")]
    kept = [r for r in records if not r.get("deleted") and not r.get("dead")]
    dropped_short = sum(1 for r in dropped if len(clean_text(r.get("text"))) < 100)
    df = pd.DataFrame(build_row(r) for r in kept)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["id", "month", "year", "posted_date", "company", "role", "location", "work_mode",
            "salary_raw", "visa_note", "header_extra", "parse_ok", "text_len", "has_email",
            "has_url", "is_short", "text_clean", "text"]
    df[cols].to_csv(OUT_CSV, index=False)
    sample = sorted(kept, key=lambda r: (r.get("thread_month", ""), r["id"]))[:20]
    (SAMPLES_DIR / "hn_raw_sample.json").write_text(json.dumps(sample, indent=2))
    sample_ids = {r["id"] for r in sample}
    df[df["id"].isin(sample_ids)][cols].to_csv(SAMPLES_DIR / "hn_clean_sample.csv", index=False)

    lines = [
        f"loaded: {n_loaded}",
        f"after dead/deleted filter: {len(kept)} (dropped {len(dropped)})",
        f"final rows written: {len(df)}",
        f"parse_ok overall: {df['parse_ok'].mean():.1%}",
        "parse_ok by year:",
    ]
    for year, grp in df.groupby("year"):
        lines.append(f"  {year}: {grp['parse_ok'].mean():.1%} (n={len(grp)})")
    lines.append("fill rate by field:")
    for c in ["company", "role", "location", "work_mode", "salary_raw", "visa_note", "header_extra"]:
        lines.append(f"  {c}: {df[c].notna().mean():.1%}")
    lines.append("top 20 companies:")
    for name, count in df["company"].value_counts().head(20).items():
        lines.append(f"  {name}: {count}")
    lines.append(f"is_short rows (kept set): {df['is_short'].sum()} ({df['is_short'].mean():.1%})")
    lines.append(
        f"of {len(dropped)} dropped dead/deleted rows, {dropped_short} ({dropped_short / max(len(dropped), 1):.1%}) "
        "were also short-text -- overlap between the two filters"
    )
    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n")

if __name__ == "__main__":
    main()
