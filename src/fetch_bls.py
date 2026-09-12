"""Fetch raw BLS time series (JOLTS, CES, OEWS) and save untouched responses.

Input: series IDs (JOLTS_SERIES/CES_SERIES/OEWS_SERIES) and a year range
via CLI args. Output: one raw JSON file per survey in data/raw/bls/, plus
a series catalog.
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bls"
BATCH_SIZE = 50

# JOLTS (monthly, seasonally adjusted) — job openings, hires, quits, and
# layoffs measure labor-market churn, a ground-truth hiring-demand signal
# independent of what actually gets posted online.
JOLTS_SERIES = {
    "Total nonfarm: job openings": "JTS000000000000000JOL",
    "Total nonfarm: hires": "JTS000000000000000HIL",
    "Total nonfarm: quits": "JTS000000000000000QUL",
    "Total nonfarm: layoffs and discharges": "JTS000000000000000LDL",
    "Information: job openings": "JTS510000000000000JOL",
    "Information: hires": "JTS510000000000000HIL",
    "Information: quits": "JTS510000000000000QUL",
    "Information: layoffs and discharges": "JTS510000000000000LDL",
    "Professional and business services: job openings": "JTS540099000000000JOL",
    "Professional and business services: hires": "JTS540099000000000HIL",
    "Professional and business services: quits": "JTS540099000000000QUL",
    "Professional and business services: layoffs and discharges": "JTS540099000000000LDL",
}

# CES (monthly, seasonally adjusted) — all-employee headcounts show
# whether tech-adjacent sectors are actually growing payroll, the
# baseline hiring signal job postings are meant to reflect.
CES_SERIES = {
    "Information: all employees": "CES5000000001",
    "Computer systems design and related services: all employees": "CES6054150001",
}

# OEWS (annual) — employment level and median annual wage by detailed
# occupation: ground truth for which tech roles are growing and what
# they pay, to compare against posting-derived role/salary signals.
OEWS_SERIES = {
    "Software Developers: employment": "OEUN000000000000015125201",
    "Software Developers: median annual wage": "OEUN000000000000015125213",
    "Data Scientists: employment": "OEUN000000000000015205101",
    "Data Scientists: median annual wage": "OEUN000000000000015205113",
    "Computer and Information Research Scientists: employment": "OEUN000000000000015122101",
    "Computer and Information Research Scientists: median annual wage": "OEUN000000000000015122113",
    "Computer Occupations, All Other: employment": "OEUN000000000000015129901",
    "Computer Occupations, All Other: median annual wage": "OEUN000000000000015129913",
}

SURVEYS = {"jolts": JOLTS_SERIES, "ces": CES_SERIES, "oews": OEWS_SERIES}

def chunked(items: list, size: int = BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i : i + size]

def build_payload(series_ids: list[str], start_year: str, end_year: str, key: str) -> dict:
    return {
        "seriesid": series_ids, "startyear": start_year, "endyear": end_year,
        "registrationkey": key, "catalog": True,
    }

def post_with_retry(payload: dict, retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.post(ENDPOINT, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as exc:
            if attempt == retries - 1:
                print(f"  giving up on batch: {exc}", file=sys.stderr)
                return None
            time.sleep(min(2**attempt, 20))
    return None

def fetch_survey(name: str, series: dict, start_year: str, end_year: str) -> tuple[list, int]:
    ids_by_id = {v: k for k, v in series.items()}
    responses, returned = [], 0
    for batch in chunked(list(series.values())):
        payload = build_payload(batch, start_year, end_year, os.environ["BLS_API_KEY"])
        data = post_with_retry(payload)
        if data is None:
            continue
        if data.get("status") != "REQUEST_SUCCEEDED":
            names = [ids_by_id.get(i, i) for i in batch]
            print(f"  batch failed for {name}: {data.get('message')} | series: {names}", file=sys.stderr)
            continue
        if data.get("message"):
            print(f"  {name} messages: {data['message']}", file=sys.stderr)
        responses.append(data)
        returned += len(data.get("Results", {}).get("series", []))
        time.sleep(1)  # respect 50 requests / 10s
    return responses, returned

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", choices=[*SURVEYS, "all"], default="all")
    parser.add_argument("--start-year", default="2015")
    parser.add_argument("--end-year", default=str(datetime.now(UTC).year))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = list(SURVEYS) if args.survey == "all" else [args.survey]
    if args.dry_run:
        for name in names:
            for batch in chunked(list(SURVEYS[name].values())):
                print(json.dumps(build_payload(batch, args.start_year, args.end_year, "<BLS_API_KEY>"), indent=2))
        return
    load_dotenv()
    if not os.getenv("BLS_API_KEY"):
        sys.exit("BLS_API_KEY not found in .env")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date().isoformat()
    catalog: dict[str, dict] = {}
    requested = returned = 0
    for name in names:
        series = SURVEYS[name]
        requested += len(series)
        responses, n = fetch_survey(name, series, args.start_year, args.end_year)
        returned += n
        (RAW_DIR / f"{name}_{today}.json").write_text(json.dumps(responses, indent=2))
        for resp in responses:
            for s in resp.get("Results", {}).get("series", []):
                readable = next((k for k, v in series.items() if v == s["seriesID"]), s["seriesID"])
                catalog[s["seriesID"]] = {"name": readable, "catalog": s.get("catalog", {})}
    (RAW_DIR / "_series_catalog.json").write_text(json.dumps(catalog, indent=2))
    print(f"\nSeries requested: {requested}\nSeries returned: {returned}\nDate range covered: {args.start_year}-{args.end_year}")

if __name__ == "__main__":
    main()
