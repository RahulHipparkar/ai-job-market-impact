"""Fetch raw job postings from the Apify multi-ATS scraper and save them.

Input: company list (GROUP_A/GROUP_B) and item caps via CLI args.
Output: raw JSON response saved to data/raw/.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ACTOR_ID = "scrapesage/multi-ats-job-scraper"
ENDPOINT = f"https://api.apify.com/v2/acts/{ACTOR_ID.replace('/', '~')}/run-sync-get-dataset-items"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TIMEOUT = 600  # seconds; run-sync-get-dataset-items can take minutes

GROUP_A = [
    "lever:palantir", "greenhouse:anthropic", "ashby:openai",
    "greenhouse:scaleai", "greenhouse:databricks", "ashby:sierra",
    "ashby:cohere", "ashby:harvey", "ashby:perplexity",
]
GROUP_B = [
    "ashby:ramp", "ashby:linear", "greenhouse:vercel",
    "greenhouse:figma", "greenhouse:stripe", "greenhouse:airtable",
    "greenhouse:asana", "greenhouse:webflow", "greenhouse:amplitude",
    "greenhouse:discord",
]
GROUPS = {"A": GROUP_A, "B": GROUP_B}


def build_payload(companies: list[str], max_items: int, max_per_company: int) -> dict:
    return {
        "companies": companies,
        "maxItems": max_items,
        "maxItemsPerCompany": max_per_company,
        "includeCompensation": True,
        "includeDescription": True,
        "includeRawData": False,
        "onlyNewItems": False,
        "proxyConfiguration": {"useApifyProxy": True},
    }

def fetch(
    companies: list[str], max_items: int, max_per_company: int, group: str, dry_run: bool = False
) -> Path | None:
    payload = build_payload(companies, max_items, max_per_company)
    if dry_run:
        print(json.dumps(payload, indent=2))
        return None

    load_dotenv()
    token = os.getenv("APIFY_TOKEN")
    if not token:
        sys.exit("APIFY_TOKEN not found in .env")

    try:
        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        sys.exit(f"Request timed out after {TIMEOUT}s — try a smaller company group.")
    except requests.exceptions.RequestException as exc:
        sys.exit(f"Apify request failed: {exc}")

    items = response.json()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"ats_{group.lower()}_{datetime.now(UTC).date().isoformat()}.json"
    out_path.write_text(json.dumps(items, indent=2))
    print(f"Saved {len(items)} items to {out_path}")
    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--max-per-company", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fetch(GROUPS[args.group], args.max_items, args.max_per_company, args.group, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
