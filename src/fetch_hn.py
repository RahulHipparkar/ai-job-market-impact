"""Fetch "Ask HN: Who is hiring?" threads and their top-level comments.

Input: a start/end month range via CLI args.
Output: one raw JSON file per month in data/raw/hn/, plus a thread index.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tqdm import tqdm

ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"
FIREBASE_ITEM = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "hn"
COMMENT_FIELDS = ["id", "parent", "by", "time", "text", "deleted", "dead"]

def month_range(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

def month_ts(y: int, m: int) -> int:
    return int(datetime(y, m, 1, tzinfo=UTC).timestamp())

async def fetch_threads(client: httpx.AsyncClient, start: str, end: str) -> tuple[list[dict], list[str]]:
    ey, em = (int(x) for x in end.split("-"))
    lo = month_ts(*(int(x) for x in start.split("-")))
    hi = month_ts(*((ey, em + 1) if em < 12 else (ey + 1, 1)))
    hits, page, pages = [], 0, 1
    while page < pages:
        params = {
            "query": "Ask HN: Who is hiring",
            "tags": "story,author_whoishiring",
            "numericFilters": f"created_at_i>={lo},created_at_i<{hi}",
            "hitsPerPage": 100,
            "page": page,
        }
        r = await client.get(ALGOLIA_URL, params=params)
        r.raise_for_status()
        data = r.json()
        hits += data["hits"]
        pages, page = data["nbPages"], page + 1
    by_month: dict[str, dict] = {}
    for h in hits:
        title = h["title"].lower()
        if "who is hiring" not in title or "wants to be hired" in title or "freelancer" in title:
            continue
        month = datetime.fromtimestamp(h["created_at_i"], tz=UTC).strftime("%Y-%m")
        prior = by_month.get(month)
        if prior and prior["created_at_i"] <= h["created_at_i"]:
            continue
        by_month[month] = {
            "id": int(h["objectID"]),
            "title": h["title"],
            "month": month,
            "num_comments": h.get("num_comments", 0),
            "created_at_i": h["created_at_i"],
        }
    expected = [f"{y}-{m:02d}" for y, m in month_range(start, end)]
    missing = [m for m in expected if m not in by_month]
    for m in missing:
        print(f"warning: no thread found for {m}", file=sys.stderr)
    return [by_month[m] for m in expected if m in by_month], missing

async def fetch_json(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore, retries: int = 5):
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.get(url, timeout=20)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if attempt == retries - 1:
                    print(f"  giving up on {url}: {exc}", file=sys.stderr)
                    return None
                await asyncio.sleep(min(2**attempt, 20))
    return None

async def fetch_comment(client, sem, cid: int, thread_id: int, month: str) -> dict | None:
    item = await fetch_json(client, FIREBASE_ITEM.format(id=cid), sem)
    if item is None:  # deleted comments return null — skip
        return None
    rec = {k: item.get(k) for k in COMMENT_FIELDS}
    rec["thread_id"], rec["thread_month"] = thread_id, month
    return rec

async def process_thread(client, sem, thread: dict, force: bool) -> int:
    path = RAW_DIR / f"{thread['month']}.json"
    if path.exists() and not force:
        return len(json.loads(path.read_text()))
    story = await fetch_json(client, FIREBASE_ITEM.format(id=thread["id"]), sem)
    kids = (story or {}).get("kids", [])
    results = await asyncio.gather(
        *(fetch_comment(client, sem, cid, thread["id"], thread["month"]) for cid in kids)
    )
    comments = [c for c in results if c is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comments, indent=2))
    return len(comments)

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-month", default="2019-01")
    parser.add_argument("--end-month", default=datetime.now(UTC).strftime("%Y-%m"))
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--threads-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    concurrency = min(max(args.concurrency, 1), 20)  # never exceed 20 — be a polite client
    async with httpx.AsyncClient() as client:
        threads, missing = await fetch_threads(client, args.start_month, args.end_month)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "_threads.json").write_text(json.dumps(threads, indent=2))

    if args.threads_only:
        for t in threads:
            print(f"{t['month']}  id={t['id']}  ({t['num_comments']:>4} comments)  {t['title']}")
        print(f"\n{len(threads)} threads found. Missing months: {missing or 'none'}")
        return

    sem, total_comments = asyncio.Semaphore(concurrency), 0
    async with httpx.AsyncClient() as client:
        for thread in tqdm(threads, desc="threads"):
            total_comments += await process_thread(client, sem, thread, args.force)
    print(f"\nThreads fetched: {len(threads)}")
    print(f"Total comments: {total_comments}")
    print(f"Months with no data: {missing or 'none'}")

if __name__ == "__main__":
    asyncio.run(main())
