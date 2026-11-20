"""Extract pay ranges from the description text of the job board postings (Module 4 data step).

Input: the 3,718 deduplicated postings (text_clusters.load_postings, which reads ats_final.csv).
Output: data/module4/pay_ranges.csv, one row per posting with the range chosen from its
description, the currency, interval and basis, the annualised values and their flags.
Nothing under data/processed is written.

Rules
- A range is two amounts joined by -, –, —, "to" or "and". The first carries a currency symbol
  ($, US$, C$, CA$, A$, S$, £, €), or the pair is followed by a currency code. Amounts may use
  commas, decimals, a k suffix, or dots as thousands separators (€155.000). An amount with an
  M or B suffix is a company or deal size, not pay, and the range is skipped.
- It needs a pay word (salary, pay, compensation, range, rate, wage, earnings, OTE, base,
  stipend, hourly, annual) in its label or the 250 characters before it, or must follow a kept
  range within 200 characters (the later bullets of a zone list).
- Currency: a code right after the range wins, so "$196,800 - $270,600 CAD" is CAD. For a bare $,
  next come a code in the text since the previous range ("between CAD", "figures cited below in
  CAD"), then the currency of the previous range in the same list, then a mention of Canada. A
  bare $ left over is "unknown" when the posting is located outside the US, and otherwise USD,
  flagged as assumed. £, €, US$, C$, A$ and S$ decide on their own.
- Interval: hour or month when the text right after the range or its label says so. Otherwise it
  is inferred from magnitude as clean_ats.infer_interval does (below 1,000 is hourly). Hourly
  figures are annualised x2080 (clean_ats.HOURS_PER_YEAR) and monthly x12; both are flagged.
- Basis: "ote" when the range's label or the text right after it says OTE, on-target or total
  target; "base" when it says base; otherwise "unspecified".
- Placeholder: both amounts below 10 (such as $1-$1). Suspect: annualised outside the clean_ats
  thresholds (20,000 to 1,000,000), or minimum above maximum.
- Several ranges in one posting (location zones, state disclosures, an OTE range and its base):
  keep the valid ranges, prefer USD, then prefer a non-OTE range, then take the highest
  annualised midpoint (ties go to the first listed). Employers list zones from the highest-cost
  area down (Zone 1, Zone A, "San Francisco, New York, Seattle"), so the highest range is the
  top-zone band, the same band a single-location posting in those cities would quote. Matching
  the posting's own location is not possible for most of these postings, because the zones are
  defined on the employer's website rather than in the text. multi_zone marks the postings where
  this choice was made among two or more different USD base-pay ranges, so an analysis can be
  run without them.
- Usable as a dollar target (text_usable_usd): USD, not a placeholder, not suspect, not OTE, and
  not an hourly figure whose interval was only inferred from its magnitude. A stated hourly rate
  ("per hour", "/hr", "Hourly Rate") stays usable.

usage: python src/extract_pay_ranges.py
"""
import re

import numpy as np
import pandas as pd

try:
    from src import text_clusters as tc
    from src.clean_ats import HOURLY_THRESHOLD, HOURS_PER_YEAR, SUSPECT_HIGH, SUSPECT_LOW
except ImportError:
    import text_clusters as tc
    from clean_ats import HOURLY_THRESHOLD, HOURS_PER_YEAR, SUSPECT_HIGH, SUSPECT_LOW

OUT_CSV = tc.ROOT / "data" / "module4" / "pay_ranges.csv"
MONTHS_PER_YEAR = 12
PLACEHOLDER_BELOW = 10
LOOKBACK = 250  # characters searched for a pay word or a currency code before a range
CHAIN = 200  # a range this close after a kept range belongs to the same list
CODES = "USD|CAD|GBP|EUR|AUD|SGD|CHF|INR|JPY"
US_LOCATION = re.compile(r", [A-Z]{2}$|^(Remote|United States|North America)$")
SYMBOLS = {"US$": "USD", "CA$": "CAD", "C$": "CAD", "A$": "AUD", "AU$": "AUD", "S$": "SGD", "SG$": "SGD",
           "£": "GBP", "€": "EUR", "$": "USD"}
CUR = r"(?:US\$|CA\$|C\$|AU\$|A\$|SG\$|S\$|\$|£|€)"
NUM = r"\d{1,3}(?:[,.]\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
SUFFIX = r"(?:\s?[kKmMbB](?![A-Za-z]))?"
SEP = r"\s*(?:-|–|—|\bto\b|\band\b)\s*"
RANGE_SYMBOL = re.compile(rf"(?P<c1>{CUR})\s?(?P<n1>{NUM})(?P<s1>{SUFFIX}){SEP}(?P<c2>{CUR})?\s?(?P<n2>{NUM})(?P<s2>{SUFFIX})")
RANGE_CODE = re.compile(rf"(?<![\d$£€.,])(?P<n1>{NUM})(?P<s1>{SUFFIX}){SEP}(?P<n2>{NUM})(?P<s2>{SUFFIX})\s?(?P<code>{CODES})\b")
CODE_AFTER = re.compile(rf"^\s*\(?\s*({CODES})\b")
PAY_WORD = re.compile(r"salar|\bpay\b|compensation|\branges?\b|\brates?\b|\bwage|earnings|\bOTE\b|\bbase\b|stipend|"
                      r"hourly|annual", re.I)
OTE = re.compile(r"\bOTE\b|on.target|total target|target (?:cash|earnings|compensation)|"
                 r"inclusive of (?:bonus|commission)", re.I)
BASE = re.compile(r"\bbase\b", re.I)
HOUR = re.compile(r"/\s?h(?:ou)?r\b|per hour|an hour|hourly", re.I)
MONTH = re.compile(r"/\s?mo(?:nth)?\b|per month|a month|monthly", re.I)

COLUMNS = ["jobId", "company_clean", "n_ranges_found", "n_ranges_valid", "n_ranges_distinct", "n_pool_distinct",
           "multi_zone", "pick_rule", "pick_is_first", "text_min", "text_max", "text_currency", "currency_source", "text_interval",
           "interval_source", "interval_converted", "text_annual_min", "text_annual_max", "text_basis",
           "text_placeholder", "text_suspect", "text_usable_usd", "text_label", "text_span"]


def amount(number: str, suffix: str) -> float:
    """A number as written: dots are thousands separators only in the 155.000 form."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", number):
        value = float(number.replace(".", ""))
    else:
        value = float(number.replace(",", ""))
    return value * 1000 if suffix.strip().lower() == "k" else value


def label_before(text: str, start: int, floor: int) -> str:
    """The range's label: its own line before it, or the line above when the range starts a line."""
    lines = text[max(floor, start - 160):start].split("\n")
    own = lines[-1].strip()
    if re.search(r"[A-Za-z]", own) or len(lines) == 1:
        return own[-120:]
    return (lines[-2].strip() if len(lines) > 1 else "")[-120:]


def currency_of(m: re.Match, after: str, window: str, previous: dict | None, location: str) -> tuple[str, str]:
    """The range's currency and where it came from, by the order in the module docstring."""
    code = m.groupdict().get("code") or (CODE_AFTER.match(after).group(1) if CODE_AFTER.match(after) else None)
    if code:
        return code, "code after"
    symbol = m.group("c1")
    if symbol != "$":
        return SYMBOLS[symbol], "symbol"
    near = re.findall(rf"\b({CODES})\b", window)
    if near:
        return near[-1], "code before"
    if previous is not None:
        return previous["text_currency"], "same list"
    if re.search(r"canad", window, re.I):
        return "CAD", "mentions Canada"
    if location and not US_LOCATION.search(location):
        return "unknown", "bare $, non-US location"
    return "USD", "bare $, assumed"


def candidates(text: str, location: str = "") -> list[dict]:
    """Every pay range in the text, in order, with its currency, interval and basis."""
    found, taken = [], []
    for rx in (RANGE_SYMBOL, RANGE_CODE):
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            if any(g.strip().lower() in ("m", "b") for g in (m.group("s1"), m.group("s2"))):
                continue
            taken.append((m.start(), m.end()))
            found.append(m)
    found.sort(key=lambda m: m.start())

    out, prev_end, kept_end = [], 0, None
    for m in found:
        after = text[m.end():m.end() + 60].split("\n")[0]
        label = label_before(text, m.start(), prev_end)
        before = text[max(0, m.start() - LOOKBACK):m.start()]
        window = text[max(prev_end, m.start() - LOOKBACK):m.start()]
        chained = kept_end is not None and m.start() - kept_end <= CHAIN
        prev_end = m.end()
        if not (PAY_WORD.search(label) or PAY_WORD.search(before) or chained):
            continue
        kept_end = m.end()
        if "c1" in m.groupdict():
            currency, currency_source = currency_of(m, after, window, out[-1] if chained and out else None, location)
        else:
            currency, currency_source = m.group("code"), "code after"
        lo, hi = amount(m.group("n1"), m.group("s1")), amount(m.group("n2"), m.group("s2"))
        near = f"{label} {after[:30]}"
        if HOUR.search(near):
            interval, source = "hour", "stated"
        elif MONTH.search(near):
            interval, source = "month", "stated"
        else:
            interval, source = ("hour" if lo < HOURLY_THRESHOLD else "year"), "inferred"
        factor = {"hour": HOURS_PER_YEAR, "month": MONTHS_PER_YEAR, "year": 1}[interval]
        basis = "ote" if OTE.search(near) else "base" if BASE.search(label) else "unspecified"
        out.append({"text_min": lo, "text_max": hi, "text_currency": currency, "currency_source": currency_source,
                    "text_interval": interval, "interval_source": source, "interval_converted": factor != 1,
                    "text_annual_min": lo * factor, "text_annual_max": hi * factor, "text_basis": basis,
                    "text_placeholder": hi < PLACEHOLDER_BELOW,
                    "text_label": label, "text_span": m.group(0).replace("\n", " ")})
    for r in out:
        r["text_suspect"] = bool(r["text_annual_min"] < SUSPECT_LOW or r["text_annual_max"] > SUSPECT_HIGH or
                                 r["text_annual_min"] > r["text_annual_max"]) and not r["text_placeholder"]
    return out


def choose(ranges: list[dict]) -> dict:
    """The one range kept for a posting, by the rule in the module docstring."""
    valid = [r for r in ranges if not r["text_placeholder"] and not r["text_suspect"]]
    distinct = {(r["text_annual_min"], r["text_annual_max"], r["text_currency"]) for r in valid}
    base = {"n_ranges_found": len(ranges), "n_ranges_valid": len(valid), "n_ranges_distinct": len(distinct)}
    if not ranges:
        return {**base, "pick_rule": "none"}
    if not valid:
        return {**base, **ranges[0], "pick_rule": "no valid range", "pick_is_first": True}
    pool = [r for r in valid if r["text_currency"] == "USD"] or valid
    pool = [r for r in pool if r["text_basis"] != "ote"] or pool
    pick = max(pool, key=lambda r: (r["text_annual_min"] + r["text_annual_max"]) / 2)  # max keeps the first tie
    rule = "single" if len(distinct) == 1 else "highest of several"
    n_pool = len({(r["text_annual_min"], r["text_annual_max"]) for r in pool})
    multi = n_pool > 1 and pick["text_currency"] == "USD" and pick["text_basis"] != "ote"
    return {**base, **pick, "n_pool_distinct": n_pool, "multi_zone": multi, "pick_rule": rule,
            "pick_is_first": pick is valid[0]}


def extract(postings: pd.DataFrame) -> pd.DataFrame:
    rows = [{"jobId": j, "company_clean": c,
             **choose(candidates(t if isinstance(t, str) else "", loc if isinstance(loc, str) else ""))}
            for j, c, t, loc in zip(postings["jobId"], postings["company_clean"], postings["description_clean"],
                                    postings["location_clean"])]
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    found = df["n_ranges_found"] > 0
    inferred_hourly = df["interval_converted"].eq(True) & df["interval_source"].eq("inferred")
    df["text_usable_usd"] = (found & df["text_currency"].eq("USD") & ~df["text_placeholder"].eq(True) &
                             ~df["text_suspect"].eq(True) & df["text_basis"].ne("ote") & ~inferred_hourly)
    df["multi_zone"] = df["multi_zone"].eq(True)
    for col in ["pick_is_first", "interval_converted", "text_placeholder", "text_suspect"]:
        df[col] = df[col].where(found, np.nan)
    return df


def main() -> None:
    postings, _ = tc.load_postings()
    df = extract(postings)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    found = df["n_ranges_found"] > 0
    print(f"postings {len(df):,}; with a pay range in the text {found.sum():,}; usable USD annual "
          f"{df['text_usable_usd'].sum():,}; several distinct valid ranges {(df['n_ranges_distinct'] > 1).sum():,}; "
          f"multi_zone {df['multi_zone'].sum():,}")
    print(f"saved {OUT_CSV.relative_to(tc.ROOT)}")


if __name__ == "__main__":
    main()
