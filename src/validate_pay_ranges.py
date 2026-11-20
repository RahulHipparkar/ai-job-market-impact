"""Validation record for the pay ranges read from the description text (Module 4 data step).

Recomputed from data/module4/pay_ranges.csv and the postings it was read from (text_clusters.load_postings):
1. the Module 3 subset postings that have both a usable salary field and a pay range in the text: how often
   the range read from the text matches the field, and every mismatch with its surrounding text;
2. the 80 text-only extractions checked by hand, in two rounds, each with its surrounding text and the
   reviewer's notes;
3. the rule for postings that list several pay ranges, and what is kept out of the dollar target.

The reviewer's verdicts and notes were recorded by hand when the rows were read; everything else is
recomputed on every run.

Output: results/pay_range_validation.txt

usage: python src/validate_pay_ranges.py
"""
import textwrap

import pandas as pd

try:
    from src import extract_pay_ranges as xp
    from src import regression_models as rm
    from src import text_clusters as tc
    from src import text_nb as nb
except ImportError:
    import extract_pay_ranges as xp
    import regression_models as rm
    import text_clusters as tc
    import text_nb as nb

OUT_PATH = tc.ROOT / "results" / "pay_range_validation.txt"
TOLERANCE = 0.5  # dollars: two values closer than this match
BEFORE, AFTER = 220, 110  # characters of description shown around an extracted range
WIDTH = 104
RULE = "=" * WIDTH

ROUND1 = (30, 42)  # validation round: (rows, seed), one random draw from the text-only pool
ROUND2 = ([("Databricks", 20), ("Anthropic", 20), ("Scale AI", 10)], 7)  # pre-modelling check: one draw per company
VERDICTS = {
    1: "30 of 30 extracted values match the text.",
    2: "50 of 50 extracted values, currencies and intervals match the text.",
}
ROUND2_GENERAL = ("Anthropic quotes its ranges under a note that sales roles are given as on-target earnings, "
                  "without marking which roles those are.")
# The reviewer's notes, by jobId. Kinds: location (the highest-zone rule picked a band quoted for another
# location), basis (the text leaves open whether the range is base pay), excluded (rightly kept out of the
# dollar target).
NOTES = {
    "8590452002": ("basis", ("Amplitude lists one range per state. This San Francisco posting was given the New York "
                             "City range, which is the same $110,000 - $165,000 as its California range. Each range "
                             "is described as 'base salary (inclusive of bonus or commission)'; that wording lies "
                             "beyond the 30 characters the extractor checks for OTE, so the range is used. Whether it "
                             "includes commission is unclear.")),
    "4481921005": ("location", ("An Austin, TX posting given the San Francisco, New York, Seattle band by the "
                                "highest-zone rule; the text puts Texas in the second band, $165,600-$263,304.")),
    "5400160008": ("excluded", "A GBP range: kept in the file and excluded from the dollar target."),
    "6883068002": ("location", "A Washington, DC intern given the SF Bay Area hourly rate by the highest-zone rule."),
    "8611106002": ("location", ("A 'Central - United States' posting given Zone 1 by the highest-zone rule; the text "
                                "does not say which zone that location is in.")),
    "4717648005": ("location", ("An Austin, TX posting given the Washington DC/Hawaii band by the highest-zone rule; "
                                "Texas is not listed.")),
    "8569606002": ("basis", ("A sales director. Databricks quotes 'the expected salary range for non-commissionable "
                             "roles or on-target earnings for commissionable roles' without saying which applies, so "
                             "this range may be on-target earnings.")),
}


# ---------------------------------------------------------------- data

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 3,718 postings joined to their extracted range, and the Module 3 subset with its salary target."""
    postings, _ = tc.load_postings()
    postings["jobId"] = postings["jobId"].astype(str)
    pr = pd.read_csv(xp.OUT_CSV, dtype={"jobId": str})
    df = postings.merge(pr.drop(columns="company_clean"), on="jobId", how="left", validate="1:1")
    subset, _ = nb.load_subset()
    df["in_subset"] = df["jobId"].isin(subset["jobId"].astype(str))
    return df, rm.salary_frame(subset)


def structured_band(df: pd.DataFrame) -> pd.Series:
    """A usable salary field, as in regression_models.salary_frame: both ends, not suspect, USD."""
    return (df["salary_annual_min"].notna() & df["salary_annual_max"].notna()
            & ~df["salary_suspect"].eq(True) & df["salary_currency"].eq("USD"))


def has_text_range(df: pd.DataFrame) -> pd.Series:
    return df["n_ranges_found"] > 0


def hand_checked(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """The two hand-checked samples, drawn again exactly as they were drawn, and the pool sizes."""
    pool = df[has_text_range(df) & df["salary_annual_min"].isna()]
    n, seed = ROUND1
    round1 = pool.sample(n, random_state=seed)
    rest = pool[~pool["jobId"].isin(set(round1["jobId"]))]
    plan, seed = ROUND2
    round2 = pd.concat([rest[rest["company_clean"] == c].sample(k, random_state=seed) for c, k in plan])
    assert set(NOTES) <= set(round1["jobId"]) | set(round2["jobId"]), "a noted jobId is not in the samples"
    return round1, round2, len(pool), len(rest)


# ---------------------------------------------------------------- formatting

def money(value: float) -> str:
    return f"{value:,.0f}" if value == round(value) else f"{value:,.2f}"


def wrap(text: str, indent: int) -> list[str]:
    return textwrap.wrap(text, WIDTH, initial_indent=" " * indent, subsequent_indent=" " * indent,
                         break_long_words=False, break_on_hyphens=False)


def surrounding(r: pd.Series) -> str:
    """The description around the extracted range: the range marked [[ ]], line breaks shown as |."""
    desc = r["description_clean"]
    flat = desc.replace("\n", " ")  # text_span was taken from the text with line breaks as spaces
    span, label = r["text_span"], r["text_label"]
    starts, i = [], flat.find(span)
    while i >= 0:
        starts.append(i)
        i = flat.find(span, i + 1)
    if not starts:
        return "(range not found in the text)"
    labelled = [s for s in starts if isinstance(label, str) and label and label in flat[max(0, s - 200):s]]
    s = (labelled or starts)[0]
    e = s + len(span)
    a, b = max(0, s - BEFORE), min(len(desc), e + AFTER)
    piece = desc[a:s] + "[[" + flat[s:e] + "]]" + desc[e:b]
    return ("…" if a else "") + " ".join(piece.replace("\n", " | ").split()) + ("…" if b < len(desc) else "")


def extracted(r: pd.Series) -> str:
    per = {"year": "a year", "hour": "an hour", "month": "a month"}[r["text_interval"]]
    out = (f"'{r['text_span']}', {r['text_currency']}, {per} (interval {r['interval_source']}), "
           f"basis {r['text_basis']}")
    if r["interval_converted"] is True or r["interval_converted"] == 1.0:
        out += f" -> {money(r['text_annual_min'])}-{money(r['text_annual_max'])} a year"
    return out


def posting_line(k: int, r: pd.Series) -> str:
    return f"{k:>2}. jobId {r['jobId']} | {r['company_clean']} | {r['title']} | {r['location_clean']}"


# ---------------------------------------------------------------- sections

def matches(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lo = (df["salary_annual_min"] - df["text_annual_min"]).abs() < TOLERANCE
    hi = (df["salary_annual_max"] - df["text_annual_max"]).abs() < TOLERANCE
    return lo, hi


def agreement_lines(df: pd.DataFrame) -> list[str]:
    band, text = structured_band(df), has_text_range(df)
    sub = df[df["in_subset"] & band & text]
    lo, hi = matches(sub)
    both = lo & hi
    full = df[band & text]
    flo, fhi = matches(full)
    lines = [RULE, "1. THE RANGE READ FROM THE TEXT AGAINST THE SALARY FIELD", "",
             *wrap(f"Module 3 subset ({int(df['in_subset'].sum()):,} postings): {int((df['in_subset'] & band).sum())} "
                   f"have a usable salary field, and {len(sub)} of those also state a pay range in the text. A value "
                   f"matches when it is within ${TOLERANCE:.2f} of the other, after converting both to a year.", 0),
             "",
             f"  minimum matches  {int(lo.sum())} of {len(sub)}",
             f"  maximum matches  {int(hi.sum())} of {len(sub)}",
             f"  both match       {int(both.sum())} of {len(sub)} ({both.mean():.1%})",
             "",
             *wrap(f"For reference, the full set of {len(df):,} postings: {len(full)} with both; minimum matches "
                   f"{int(flo.sum())}, maximum {int(fhi.sum())}, both {int((flo & fhi).sum())}.", 0),
             "",
             *wrap(f"The {int((~both).sum())} subset postings where they differ. In each, the extracted range is "
                   "what the text says; the text and the salary field give different bands.", 0)]
    for k, (_, r) in enumerate(sub[~both].sort_values(["company_clean", "jobId"]).iterrows(), 1):
        per = r["salaryInterval"] if isinstance(r["salaryInterval"], str) else "interval not given"
        same = "minimum matches" if lo[r.name] else "maximum matches" if hi[r.name] else "neither end matches"
        lines += ["", posting_line(k, r),
                  *wrap(f"salary field: {money(r['salaryMin'])}-{money(r['salaryMax'])} {r['salaryCurrency']}, {per} "
                        f"(a year: {money(r['salary_annual_min'])}-{money(r['salary_annual_max'])})", 4),
                  *wrap(f"text range: {extracted(r)}; {same}", 4),
                  *wrap("text: " + surrounding(r), 4)]
    return lines


def hand_check_lines(df: pd.DataFrame) -> list[str]:
    round1, round2, n_pool, n_rest = hand_checked(df)
    n1, seed1 = ROUND1
    plan, seed2 = ROUND2
    drawn = pd.concat([round1, round2])
    found = sum(r["text_span"].replace("\n", " ") in r["description_clean"].replace("\n", " ")
                for _, r in drawn.iterrows())
    kinds = pd.Series([NOTES[j][0] for j in drawn["jobId"] if j in NOTES]).value_counts()
    lines = ["", RULE, "2. TEXT-ONLY EXTRACTIONS CHECKED BY HAND", "",
             *wrap(f"Pool: the {n_pool:,} postings in the full set with a pay range in the text and no salary field. "
                   f"Round 1, during the validation of the extractor: {n1} drawn at random from the pool "
                   f"(pandas sample, random_state={seed1}). Round 2, before any model was fitted: "
                   + ", ".join(f"{k} {c}" for c, k in plan)
                   + f", drawn from the {n_rest:,} not in round 1 (random_state={seed2} within each company).", 0),
             "",
             *wrap("Each row was read against the text around it. Verdicts, recorded by hand when the rows were "
                   f"read: round 1, {VERDICTS[1]} Round 2, {VERDICTS[2]}", 0),
             "",
             *wrap(f"Recomputed: the extracted range appears word for word in the posting's text in {found} of "
                   f"{len(drawn)}. Notes below record the judgement calls: "
                   + ", ".join(f"{kinds.get(k, 0)} {k}" for k in ["location", "basis", "excluded"]) + ".", 0)]
    for label, sample, extra in [(f"ROUND 1 ({len(round1)})", round1, None), (f"ROUND 2 ({len(round2)})", round2,
                                                                              ROUND2_GENERAL)]:
        lines += ["", label]
        if extra:
            lines += wrap("General note: " + extra, 0)
        for k, (_, r) in enumerate(sample.iterrows(), 1):
            usable = "usable as a salary" if r["text_usable_usd"] else "not usable as a salary"
            zone = "; several ranges, highest-zone rule" if r["multi_zone"] else ""
            lines += ["", posting_line(k, r), *wrap(f"extracted: {extracted(r)}; {usable}{zone}", 4),
                      *wrap("text: " + surrounding(r), 4)]
            if r["jobId"] in NOTES:
                kind, note = NOTES[r["jobId"]]
                lines += wrap(f"note ({kind}): {note}", 4)
    return lines


def rule_lines(df: pd.DataFrame, sf: pd.DataFrame) -> list[str]:
    found = has_text_range(df)
    usable = df["text_usable_usd"].eq(True)
    inferred_hourly = df["interval_converted"].eq(True) & df["interval_source"].eq("inferred")
    stated_hourly = df["interval_converted"].eq(True) & df["interval_source"].eq("stated")
    # each excluded posting is counted once, under the first reason that applies, in the order of this list
    reasons = [("placeholder only (both amounts below $10, such as $1-$1)", df["text_placeholder"].eq(True)),
               ("outside the clean_ats salary thresholds, or minimum above maximum", df["text_suspect"].eq(True)),
               ("currency not USD (GBP, EUR, CAD, AUD and others)",
                df["text_currency"].ne("USD") & df["text_currency"].ne("unknown")),
               ("currency unknown (a bare $ in a posting located outside the US)", df["text_currency"].eq("unknown")),
               ("on-target earnings (OTE) only", df["text_basis"].eq("ote")),
               ("hourly, with the interval only inferred from the amount", inferred_hourly)]

    def count(mask: pd.Series) -> str:
        return f"{int((mask & found).sum()):>8,} {int((mask & found & df['in_subset']).sum()):>7,}"

    n_location = sum(kind == "location" for kind, _ in NOTES.values())
    lines = ["", RULE, "3. SEVERAL RANGES IN ONE POSTING, AND WHAT IS LEFT OUT", "",
             *wrap("Rule: keep the valid ranges (not a placeholder, not suspect), prefer USD, then prefer a range "
                   "that is not on-target earnings, then take the highest yearly midpoint, ties going to the first "
                   "listed. Employers list pay zones from the highest-cost area down (Zone 1, Zone A, 'San Francisco, "
                   "New York, Seattle'), so the highest range is the top-zone band, the band a posting in those "
                   "cities would quote. The posting's own location cannot be matched to a zone for most of these "
                   "postings, because the zones are defined on the employer's website. multi_zone marks the "
                   "postings where the pick was made among two or more different USD base-pay ranges. The rule "
                   f"overstates pay for a posting in a lower-cost zone: the hand checks found {n_location} such "
                   "picks among the 80 (section 2, notes marked location).", 0),
             "", f"{'':66}{'full set':>8} {'subset':>7}",
             f"{'postings':66}{len(df):>8,} {int(df['in_subset'].sum()):>7,}",
             f"{'  with a pay range in the text':66}{count(found)}",
             f"{'    more than one distinct valid range':66}{count(df['n_ranges_distinct'] > 1)}",
             f"{'    multi_zone: picked among two or more different USD base ranges':66}{count(df['multi_zone'])}",
             f"{'    range picked is not the first one listed':66}{count(df['pick_is_first'].eq(False))}",
             "", "Postings whose picked range is left out of the dollar target (first reason that applies):"]
    left = pd.Series(False, index=df.index)
    for name, mask in reasons:
        first = mask & ~left
        lines.append(f"{'  ' + name:66}{count(first)}")
        left |= mask
    lines += [f"{'  left out, all reasons':66}{count(~usable)}",
              f"{'usable as a yearly US dollar salary (text_usable_usd)':66}{count(usable)}",
              f"{'  of which an hourly rate stated in the text, x 2,080 hours':66}{count(usable & stated_hourly)}",
              "", "The postings left out only because their hourly interval was inferred from the amount:"]
    for _, r in df[found & first].iterrows():  # first: the last reason in the list, inferred hourly
        lines += wrap(f"jobId {r['jobId']} | {r['company_clean']} | '{r['text_span']}' under '{r['text_label']}' "
                      f"(would have been {money(r['text_annual_min'])}-{money(r['text_annual_max'])} a year); "
                      f"in the subset: {r['in_subset']}", 2)

    src = sf["salary_source"]
    used = sf["multi_zone_used"]
    lines += ["", *wrap(f"The salary target of the regression (subset): {int((src == 'structured').sum())} from the "
                        f"salary field, {int((src == 'text').sum())} added from the text, "
                        f"{int((src != 'none').sum())} in all. Of the text salaries, {int(used.sum())} are multi_zone "
                        f"picks: {int((used & sf['split'].eq('train')).sum())} in the training set and "
                        f"{int((used & sf['split'].eq('test')).sum())} in the test set. How much the rule can add to "
                        "those salaries is in results/regression_report.txt.", 0)]
    return lines


def main() -> None:
    df, sf = load()
    assert int(structured_band(df[df["in_subset"]]).sum()) == int((sf["salary_source"] == "structured").sum())
    header = ["PAY RANGES READ FROM THE DESCRIPTION TEXT: VALIDATION RECORD",
              *wrap(f"Recomputed by src/validate_pay_ranges.py from {xp.OUT_CSV.relative_to(tc.ROOT)} and the "
                    f"{len(df):,} deduplicated postings it was read from. Salary field: the job board's structured "
                    "band, usable when both ends are present, it is not suspect and it is in US dollars.", 0), ""]
    report = header + agreement_lines(df) + hand_check_lines(df) + rule_lines(df, sf)
    OUT_PATH.write_text("\n".join(report) + "\n")
    print(f"saved {OUT_PATH.relative_to(tc.ROOT)} ({len(report):,} lines)")


if __name__ == "__main__":
    main()
