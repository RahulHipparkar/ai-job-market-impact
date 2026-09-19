# AI job market impact

Separating generative AI from interest rates and the post-pandemic correction as explanations for the fall in tech job postings, 2015 to 2026.

## The question

Job postings in AI-exposed sectors fell by roughly 69 percent from their mid-2022 peak, while postings in control sectors fell by roughly 37 percent and recovered above their pre-pandemic level. This project asks how much of that gap generative AI explains, and how much belongs to higher interest rates and the correction of pandemic-era overhiring. It works from five data sources covering 2015 to 2026.

## Live site

https://rahulhipparkar.github.io/ai-job-market-impact/

## Data sources

| Source | Type | What it gives | Records | Dates covered |
|---|---|---|---|---|
| Hacker News | API | Monthly "Who is hiring?" posts | 48,296 postings | 2019-01-02 to 2026-09-07 |
| ATS job boards (Apify) | API | Postings from 19 companies on Greenhouse, Lever and Ashby | 4,294 postings | 2009-12-05 to 2026-09-08 |
| Indeed Hiring Lab | Download | Posting indexes by sector, metro and state, and the AI share of postings | 1,801,661 rows | 2019-01-01 to 2026-08-28 |
| BLS Public Data API | API | JOLTS, CES and OEWS series | 1,956 rows across 22 series | 2015-01-01 to 2026-08-01 |
| OEWS annual files | Download | Employment and wages by occupation | 33 trend rows from 7 workbooks | 2019 to 2025 |

## Pipeline

- Hacker News: 52,040 raw comments to 48,296 cleaned postings, dropping 3,744 dead or deleted.
- ATS job boards: 4,409 raw rows from four exports to 4,294 postings, dropping 115 duplicate job IDs.
- Indeed Hiring Lab: five already tidy files reshaped to 1,801,661 rows, no rows dropped.
- BLS API: 22 series to 1,956 rows (JOLTS 1,668, CES 280, OEWS 8).
- OEWS annual files: 9,670 rows read from seven workbooks to 33 trend rows for six occupations.

Issues are flagged in columns rather than dropped. Each cleaning script writes a report next to its output in `data/processed/`.

## Repository layout

```
data/       raw, interim and processed tables, plus small committed samples
src/        fetch and clean scripts, and the shared plot style
notebooks/  01_data_prep_eda.ipynb, the exploratory pass
figures/    charts and thumbnails used by the site
site/       Quarto source for the website
docs/       rendered site, served by GitHub Pages
results/    empty placeholder for model output
```

## Running it

Install the environment (Python 3.12):

```
uv sync
```

Create a `.env` file in the repository root with two variables:

- `APIFY_TOKEN` — created in an Apify account. How it is sent is documented at https://docs.apify.com/api/v2
- `BLS_API_KEY` — free registration at https://www.bls.gov/developers/

Fetch the raw data. `fetch_apify.py` calls a paid actor and costs money on every run, billed per job scraped, so check the group and caps before running it.

```
python src/fetch_hn.py --start-month 2019-01
python src/fetch_bls.py --survey all
python src/fetch_apify.py --group A --max-items 4000 --max-per-company 1000
python src/fetch_apify.py --group B --max-items 1200 --max-per-company 120
```

Two sources have no fetch script. Clone https://github.com/hiring-lab/job_postings_tracker and https://github.com/hiring-lab/ai-tracker into `data/raw/indeed/`, and download the national OEWS workbook for each year from https://www.bls.gov/oes/tables.htm into `data/raw/bls/oews/`.

Then clean, in this order, because the stage 2 scripts read the stage 1 output:

```
python src/clean_hn.py
python src/clean_hn_stage2.py
python src/clean_ats.py
python src/clean_ats_stage2.py
python src/clean_bls.py
python src/clean_indeed.py
python src/clean_oews_files.py
```

Then run `notebooks/01_data_prep_eda.ipynb`. The sample tables and figures on the site are rebuilt with `make_samples.py`, `make_sample_images.py` and `make_intro_figure.py`, and the site itself with `quarto render` from `site/`.

## What is not in the repository

`data/raw/` and `data/interim/` are gitignored, along with three processed files that are too large to track: `hn_postings.csv` (125 MB), `hn_clean.csv` (177 MB) and `indeed_metro.csv` (112 MB). The fetch and clean scripts regenerate all of them. Small raw and cleaned samples of every source are committed in `data/samples/`, so the shape of the data can be seen without downloading anything.

## Course

CSCI 5612 Machine Learning for Data Science, CU Boulder, Fall 2026.
