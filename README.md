# AI job market impact

Separating AI from interest rates and the post-pandemic correction as explanations for the fall in tech job postings, 2019 to 2026.

## The question

From their peak in March 2022 to August 2026, job postings in AI-exposed sectors fell by roughly 67 percent, while postings in control sectors fell by roughly 36 percent and recovered above their pre-pandemic level. This project asks how much of that gap AI explains, and how much belongs to higher interest rates and the correction of pandemic-era overhiring. It works from five data sources covering 2019 to 2026.

## Live site

https://rahulhipparkar.github.io/ai-job-market-impact/

## Status

- Data preparation and exploratory analysis — complete
- Clustering (k-means and hierarchical) and PCA — complete
- Naive Bayes (multinomial and Bernoulli) and decision trees — complete
- Regression, AdaBoost, gradient boosting, XGBoost and the conclusions — to come

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
src/        fetch and clean scripts, clustering, PCA, Naive Bayes and decision tree modules, figure and sample scripts, and the shared plot style
notebooks/  01_data_prep_eda.ipynb, the exploratory pass, 02_clustering_pca.ipynb, clustering and PCA,
            and 03_naive_bayes_dt.ipynb, Naive Bayes and decision trees
figures/    charts and thumbnails used by the site
site/       Quarto source for the website
docs/       rendered site, served by GitHub Pages
results/    text reports written by the analysis notebooks
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

Then run `notebooks/02_clustering_pca.ipynb` with the `ai-job-market` kernel. It reads `data/processed/ats_final.csv` and `data/processed/indeed_sector.csv`, and writes the cluster labels for each posting to `data/processed/ats_clusters.csv`, a text report to `results/clustering_report.txt`, and its charts to `figures/`. It uses three modules:

- `src/text_clusters.py` — TF-IDF, k-means, hierarchical clustering and SVD helpers for the job board postings, run on the description text as-is and with each company's boilerplate removed.
- `src/cluster_plots.py` — the charts the notebook draws once for each of those two text versions.
- `src/sector_pca.py` — PCA on the Indeed sector panel, to isolate the shock common to every sector.

The Module 2 sample tables and explanatory figures on the site come from four more scripts:

- `src/make_module2_samples.py` — writes samples of the clustering and PCA inputs to `data/samples/` and draws them as PNG tables.
- `src/make_clustering_concept_figures.py` — the two explanatory figures on the Clustering page, drawn from small made-up data.
- `src/make_pca_concept_figures.py` — the two explanatory figures on the PCA page, drawn from small made-up data.
- `src/make_intro_postings_figure.py` — the software development postings chart on the Introduction page.

Then run `notebooks/03_naive_bayes_dt.ipynb` with the same kernel. It does not need the output of the first two notebooks. It reads `data/processed/ats_final.csv`, plus the raw job board exports in `data/raw/ats/` that `clean_ats.py` reads, for one check on the department column. It writes `results/nb_report.txt`, `results/dt_report.txt` and its charts to `figures/`. It uses two modules:

- `src/text_nb.py` — Naive Bayes on the posting text for three role families (forward deployed, solutions and software engineering), run on the raw descriptions, with company boilerplate removed, and with each posting's title and role phrases also masked, on one shared training and test split.
- `src/role_dt.py` — decision trees on the same postings and split, with four feature sets: all tabular columns, department removed, department and company removed, and the masked description text.

The Module 3 sample tables and explanatory figures on the site come from two more scripts, which can run before or after the notebook:

- `src/make_module3_samples.py` — writes samples of the Naive Bayes and decision tree inputs, and of each side of the training and test split, to `data/samples/` and draws them as PNG tables.
- `src/make_module3_concept_figures.py` — the two explanatory figures on each of the Naive Bayes and Decision Trees pages, drawn from small made-up data.

## What is not in the repository

`data/raw/` and `data/interim/` are gitignored, along with three processed files that are too large to track: `hn_postings.csv` (125 MB), `hn_clean.csv` (177 MB) and `indeed_metro.csv` (112 MB). The fetch and clean scripts regenerate all of them. Small raw and cleaned samples of every source are committed in `data/samples/`, so the shape of the data can be seen without downloading anything. Among them, `ats_raw_sample.csv` and `indeed_sector_raw_sample.csv` are the 20-row raw samples linked from the Clustering, PCA, Naive Bayes and Decision Trees pages. `clustering_input_sample.csv` and `pca_input_sample.csv` show ten rows of the matrices the clustering and PCA are run on, and `clustering_before_sample.csv` and `pca_before_sample.csv` show the same rows before TF-IDF and standardising. `nb_input_sample.csv`, `nb_train_sample.csv` and `nb_test_sample.csv` show ten rows each of the Naive Bayes input and of its training and test sets, and `dt_input_sample.csv`, `dt_train_sample.csv` and `dt_test_sample.csv` do the same for the decision trees.

## Course

CSCI 5612 Machine Learning for Data Science, CU Boulder, Fall 2026.
