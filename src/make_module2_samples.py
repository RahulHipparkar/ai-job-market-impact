"""Write samples of the data fed to clustering and PCA, and draw them as PNG tables.

Clustering input: the TF-IDF matrix used for k-means and hierarchical clustering,
unlabeled and numeric only. Ten postings drawn at random, and the twelve terms that
appear in the most postings, so the sample is not almost all zeros.

PCA input: the standardised Indeed sector matrix (sectors x months). The three
AI-exposed sectors plus seven control sectors drawn at random, over the twelve months
of 2022.

Each input also gets a "before" table of the same rows: the postings as text before
TF-IDF, and the same sectors and months as raw index values before standardising.

Output: data/samples/clustering_input_sample.csv, data/samples/pca_input_sample.csv,
data/samples/clustering_before_sample.csv, data/samples/pca_before_sample.csv, and the
matching PNG tables in figures/samples/.
"""

import numpy as np
import pandas as pd

import sector_pca as sp
import text_clusters as tc
from make_sample_images import SAMPLES_DIR, render, truncate
from plotstyle import apply_style

# "stripped" = company boilerplate removed; "raw" = description_clean as-is
VARIANT = "stripped"
N_ROWS, N_COLS = 10, 12
PCA_MONTHS = ("2022-01", "2022-12")
BEFORE_COLS = ["company_clean", "title", "role_family", "description_clean"]
DESCRIPTION_CHARS = 120


def posting_rows(n_postings: int) -> np.ndarray:
    """Positions of the sampled postings, shared by the before and after tables."""
    return np.sort(np.random.default_rng(tc.RANDOM_STATE).choice(n_postings, N_ROWS, replace=False))


def clustering_sample() -> tuple[pd.DataFrame, int, int]:
    """10 random postings x the 12 most widespread terms of the TF-IDF matrix."""
    postings, _ = tc.load_postings()
    text = postings.description_clean if VARIANT == "raw" else tc.strip_boilerplate(postings)[0]
    X, vectorizer = tc.build_tfidf(text)
    terms = vectorizer.get_feature_names_out()
    cols = np.argsort(X.getnnz(axis=0))[::-1][:N_COLS]
    rows = posting_rows(X.shape[0])
    sample = pd.DataFrame(X[rows][:, cols].toarray(), columns=terms[cols]).round(4)
    return sample, X.shape[0], X.shape[1]


def clustering_before_sample() -> pd.DataFrame:
    """The same 10 postings as text, before boilerplate removal and TF-IDF."""
    postings, _ = tc.load_postings()
    sample = postings.iloc[posting_rows(len(postings))][BEFORE_COLS].reset_index(drop=True)
    sample["description_clean"] = sample["description_clean"].map(
        lambda t: truncate(t, DESCRIPTION_CHARS))
    return sample


def pca_sectors(groups: pd.Series) -> list[str]:
    """The 3 AI-exposed sectors and 7 random control sectors, shared by both PCA tables."""
    exposed = groups[groups == "ai_exposed"].index.tolist()
    control = groups[groups == "control"].sample(N_ROWS - len(exposed),
                                                 random_state=tc.RANDOM_STATE).index.tolist()
    return sorted(exposed + control)


def pca_sample(standardised: bool = True) -> tuple[pd.DataFrame, int, int]:
    """The sampled sectors over the months of 2022, standardised or as raw index values."""
    raw, z, groups = sp.sector_matrix(pd.read_csv(sp.SECTOR_CSV), "total postings")
    matrix = z if standardised else raw
    sample = matrix.loc[pca_sectors(groups), PCA_MONTHS[0]:PCA_MONTHS[1]]
    sample.columns = sample.columns.strftime("%Y-%m")
    sample.index.name = "sector"
    return sample.round(3 if standardised else 1), z.shape[0], z.shape[1]


def main() -> None:
    apply_style()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    tfidf, n_docs, n_terms = clustering_sample()
    print(f"clustering sample: {N_ROWS} of {n_docs:,} postings, {N_COLS} of {n_terms:,} terms")
    tfidf.to_csv(SAMPLES_DIR / "clustering_input_sample.csv", index=False)
    stage = "company boilerplate removed" if VARIANT == "stripped" else "raw descriptions"
    print("saved", render(tfidf.map(lambda v: f"{v:.3f}"), "clustering_input_sample",
                          f"ATS job postings: TF-IDF input to clustering ({stage})",
                          list(tfidf.columns), max_rows=N_ROWS, max_cols=N_COLS))

    sectors, n_sectors, n_months = pca_sample()
    sectors.to_csv(SAMPLES_DIR / "pca_input_sample.csv")
    table = sectors.map(lambda v: f"{v:.2f}").reset_index()
    print("saved", render(table, "pca_input_sample",
                          f"Indeed Hiring Lab: standardised sector matrix, input to PCA "
                          f"({N_ROWS} of {n_sectors} sectors, {N_COLS} of {n_months} months)",
                          list(table.columns), max_rows=N_ROWS, max_cols=N_COLS + 1))

    before = clustering_before_sample()
    before.to_csv(SAMPLES_DIR / "clustering_before_sample.csv", index=False)
    print("saved", render(before.rename(columns={"description_clean": "description (first 120 characters)"}),
                          "clustering_before_sample",
                          f"ATS job postings: the same {N_ROWS} postings as text, before TF-IDF",
                          ["company_clean", "title", "role_family", "description (first 120 characters)"],
                          max_rows=N_ROWS, max_chars=DESCRIPTION_CHARS,
                          wrap_chars=DESCRIPTION_CHARS // 2 + 6))

    raw_sectors, _, _ = pca_sample(standardised=False)
    raw_sectors.to_csv(SAMPLES_DIR / "pca_before_sample.csv")
    raw_table = raw_sectors.map(lambda v: f"{v:.1f}").reset_index()
    print("saved", render(raw_table, "pca_before_sample",
                          "Indeed Hiring Lab: the same sectors and months as raw index values "
                          "(1 Feb 2020 = 100), before standardising",
                          list(raw_table.columns), max_rows=N_ROWS, max_cols=N_COLS + 1))


if __name__ == "__main__":
    main()
