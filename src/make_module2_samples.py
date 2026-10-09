"""Write samples of the data fed to clustering and PCA, and draw them as PNG tables.

Clustering input: the TF-IDF matrix used for k-means and hierarchical clustering,
unlabeled and numeric only. Ten postings drawn at random, and the twelve terms that
appear in the most postings, so the sample is not almost all zeros.

PCA input: the standardised Indeed sector matrix (sectors x months). The three
AI-exposed sectors plus seven control sectors drawn at random, over the twelve months
of 2022.

Output: data/samples/clustering_input_sample.csv, data/samples/pca_input_sample.csv,
and the matching PNG tables in figures/samples/.
"""

import numpy as np
import pandas as pd

import sector_pca as sp
import text_clusters as tc
from make_sample_images import SAMPLES_DIR, render
from plotstyle import apply_style

# "stripped" = company boilerplate removed; "raw" = description_clean as-is
VARIANT = "stripped"
N_ROWS, N_COLS = 10, 12
PCA_MONTHS = ("2022-01", "2022-12")


def clustering_sample() -> tuple[pd.DataFrame, int, int]:
    """10 random postings x the 12 most widespread terms of the TF-IDF matrix."""
    postings, _ = tc.load_postings()
    text = postings.description_clean if VARIANT == "raw" else tc.strip_boilerplate(postings)[0]
    X, vectorizer = tc.build_tfidf(text)
    terms = vectorizer.get_feature_names_out()
    cols = np.argsort(X.getnnz(axis=0))[::-1][:N_COLS]
    rows = np.sort(np.random.default_rng(tc.RANDOM_STATE).choice(X.shape[0], N_ROWS, replace=False))
    sample = pd.DataFrame(X[rows][:, cols].toarray(), columns=terms[cols]).round(4)
    return sample, X.shape[0], X.shape[1]


def pca_sample() -> tuple[pd.DataFrame, int, int]:
    """The 3 AI-exposed sectors and 7 random control sectors over the months of 2022."""
    _, z, groups = sp.sector_matrix(pd.read_csv(sp.SECTOR_CSV), "total postings")
    exposed = groups[groups == "ai_exposed"].index.tolist()
    control = groups[groups == "control"].sample(N_ROWS - len(exposed),
                                                 random_state=tc.RANDOM_STATE).index.tolist()
    sample = z.loc[sorted(exposed + control), PCA_MONTHS[0]:PCA_MONTHS[1]]
    sample.columns = sample.columns.strftime("%Y-%m")
    sample.index.name = "sector"
    return sample.round(3), z.shape[0], z.shape[1]


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


if __name__ == "__main__":
    main()
