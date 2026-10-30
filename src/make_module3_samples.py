"""Write samples of the data fed to Naive Bayes, and draw them as PNG tables.

Input sample: ten of the 1,159 forward_deployed / solutions / software_eng postings as
the masked text the models read (boilerplate removed, title and role phrases masked),
with the role_family label.

Train and test samples: ten rows from each side of the stratified 80/20 split, as TF-IDF
values (vectoriser fitted on the training rows only) for the eight terms that appear in
the most training postings, with jobId and the label. The two sides share no jobId.

Decision tree samples: the same postings' tabular feature columns as the trees receive them
(before one-hot encoding and imputation; suspect salaries already blanked), with jobId and
the label. Input: ten of the 1,159 postings; train and test: ten rows from each side of
the same split. The PNG tables show each jobId cut to its first 8 characters (more if two
would look the same); the CSVs keep the full identifier.

Output: data/samples/nb_input_sample.csv, data/samples/nb_train_sample.csv,
data/samples/nb_test_sample.csv, data/samples/dt_input_sample.csv,
data/samples/dt_train_sample.csv, data/samples/dt_test_sample.csv, and the matching PNG
tables in figures/samples/.
"""

import numpy as np
import pandas as pd

import role_dt as dt
import text_clusters as tc
import text_nb as nb
from make_sample_images import SAMPLES_DIR, render, truncate
from plotstyle import apply_style

N_ROWS, N_COLS = 10, 8
DESCRIPTION_CHARS = 120
JOB_ID_CHARS = 8  # jobId characters shown in the decision-tree PNG tables
NOT_IN = "none are in the {} set"  # train/test titles name the other set; main() asserts it
ID_COLS = ["jobId", "role_family"]


def sample_positions(n: int) -> np.ndarray:
    """Sorted positions of N_ROWS rows drawn at random from n."""
    return np.sort(np.random.default_rng(tc.RANDOM_STATE).choice(n, N_ROWS, replace=False))


def input_sample(df: pd.DataFrame, masked: pd.Series) -> pd.DataFrame:
    """10 postings as masked text, with company and label."""
    rows = sample_positions(len(df))
    sample = df.iloc[rows][["jobId", "company_clean", "role_family"]].reset_index(drop=True)
    sample["masked_description"] = masked.iloc[rows].map(
        lambda t: truncate(t, DESCRIPTION_CHARS)).values
    return sample


def split_samples(df: pd.DataFrame, masked: pd.Series, train: np.ndarray,
                  test: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """10 training and 10 test rows x the 8 most widespread training terms."""
    X_train, X_test, vec = nb.vectorize(masked, train, test)
    terms = vec.get_feature_names_out()
    cols = np.argsort(X_train.getnnz(axis=0))[::-1][:N_COLS]

    def table(X, positions: np.ndarray) -> pd.DataFrame:
        rows = sample_positions(len(positions))
        out = pd.DataFrame(X[rows][:, cols].toarray(), columns=terms[cols]).round(4)
        out.insert(0, "role_family", df["role_family"].iloc[positions[rows]].values)
        out.insert(0, "jobId", df["jobId"].iloc[positions[rows]].values)
        return out

    return table(X_train, train), table(X_test, test), len(terms)


def tabular_sample(df: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    """10 rows (drawn from positions) of the decision-tree tabular features, with jobId and label."""
    rows = positions[sample_positions(len(positions))]
    out = dt.tabular_frame(df).iloc[rows].reset_index(drop=True)
    out.insert(0, "role_family", df["role_family"].iloc[rows].values)
    out.insert(0, "jobId", df["jobId"].iloc[rows].values)
    return out


def tabular_formatted(sample: pd.DataFrame) -> pd.DataFrame:
    """Whole numbers without decimals; missing values shown as NaN."""
    out = sample.astype(object)
    for c in dt.NUMERIC:
        out[c] = sample[c].map(lambda v: "NaN" if pd.isna(v) else f"{v:,.0f}")
    return out.fillna("NaN")


def id_chars(ids: pd.Series) -> int:
    """Fewest characters (at least JOB_ID_CHARS) that keep every displayed jobId distinct."""
    n = JOB_ID_CHARS
    while len({truncate(v, n + 1) for v in ids}) < ids.nunique():
        n += 1
    return n


def formatted(sample: pd.DataFrame) -> pd.DataFrame:
    """TF-IDF columns to three decimals; jobId and label as they are."""
    out = sample.copy()
    for c in out.columns.difference(ID_COLS):
        out[c] = out[c].map(lambda v: f"{v:.3f}")
    return out


def main() -> None:
    apply_style()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    df, _ = nb.load_subset()
    masked = nb.build_variants(df)["masked"]
    train, test = nb.split_rows(df["role_family"])
    overlap = set(df["jobId"].iloc[train]) & set(df["jobId"].iloc[test])
    print(f"train {len(train)} rows, test {len(test)} rows, jobId overlap: {len(overlap)}")
    assert not overlap, "train and test share a jobId"

    sample = input_sample(df, masked)
    sample.to_csv(SAMPLES_DIR / "nb_input_sample.csv", index=False)
    columns = ["jobId", "company_clean", "role_family", "masked description (first 120 characters)"]
    print("saved", render(sample.rename(columns={"masked_description": columns[-1]}),
                          "nb_input_sample",
                          f"Naive Bayes input: {N_ROWS} of {len(df):,} postings, boilerplate removed "
                          "and role phrases masked, with the role_family label",
                          columns, max_rows=N_ROWS, max_cols=len(columns),
                          max_chars=DESCRIPTION_CHARS, wrap_chars=DESCRIPTION_CHARS // 2 + 6))

    train_sample, test_sample, n_terms = split_samples(df, masked, train, test)
    print(f"TF-IDF vocabulary: {n_terms:,} terms (fitted on training rows only); tables show {N_COLS}")
    for name, part, rows, side, other in [("nb_train_sample", train_sample, train, "training", "test"),
                                          ("nb_test_sample", test_sample, test, "test", "training")]:
        part.to_csv(SAMPLES_DIR / f"{name}.csv", index=False)
        print("saved", render(formatted(part), name,
                              f"Naive Bayes {side} set: {N_ROWS} of {len(rows):,} rows, "
                              f"TF-IDF of the masked text; {NOT_IN.format(other)}",
                              list(part.columns), max_rows=N_ROWS, max_cols=len(part.columns)))
    shown = set(train_sample["jobId"]) & set(test_sample["jobId"])
    print(f"jobId overlap between the two sample tables: {len(shown)}")

    columns = ["jobId", "role_family", *dt.TABULAR]
    tables = {}
    for name, positions, title in [
            ("dt_input_sample", np.arange(len(df)),
             (f"Decision tree input: {N_ROWS} of {len(df):,} postings, tabular features before encoding, "
              "with the role_family label")),
            ("dt_train_sample", train,
             (f"Decision tree training set: {N_ROWS} of {len(train):,} rows, tabular features "
              f"before encoding; {NOT_IN.format('test')}")),
            ("dt_test_sample", test,
             (f"Decision tree test set: {N_ROWS} of {len(test):,} rows, tabular features "
              f"before encoding; {NOT_IN.format('training')}"))]:
        part = tabular_sample(df, positions)
        part.to_csv(SAMPLES_DIR / f"{name}.csv", index=False)
        tables[name] = (tabular_formatted(part), title)
    # the PNGs show a shortened jobId; checked across all three tables, so no two look the same
    n = id_chars(pd.concat([t["jobId"] for t, _ in tables.values()]))
    print(f"decision-tree tables: jobId shown to {n} characters"
          + ("" if n == JOB_ID_CHARS else f" ({JOB_ID_CHARS} would make two look identical)"))
    for name, (table, title) in tables.items():
        table["jobId"] = table["jobId"].map(lambda v: truncate(v, n + 1))
        print("saved", render(table, name, title, columns, max_rows=N_ROWS, max_cols=len(columns)))


if __name__ == "__main__":
    main()
