"""Write samples of the data fed to the Module 4 models, and draw them as PNG tables.

Regression: ten postings that carry a salary, as the six features before encoding, with the salary
midpoint the model predicts and where that salary came from (the structured band or the
description text). Input: ten of the subset's postings with a salary; train and test: ten rows from
each side of the Module 3 split that carry a salary.

Logistic regression: the same ten postings as Module 3's nb_input_sample, with role_family, the binary
forward_deployed (1) vs rest (0) label and the eight masked TF-IDF columns the Module 3 Naive Bayes
tables show; the vectoriser is fitted on the training rows only.

AdaBoost: the input table shows the one-hot encoding step itself, two categorical columns beside
the 0/1 columns role_dt's tree-1 encoder turns them into (fitted on the training rows). The train
and test tables show rows as the stumps receive them, after encoding and median imputation: the
numeric columns, the missing-salary indicator, and the four one-hot columns that are 1 in the most
training rows.

The PNG tables show each jobId cut to its first 8 characters (more if two would look the same);
the CSVs keep the full identifier. Train and test samples never share a jobId.

Output: data/samples/reg_input_sample.csv, reg_train_sample.csv, reg_test_sample.csv,
reg_logistic_sample.csv, ab_input_sample.csv, ab_train_sample.csv, ab_test_sample.csv, and the matching PNG tables in
figures/samples/.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import make_module3_samples as m3
import regression_models as rm
import role_dt as dt
import text_nb as nb
from make_sample_images import SAMPLES_DIR, render, truncate
from plotstyle import apply_style

N_ROWS = m3.N_ROWS
REG_COLUMNS = ["jobId", *rm.CATEGORICAL, *rm.NUMERIC, "salary_mid", "salary_source"]
ONE_HOT_SOURCES = ["workplaceType", "company_clean"]  # shown beside their 0/1 columns
N_FREQUENT = 4  # one-hot columns shown in the AdaBoost train and test tables


def pick(frame: pd.DataFrame) -> pd.DataFrame:
    """N_ROWS rows drawn by Module 3's sampler (same seed)."""
    return frame.iloc[m3.sample_positions(len(frame))].reset_index(drop=True)


def regression_tables(subset: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
    df = rm.salary_frame(subset)
    tr, te = rm.linear_rows(df)
    salaried = df[df["salary_source"] != "none"]
    return {
        "reg_input_sample": (pick(salaried)[REG_COLUMNS],
                             (f"Regression input: {N_ROWS} of the {len(salaried):,} postings with a salary "
                              f"(of {len(df):,}), before encoding, with the salary target")),
        "reg_train_sample": (pick(tr)[REG_COLUMNS],
                             (f"Regression training set: {N_ROWS} of the {len(tr):,} rows with a salary (of 927); "
                              f"{m3.NOT_IN.format('test')}")),
        "reg_test_sample": (pick(te)[REG_COLUMNS],
                            (f"Regression test set: {N_ROWS} of the {len(te):,} rows with a salary (of 232); "
                             f"{m3.NOT_IN.format('training')}")),
    }


def regression_formatted(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.astype(object)
    out["desc_len"] = sample["desc_len"].map(lambda v: f"{v:,.0f}")
    out["salary_mid"] = sample["salary_mid"].map(lambda v: f"{v:,.0f}")
    return out.fillna("NaN").rename(columns={"salary_mid": "salary_mid (target)"})


def encoded_frame(X: pd.DataFrame, encoder: Pipeline) -> pd.DataFrame:
    """Rows as the stumps receive them, with readable column names."""
    Z = encoder.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    return pd.DataFrame(Z, columns=dt.feature_names(encoder), index=X.index)


def adaboost_tables(subset: pd.DataFrame) -> tuple[dict[str, tuple[pd.DataFrame, str]], int]:
    train, test = nb.split_rows(subset["role_family"])
    X = dt.tabular_frame(subset)
    encoder = Pipeline([("features", dt.tabular_encoder(dt.TREES["tree1"][1]))]).fit(X.iloc[train])
    Z = encoded_frame(X, encoder)
    n_columns = Z.shape[1]

    rows = m3.sample_positions(len(subset))
    before = pd.DataFrame({"jobId": subset["jobId"].iloc[rows].values})
    for col in ONE_HOT_SOURCES:
        before[col] = X[col].iloc[rows].fillna("missing").values
        shown = [c for c in Z.columns if c.startswith(col + " = ")]
        if col != "workplaceType":  # every work-mode column; for company only those the ten rows use
            shown = [c for c in shown if Z[c].iloc[rows].sum() > 0]
        for c in shown:
            before[c] = Z[c].iloc[rows].astype(int).values
    n_company = sum(c.startswith("company_clean = ") for c in Z.columns)
    n_shown = sum(c.startswith("company_clean = ") for c in before.columns)
    title_in = (f"AdaBoost input, one-hot encoded: {N_ROWS} of {len(subset):,} postings; tree 1's 9 columns become "
                f"{n_columns} ({n_shown} of {n_company} company columns shown)")

    numeric = [c for c in Z.columns if " = " not in c]
    frequent = Z.iloc[train][[c for c in Z.columns if " = " in c]].sum().sort_values(ascending=False)
    shown = numeric + list(frequent.index[:N_FREQUENT])

    def side(positions: np.ndarray) -> pd.DataFrame:
        r = positions[m3.sample_positions(len(positions))]
        out = Z[shown].iloc[r].reset_index(drop=True)
        out.insert(0, "role_family", subset["role_family"].iloc[r].values)
        out.insert(0, "jobId", subset["jobId"].iloc[r].values)
        return out

    tables = {"ab_input_sample": (before, title_in)}
    for name, positions, label, other in [("ab_train_sample", train, "training set", "test"),
                                          ("ab_test_sample", test, "test set", "training")]:
        tables[name] = (side(positions),
                        (f"AdaBoost {label}: {N_ROWS} of {len(positions):,} rows, encoded and median-imputed "
                         f"({len(shown)} of {n_columns} columns shown); {m3.NOT_IN.format(other)}"))
    return tables, n_columns


SHORT_NAMES = {"salary_annual_min": "salary_min", "salary_annual_max": "salary_max",
               "is_multi_location": "multi_location", "salary missing": "salary_missing"}
SHORT_PREFIX = {"company_clean": "company", "location_clean": "location", "workplaceType": "work mode",
                "department": "dept"}


def display_name(column: str, beside_source: bool) -> str:
    """A shorter header for the PNG: "=Databricks" beside its source column, "company=Databricks" elsewhere."""
    if " = " not in column:
        return SHORT_NAMES.get(column, column)
    source, value = column.split(" = ", 1)
    return f"={value}" if beside_source else f"{SHORT_PREFIX.get(source, source)}={value}"


def adaboost_formatted(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.astype(object)
    for c in sample.columns:
        if c in ("jobId", "role_family", *ONE_HOT_SOURCES):
            continue
        out[c] = sample[c].map(lambda v: f"{v:,.0f}")
    beside = "company_clean" in sample.columns  # the before-and-after table
    return out.rename(columns={c: display_name(c, beside) for c in sample.columns})


def main() -> None:
    apply_style()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    subset, _ = nb.load_subset()
    train, test = nb.split_rows(subset["role_family"])
    assert not set(subset["jobId"].iloc[train]) & set(subset["jobId"].iloc[test]), "train and test share a jobId"

    reg = regression_tables(subset)
    ab, n_columns = adaboost_tables(subset)
    print(f"tree-1 encoder: {n_columns} columns after one-hot encoding (fitted on the training rows only)")
    for family, tables, fmt in [("regression", reg, regression_formatted), ("AdaBoost", ab, adaboost_formatted)]:
        ids = pd.concat([t["jobId"] for t, _ in tables.values()])
        n = m3.id_chars(ids)
        train_ids, test_ids = (set(tables[k][0]["jobId"]) for k in tables
                               if k.endswith(("train_sample", "test_sample")))
        assert not train_ids & test_ids, f"{family} train and test samples share a jobId"
        print(f"{family} tables: jobId shown to {n} characters")
        for name, (table, title) in tables.items():
            table.to_csv(SAMPLES_DIR / f"{name}.csv", index=False)
            shown = fmt(table)
            shown["jobId"] = [truncate(v, n + 1) for v in shown["jobId"]]
            print("saved", render(shown, name, title, list(shown.columns), max_rows=N_ROWS,
                                  max_cols=len(shown.columns)))
    write_logistic(subset)


LOGISTIC_LABEL = "forward_deployed vs rest (1/0)"


def logistic_table(subset: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Ten postings as the logistic model receives them: role_family, the binary label and the eight masked TF-IDF columns
    the Module 3 Naive Bayes tables show (the words in the most training postings). The vectoriser is fitted on
    the training rows only; the ten postings are the ones nb_input_sample shows (Module 3's sampler)."""
    texts, y, train, test = rm.binary_data(subset)
    X_train, _, vec = nb.vectorize(texts, train, test)
    terms = vec.get_feature_names_out()
    cols = np.argsort(X_train.getnnz(axis=0))[::-1][:m3.N_COLS]
    rows = m3.sample_positions(len(subset))
    out = pd.DataFrame(vec.transform(texts.iloc[rows])[:, cols].toarray(), columns=terms[cols]).round(4)
    out.insert(0, LOGISTIC_LABEL, y.iloc[rows].values)
    out.insert(0, "role_family", subset["role_family"].iloc[rows].values)
    out.insert(0, "jobId", subset["jobId"].iloc[rows].values)
    title = (f"Logistic input: {N_ROWS} of {len(subset):,} postings, {int(y.sum())} labelled 1; {m3.N_COLS} of "
             f"{len(terms):,} TF-IDF columns, vectorizer fit on training rows only")
    return out, title


def write_logistic(subset: pd.DataFrame) -> None:
    table, title = logistic_table(subset)
    table.to_csv(SAMPLES_DIR / "reg_logistic_sample.csv", index=False)
    shown = table.astype(object)
    for c in table.columns.difference(["jobId", "role_family", LOGISTIC_LABEL]):
        shown[c] = table[c].map(lambda v: f"{v:.3f}")
    n = m3.id_chars(table["jobId"])
    shown["jobId"] = [truncate(v, n + 1) for v in shown["jobId"]]
    print("saved", render(shown, "reg_logistic_sample", title, list(shown.columns), max_rows=N_ROWS,
                          max_cols=len(shown.columns)))


if __name__ == "__main__":
    main()
