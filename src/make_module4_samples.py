"""Write samples of the data fed to the Module 4 models, and draw them as PNG tables.

Regression: ten postings that carry a salary, as the six features before encoding, with the salary
midpoint the model predicts and where that salary came from (the structured band or the
description text). Input: ten of the subset's postings with a salary; train and test: ten rows from
each side of the Module 3 split that carry a salary. Encoded: the same ten rows as the input sample, after
the all-features fit's own encoder (the features step of the pipeline regression_models.fit_linear fits on
the training rows), with every column of the design matrix in the CSV and a readable subset in the PNG.

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

Output: data/samples/reg_input_sample.csv, reg_train_sample.csv, reg_test_sample.csv, reg_encoded_sample.csv,
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


REG_ENCODED_BLOCKS = {"role_family": "role", "workplaceType": "work mode", "company_clean": "company"}  # shown in the PNG
TAB_REFERENCE = {"company_clean": "OpenAI", "department": "Engineering", "location_clean": "San Francisco, CA",
                 "workplaceType": "missing", "role_family": "software_eng"}  # the reference levels the Regression tab names


def regression_encoded(subset: pd.DataFrame, input_sample: pd.DataFrame) -> dict:
    """The regression input sample, same rows in the same order, after the all-features fit's own encoder.

    The encoder is the features step of the "all features" pipeline that regression_models.fit_linear fits on
    the training rows that carry a salary, with regression_models.reference_levels as the dropped levels, so the
    columns are the model's design matrix, named as in the coefficient table of results/regression_report.txt.
    """
    df = rm.salary_frame(subset)
    tr, te = rm.linear_rows(df)
    refs = rm.reference_levels(tr, rm.CATEGORICAL)
    fit = rm.fit_linear(tr, te, refs)
    model = fit["models"]["all features"]
    features = model.named_steps["features"]
    names = [n.split("__", 1)[1] for n in features.get_feature_names_out()]
    Z = features.transform(input_sample[rm.CATEGORICAL + rm.NUMERIC])
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    assert np.allclose(Z, np.round(Z)), "a design value is not a whole number"
    design = pd.DataFrame(np.round(Z).astype(int), columns=names)
    table = pd.concat([input_sample[["jobId"]].reset_index(drop=True), design,
                       input_sample[["salary_mid"]].reset_index(drop=True)], axis=1)

    def block(col: str) -> list[str]:
        return [c for c in names if c.startswith(col + "_")]

    shown = ["jobId", *rm.NUMERIC, *block("role_family"), *block("workplaceType"),
             *[c for c in block("company_clean") if design[c].sum() > 0], "salary_mid"]
    n_design = len(shown) - 2
    title = (f"Regression input, one-hot encoded: the same {N_ROWS} postings; the {len(rm.CATEGORICAL + rm.NUMERIC)} "
             f"features become {len(names)} columns ({n_design} shown), encoder fit on the {len(tr):,} training rows")
    return {"table": table, "shown": shown, "title": title, "names": names, "model": model, "refs": refs,
            "scores": fit["table"], "input": input_sample, "n_train": len(tr)}


def regression_encoded_checks(enc: dict) -> None:
    """Print the checks on the encoded sample: rows, column count, one 1 per block, and the dropped levels."""
    table, names, sample = enc["table"], enc["names"], enc["input"]
    on_disk = pd.read_csv(SAMPLES_DIR / "reg_input_sample.csv", dtype={"jobId": str})["jobId"].tolist()
    ids = table["jobId"].tolist()
    print(f"check a. jobIds equal reg_input_sample.csv, same order: {ids == on_disk} "
          f"({len(ids)} rows; first {ids[0]}, last {ids[-1]})")
    n_coef = len(enc["model"].named_steps["ols"].coef_)
    print(f"check b. design columns {len(names)}; coefficients in the fitted all-features model {n_coef}; "
          f"equal: {len(names) == n_coef}")
    encoder = enc["model"].named_steps["features"].named_transformers_["cat"][-1]
    seen = dict(zip(rm.CATEGORICAL, encoder.categories_))
    print("check c. one 1 per category block, or all zeros at the block's reference level:")
    for col in rm.CATEGORICAL:
        cols = [c for c in names if c.startswith(col + "_")]
        values = sample[col].fillna("missing").tolist()
        at_ref, unseen, bad = [], [], []
        for k, (value, total) in enumerate(zip(values, table[cols].sum(axis=1)), 1):
            hit = table.loc[k - 1, f"{col}_{value}"] if f"{col}_{value}" in cols else 0
            if total == 1 and hit == 1:
                continue
            if total == 0 and value == enc["refs"][col]:
                at_ref.append(f"row {k} ({value})")
            elif total == 0 and value not in seen[col]:
                unseen.append(f"row {k} ({value})")
            else:
                bad.append(f"row {k} ({value}, sum {total})")
        print(f"   {col}: {len(cols)} columns; rows at the reference level: {', '.join(at_ref) or 'none'}"
              + (f"; unseen in training, encoded as the reference: {', '.join(unseen)}" if unseen else "")
              + f"; rows breaking the rule: {', '.join(bad) or 'none'}")
    print("check d. dropped (reference) level per block, as the encoder reports it, against the tab:")
    for col, cats, idx in zip(rm.CATEGORICAL, encoder.categories_, encoder.drop_idx_):
        print(f"   {col}: dropped {cats[idx]!r}; the tab names {TAB_REFERENCE[col]!r}; match: {cats[idx] == TAB_REFERENCE[col]}")
    row = enc["scores"].set_index("fit").loc["all features"]
    print(f"   the encoder's model, scored on the test rows: RMSE {row['RMSE']:,.0f}, MAE {row['MAE']:,.0f}, "
          f"R2 {row['R2']:.3f}; training rows {enc['n_train']}")


def write_regression_encoded(subset: pd.DataFrame, input_sample: pd.DataFrame, id_chars: int) -> None:
    enc = regression_encoded(subset, input_sample)
    enc["table"].to_csv(SAMPLES_DIR / "reg_encoded_sample.csv", index=False)
    shown = enc["table"][enc["shown"]].astype(object)
    shown["jobId"] = [truncate(v, id_chars + 1) for v in shown["jobId"]]
    for c in [*rm.NUMERIC, "salary_mid"]:
        shown[c] = enc["table"][c].map(lambda v: f"{v:,.0f}")
    headers = {"salary_mid": "salary_mid (target)"}
    for col, short in REG_ENCODED_BLOCKS.items():
        headers |= {c: f"{short}={c[len(col) + 1:]}" for c in enc["shown"] if c.startswith(col + "_")}
    shown = shown.rename(columns=headers)
    print("saved", render(shown, "reg_encoded_sample", enc["title"], list(shown.columns), max_rows=N_ROWS,
                          max_cols=len(shown.columns)))
    regression_encoded_checks(enc)


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
    reg_ids = m3.id_chars(pd.concat([t["jobId"] for t, _ in reg.values()]))  # same cut as the regression tables
    write_regression_encoded(subset, reg["reg_input_sample"][0], reg_ids)
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
