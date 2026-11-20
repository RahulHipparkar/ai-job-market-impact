"""Module 4: linear regression on salary, and logistic regression on forward_deployed vs the rest.

Both use the Module 3 subset (1,159 postings) and its stratified 80/20 split (927 training,
232 test). Every encoder, imputer, vectoriser and model is fitted on training rows only, and the
test rows are scored once.

Linear regression. Target: the salary midpoint, from the structured band where it is usable (both
ends present, not suspect, USD) and otherwise from the pay range in the description text
(data/module4/pay_ranges.csv, text_usable_usd). Features: company, department, location, work
mode, role family and description length. The description text is never a feature, because the
target is partly taken from it; title, seniority and is_near_dup are never used. Categories are
one-hot encoded against a reference level, the most common training value of each column, so
every coefficient is a dollar difference from that level; a category not seen in training is
encoded as the reference.

Logistic regression. Label: forward_deployed vs the other two role families. Features: the
Module 3 masked TF-IDF. class_weight="balanced"; C is chosen by stratified 5-fold CV on the
training rows (the Module 3 folds), scored on F1 for the positive class, ties going to the
smaller C. Multinomial and Bernoulli Naive Bayes are refitted on the same binary label, same
features, same folds and the same CV score, alpha ties going to the larger alpha as in Module 3.

Output: results/regression_report.txt, and results/regression_predictions.csv with the test-set
predicted and actual salary of every fit.

usage: python src/regression_models.py
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, mannwhitneyu, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src import extract_pay_ranges as xp
    from src import text_clusters as tc
    from src import text_nb as nb
    from src.plotstyle import CMAP, PALETTE
except ImportError:
    import extract_pay_ranges as xp
    import text_clusters as tc
    import text_nb as nb
    from plotstyle import CMAP, PALETTE

REPORT_PATH = tc.ROOT / "results" / "regression_report.txt"
PRED_PATH = tc.ROOT / "results" / "regression_predictions.csv"

CATEGORICAL = ["company_clean", "department", "location_clean", "workplaceType", "role_family"]
NUMERIC = ["desc_len"]
FITS = {  # name: (categorical columns, numeric columns, drop multi-zone rows from training)
    "all features": (CATEGORICAL, NUMERIC, False),
    "without company and department": (["location_clean", "workplaceType", "role_family"], NUMERIC, False),
    "company only": (["company_clean"], [], False),
    "all features, multi-zone rows out of training": (CATEGORICAL, NUMERIC, True),
}
PRED_COLUMN = {"all features": "pred_all", "without company and department": "pred_no_company_dept",
               "company only": "pred_company_only",
               "all features, multi-zone rows out of training": "pred_all_no_multizone"}
DUP_KEY = CATEGORICAL + ["band_min", "band_max"]  # the Module 4 data check's definition
OOF_REPEATS = 10  # out-of-fold residuals are averaged over this many shuffled 5-fold splits
N_PERMUTATIONS = 10_000

POSITIVE = "forward_deployed"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
CV_SCORE = "f1"
N_WORDS = 20

# The manual check of 50 text-only extractions, run before any model was fitted.
EXTRACTION_AUDIT = [
    "50 text-only extractions were read against their surrounding text before modelling: 20 Databricks,",
    "20 Anthropic and 10 Scale AI, drawn with seed 7 from the 1,294 text-only postings not already checked",
    "in the validation round.",
    "  value, currency and interval match the text: 50 of 50",
    "  picks that quote a band for another location than the posting's own: 3, all multi_zone, all the",
    "  highest-zone rule (a Databricks intern in Washington, DC given the SF Bay Area hourly rate; a",
    "  Databricks 'Central - United States' posting given Zone 1; a Scale AI posting in Austin, TX given",
    "  the Washington DC/Hawaii band)",
    "  basis that cannot be checked from the text: 1 (a Databricks sales director, $430,400-$591,800,",
    "  may be on-target earnings; the text does not say). Anthropic's ranges sit under its note that",
    "  sales roles are quoted as OTE without marking which roles those are.",
    "  judged wrong: 0 (3 even counting the location picks), so modelling went ahead",
]


# ---------------------------------------------------------------- data

def salary_frame(subset: pd.DataFrame | None = None) -> pd.DataFrame:
    """The subset with its split and, where one is usable, the salary band and its source.

    subset is text_nb.load_subset()'s frame, when the caller has already loaded it.
    """
    df = nb.load_subset()[0] if subset is None else subset.copy()
    _train, test = nb.split_rows(df["role_family"])
    df["split"] = "train"
    df.loc[df.index[test], "split"] = "test"
    df["jobId"] = df["jobId"].astype(str)
    pr = pd.read_csv(xp.OUT_CSV, dtype={"jobId": str})
    df = df.merge(pr[["jobId", "text_usable_usd", "text_annual_min", "text_annual_max", "multi_zone",
                      "text_span", "interval_converted"]], on="jobId", how="left", validate="1:1")
    structured = (df["salary_annual_min"].notna() & df["salary_annual_max"].notna()
                  & ~df["salary_suspect"].eq(True) & df["salary_currency"].eq("USD"))
    text = df["text_usable_usd"].eq(True) & ~structured
    df["salary_source"] = np.select([structured, text], ["structured", "text"], "none")
    df["band_min"] = df["salary_annual_min"].where(structured, df["text_annual_min"].where(text))
    df["band_max"] = df["salary_annual_max"].where(structured, df["text_annual_max"].where(text))
    df["salary_mid"] = (df["band_min"] + df["band_max"]) / 2
    df["multi_zone_used"] = text & df["multi_zone"].eq(True)
    df[CATEGORICAL] = df[CATEGORICAL].astype(object)
    return df


# ---------------------------------------------------------------- linear regression

def reference_levels(X: pd.DataFrame, cats: list[str]) -> dict[str, str]:
    """The most common training value of each categorical column, with missing as "missing"."""
    return {c: X[c].fillna("missing").value_counts().idxmax() for c in cats}


class ReferenceEncoder(OneHotEncoder):
    """OneHotEncoder that does not warn about categories unseen in training.

    With a reference level dropped, such a category is encoded as all zeros, which is the reference
    level: the intended behaviour, counted once for the test rows by data_lines. Inside
    cross-validation it happens in most folds, and scikit-learn would warn every time.
    """

    def transform(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
            return super().transform(X)


def linear_model(cats: list[str], nums: list[str], refs: dict[str, str]) -> Pipeline:
    """role_dt-style encoding (constant "missing" for categories, median for numbers), with each
    categorical one-hot encoded against its reference level, then ordinary least squares."""
    parts = []
    if cats:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                           ReferenceEncoder(drop=[refs[c] for c in cats], handle_unknown="ignore")),
                      cats))
    if nums:
        parts.append(("num", SimpleImputer(strategy="median"), nums))
    return Pipeline([("features", ColumnTransformer(parts)), ("ols", LinearRegression())])


def regression_scores(y, pred) -> dict[str, float]:
    return {"RMSE": root_mean_squared_error(y, pred), "MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred)}


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    names = [n.split("__", 1)[1] for n in model[:-1].get_feature_names_out()]
    coef = pd.Series(model[-1].coef_, index=names)
    return (pd.DataFrame({"column": coef.index, "dollars": coef.values})
            .assign(size=lambda t: t["dollars"].abs()).sort_values("size", ascending=False).drop(columns="size"))


def linear_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test rows that carry a salary."""
    rows = df[df["salary_source"] != "none"]
    return rows[rows["split"] == "train"], rows[rows["split"] == "test"]


def data_lines(tr: pd.DataFrame, te: pd.DataFrame, refs: dict[str, str]) -> list[str]:
    lines = [f"rows with a salary: training {len(tr)} of 927, test {len(te)} of 232",
             (f"  source, training: {tr['salary_source'].value_counts().to_dict()}; test: "
             f"{te['salary_source'].value_counts().to_dict()}"),
             (f"  role family, training: {tr['role_family'].value_counts().to_dict()}; test: "
             f"{te['role_family'].value_counts().to_dict()}"),
             (f"  multi-zone rows (text salary picked among several zone ranges): training "
             f"{int(tr['multi_zone_used'].sum())}, test {int(te['multi_zone_used'].sum())}"),
             f"  target mean on training rows ${tr['salary_mid'].mean():,.0f}, sd ${tr['salary_mid'].std():,.0f}"]
    lines.append("reference levels (most common training value): " + "; ".join(f"{c} = {v}" for c, v in refs.items()))
    unseen = {c: int((~te[c].fillna("missing").isin(set(tr[c].fillna("missing")))).sum()) for c in CATEGORICAL}
    lines.append(f"test rows whose category never occurs in training (encoded as the reference): {unseen}")
    return lines


def fit_linear(tr: pd.DataFrame, te: pd.DataFrame, refs: dict[str, str]) -> dict:
    """The four fits and the training-mean baseline, each scored once on the test rows."""
    preds = te[["jobId", "company_clean", "role_family", "department", "location_clean", "salary_source",
                "multi_zone_used"]].rename(columns={"salary_source": "source", "multi_zone_used": "multi_zone"})
    preds["actual_mid"] = te["salary_mid"].values
    base = np.full(len(te), tr["salary_mid"].mean())
    table = [{"fit": "baseline: training mean", "train rows": len(tr), **regression_scores(te["salary_mid"], base)}]
    no_mz = ~te["multi_zone_used"].values
    side = [{"fit": "baseline: training mean", "R2": r2_score(te["salary_mid"][no_mz], base[no_mz])}]
    models = {}
    for name, (cats, nums, drop_mz) in FITS.items():
        fit_rows = tr[~tr["multi_zone_used"]] if drop_mz else tr
        model = linear_model(cats, nums, refs).fit(fit_rows[cats + nums], fit_rows["salary_mid"])
        pred = model.predict(te[cats + nums])
        preds[PRED_COLUMN[name]] = pred
        models[name] = model
        table.append({"fit": name, "train rows": len(fit_rows), **regression_scores(te["salary_mid"], pred)})
        side.append({"fit": name, "R2": r2_score(te["salary_mid"][no_mz], pred[no_mz])})
    return {"table": pd.DataFrame(table), "side": side, "preds": preds, "models": models,
            "n_not_multizone": int(no_mz.sum())}


def fit_lines(fit: dict) -> list[str]:
    return ["", f"TEST SET ({len(fit['preds'])} rows), each model fitted on training rows only:",
            fit["table"].to_string(index=False, formatters={"RMSE": "{:,.0f}".format, "MAE": "{:,.0f}".format,
                                                            "R2": "{:.3f}".format}),
            f"same fits scored on the {fit['n_not_multizone']} test rows that are not multi-zone: " +
            "; ".join(f"{r['fit']} R2 {r['R2']:.3f}" for r in fit["side"])]


def run_linear(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    tr, te = linear_rows(df)
    refs = reference_levels(tr, CATEGORICAL)
    fit = fit_linear(tr, te, refs)
    lines = data_lines(tr, te, refs) + fit_lines(fit)
    lines += multizone_lines(tr, te, fit["preds"])
    lines += ["", "MULTI-ZONE RESIDUAL CHECK"] + multizone_residual_lines(df)
    lines += coefficient_lines(fit["models"], tr)
    lines += ["", "DUPLICATE-BAND CAVEAT"] + duplicate_lines(tr, te, refs)
    return lines, fit["preds"]


def coefficient_lines(models: dict, tr: pd.DataFrame) -> list[str]:
    """Every all-features coefficient by size, then the company-only fit's."""
    lines = []
    full_model = models["all features"]
    X = full_model[:-1].transform(tr[CATEGORICAL + NUMERIC])
    X = X.toarray() if hasattr(X, "toarray") else X
    rank = np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X]))
    lines += ["", (f"ALL-FEATURES COEFFICIENTS, dollars of salary midpoint per unit; categories are differences "
              f"from the reference level above, desc_len is dollars per character "
              f"(x1,000 = {full_model[-1].coef_[-1] * 1000:,.0f} per 1,000 characters)."),
              (f"intercept {full_model[-1].intercept_:,.0f}; {X.shape[1]} encoded columns; rank of the training "
              f"design with intercept {rank} of {X.shape[1] + 1}")]
    if rank < X.shape[1] + 1:
        lines.append("  the design is rank-deficient: some departments belong to a single company, so their "
                     "department and company columns are not separately identified, and least squares splits the "
                     "effect between them. The role family, work mode, location and desc_len coefficients, and the "
                     "company-only fit's coefficients, do not have this problem.")
    lines.append(coefficient_table(full_model).to_string(index=False, formatters={"dollars": dollars}))
    lines += ["", "COMPANY-ONLY COEFFICIENTS (difference from the reference company's mean):",
              f"intercept {models['company only'][-1].intercept_:,.0f}",
              coefficient_table(models["company only"]).to_string(index=False, formatters={"dollars": dollars})]
    return lines


def dollars(value: float) -> str:
    """Whole dollars, except a per-unit coefficient under $10 (desc_len), which keeps two decimals."""
    return f"{value:,.2f}" if abs(value) < 10 else f"{value:,.0f}"


def multizone_lines(tr: pd.DataFrame, te: pd.DataFrame, preds: pd.DataFrame) -> list[str]:
    """Where the change from dropping multi-zone training rows comes from, by company."""
    def sse(col: str) -> pd.Series:
        return (preds["actual_mid"] - preds[col]) ** 2

    change = ((sse("pred_all_no_multizone") - sse("pred_all")).groupby(preds["company_clean"]).sum() / 1e9)
    mz = tr.groupby("company_clean").agg(rows=("jobId", "size"), multi_zone=("multi_zone_used", "sum"))
    mz = mz[mz["multi_zone"] > 0]
    out = [("where the multi-zone sensitivity comes from: training rows that are multi-zone, by company "
            f"(multi-zone / all): {', '.join(f'{c} {r.multi_zone}/{r.rows}' for c, r in mz.iterrows())}"),
           "  change in test squared error when they are left out, by company (billions of dollars squared; + is worse): "
           + ", ".join(f"{c} {v:+.1f}" for c, v in change.sort_values(ascending=False).items() if abs(v) >= 0.5)]
    dept = tr[tr["multi_zone_used"]].groupby(["company_clean", "department"]).size()
    whole = tr.groupby(["company_clean", "department"]).size()
    heavy = [(c, d, n, whole[(c, d)]) for (c, d), n in dept.items() if n / whole[(c, d)] >= 0.5 and whole[(c, d)] >= 3]
    out.append("  departments with at least 3 training rows, half or more of them multi-zone: "
               + "; ".join(f"{c} '{d}' {n}/{w}" for c, d, n, w in heavy))
    return out


def group_comparison(res: pd.Series, multi: pd.Series, label: str, seed: int = 0) -> str:
    """Residuals of multi-zone rows against the rest: counts, means, medians and three tests."""
    a, b = res[multi.to_numpy()], res[~multi.to_numpy()]
    if len(a) < 2 or len(b) < 2:
        return f"  {label}: too few rows (multi-zone {len(a)}, single-zone {len(b)})"
    diff = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    pooled, k = res.to_numpy(), len(a)
    perm = np.array([pooled[idx[:k]].mean() - pooled[idx[k:]].mean()
                     for idx in (rng.permutation(len(pooled)) for _ in range(N_PERMUTATIONS))])
    p_perm = float((np.abs(perm) >= abs(diff)).mean())
    return (f"  {label}: multi-zone n {len(a)}, mean {a.mean():+,.0f}, median {a.median():+,.0f} | single-zone "
            f"n {len(b)}, mean {b.mean():+,.0f}, median {b.median():+,.0f} | difference in means {diff:+,.0f}; "
            f"Welch t-test p = {ttest_ind(a, b, equal_var=False).pvalue:.3f}, Mann-Whitney p = "
            f"{mannwhitneyu(a, b).pvalue:.3f}, permutation p = {p_perm:.3f}")


def multizone_residual_lines(df: pd.DataFrame) -> list[str]:
    """Are multi-zone rows mispredicted in a consistent direction by the all-features fit?

    If the highest-zone rule overstates pay, a multi-zone row's recorded midpoint sits above what
    similar postings are paid, so its residual (actual minus predicted) is positive: the fit
    predicts less than the recorded value. The test set holds few multi-zone rows, so the same
    comparison is also made on out-of-fold residuals of the training rows, overall and among text
    salaries at the employers that have multi-zone rows.
    """
    rows = df[df["salary_source"] != "none"]
    tr, te = rows[rows["split"] == "train"], rows[rows["split"] == "test"]
    cols = CATEGORICAL + NUMERIC
    refs = reference_levels(tr, CATEGORICAL)
    model = linear_model(CATEGORICAL, NUMERIC, refs).fit(tr[cols], tr["salary_mid"])
    test_res = te["salary_mid"] - model.predict(te[cols])
    oof = np.mean([cross_val_predict(linear_model(CATEGORICAL, NUMERIC, refs), tr[cols], tr["salary_mid"],
                                     cv=KFold(5, shuffle=True, random_state=s)) for s in range(OOF_REPEATS)], axis=0)
    train_res = tr["salary_mid"] - oof
    mz_test, mz_train = te["multi_zone_used"], tr["multi_zone_used"]
    companies = sorted(tr.loc[mz_train, "company_clean"].unique())
    same = tr["company_clean"].isin(companies) & tr["salary_source"].eq("text")
    lines = [("residual = actual minus predicted midpoint, all-features fit; an overstated multi-zone salary shows up "
              "as a positive residual (the fit predicts less than the recorded top-zone value)"),
             group_comparison(test_res, mz_test, "test rows"),
             group_comparison(train_res, mz_train, f"training rows, out of fold ({OOF_REPEATS} x 5-fold)"),
             group_comparison(train_res[same], mz_train[same],
                              f"training rows, out of fold, text salaries at {', '.join(companies)} only")]
    for comp in companies:
        m = same & tr["company_clean"].eq(comp)
        lines.append(group_comparison(train_res[m], mz_train[m], f"  {comp} text salaries only"))
    lines.append("  multi-zone test rows: " + "; ".join(
        f"{r.company_clean}, {r.location_clean}: actual {r.salary_mid:,.0f}, residual {res:+,.0f}"
        for r, res in zip(te[mz_test].itertuples(), test_res[mz_test])))

    gaps = []
    for r in rows[rows["multi_zone_used"]].itertuples():
        cands = xp.candidates(r.description_clean, r.location_clean if isinstance(r.location_clean, str) else "")
        pool = {((c["text_annual_min"] + c["text_annual_max"]) / 2) for c in cands
                if not c["text_placeholder"] and not c["text_suspect"] and c["text_currency"] == "USD"
                and c["text_basis"] != "ote"}
        gaps.append((r.salary_mid - np.mean(sorted(pool)), r.salary_mid - min(pool), r.salary_mid))
    g = pd.DataFrame(gaps, columns=["vs_mean", "vs_lowest", "mid"])
    lines.append(f"how much the rule can add, on the {len(g)} multi-zone rows (picked midpoint minus the midpoints of "
                 f"the zones listed beside it): vs their average, mean {g.vs_mean.mean():+,.0f} "
                 f"({(g.vs_mean / g.mid).mean():.1%}), "
                 f"median {g.vs_mean.median():+,.0f}; vs the lowest zone mean {g.vs_lowest.mean():+,.0f} "
                 f"({(g.vs_lowest / g.mid).mean():.1%}). That is the most the rule can overstate a posting that "
                 f"belongs in a lower zone.")
    return lines


def duplicate_lines(tr: pd.DataFrame, te: pd.DataFrame, refs: dict[str, str]) -> list[str]:
    """R2 again after collapsing rows identical on every categorical feature and on the band."""
    cats, nums = CATEGORICAL, NUMERIC
    out = []
    for label, key in [("the five categorical features and the band", DUP_KEY),
                       ("those and desc_len", DUP_KEY + ["desc_len"])]:
        tr_c, te_c = tr.drop_duplicates(key), te.drop_duplicates(key)
        twins = te.merge(tr[key].drop_duplicates(), on=key, how="inner")
        cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=nb.SPLIT_SEED)
        cv_full = cross_val_score(linear_model(cats, nums, refs), tr[cats + nums], tr["salary_mid"], cv=cv,
                                  scoring="r2").mean()
        cv_coll = cross_val_score(linear_model(cats, nums, refs), tr_c[cats + nums], tr_c["salary_mid"], cv=cv,
                                  scoring="r2").mean()
        model = linear_model(cats, nums, refs).fit(tr_c[cats + nums], tr_c["salary_mid"])
        out += [f"rows identical on {label}:",
                (f"  training {len(tr)} -> {len(tr_c)} after collapsing; test {len(te)} -> {len(te_c)}; "
                f"test rows with an identical twin in training: {len(twins)}"),
                f"  all-features CV R2 on training rows (5x20 folds): {cv_full:.3f} -> {cv_coll:.3f}",
                (f"  test R2, trained on collapsed rows: full test {r2_score(te['salary_mid'], model.predict(te[cats + nums])):.3f}; "
                f"collapsed test {r2_score(te_c['salary_mid'], model.predict(te_c[cats + nums])):.3f}")]
    out.append("(the 0.746 -> 0.70 in the Module 4 data check was ridge regression on structured salaries only, so the "
               "levels differ; the drop is the comparable part)")
    return out


# ---------------------------------------------------------------- forward_deployed vs rest

def binary_data(subset: pd.DataFrame | None = None):
    df = nb.load_subset()[0] if subset is None else subset
    train, test = nb.split_rows(df["role_family"])
    texts = nb.build_variants(df)["masked"]
    y = df["role_family"].eq(POSITIVE).astype(int)
    return texts, y, train, test


def folds() -> StratifiedKFold:
    return StratifiedKFold(nb.CV_FOLDS, shuffle=True, random_state=nb.SPLIT_SEED)


def tune(pipe: Pipeline, param: str, grid: list[float], texts, y, train, prefer_small: bool):
    search = GridSearchCV(pipe, {param: grid}, cv=folds(), scoring=CV_SCORE, refit=False)
    search.fit(texts.iloc[train], y.iloc[train])
    res = search.cv_results_
    table = pd.DataFrame({"cv_mean": res["mean_test_score"], "cv_sd": res["std_test_score"]},
                         index=pd.Index(res[f"param_{param}"].astype(float), name=param.split("__")[1]))
    best = table.index[np.isclose(table["cv_mean"], table["cv_mean"].max())]
    return table, float(best.min() if prefer_small else best.max())


def class_metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "accuracy": (tp + tn) / len(y),
            "balanced_acc": balanced_accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "F1": f1_score(y, pred), "AUC": roc_auc_score(y, proba),
            "AP": average_precision_score(y, proba), "pred": pred}


def word_table(model: Pipeline, texts, y, train) -> pd.DataFrame:
    vec, lr = model.named_steps["vec"], model.named_steps["lr"]
    X = vec.transform(texts.iloc[train])
    ytr = y.iloc[train].to_numpy()
    present = (X > 0).astype(int)
    Xc = X.tocsc()
    typical = np.array([np.median(Xc[:, j].data) if Xc[:, j].nnz else 0.0 for j in range(X.shape[1])])
    coef = lr.coef_[0]
    return pd.DataFrame({"word": vec.get_feature_names_out(), "coef": coef,
                         "OR_per_1.0": np.exp(coef), "OR_per_0.1": np.exp(coef * 0.1),
                         "typical_weight": typical, "OR_at_typical": np.exp(coef * typical),
                         f"in_{POSITIVE}_of_{int(ytr.sum())}": np.asarray(present[ytr == 1].sum(axis=0)).ravel(),
                         f"in_rest_of_{int((ytr == 0).sum())}": np.asarray(present[ytr == 0].sum(axis=0)).ravel()})


def baseline_lines(ytr: pd.Series, yte: pd.Series) -> list[str]:
    k, n = int(yte.sum()), len(yte)
    return [f"label: {POSITIVE} = 1, solutions and software_eng = 0; masked text, Module 3 split",
            f"  training {int(ytr.sum())} positive / {len(ytr) - int(ytr.sum())} negative; test {k} / {n - k}",
            (f"  majority baseline on test (always 0): accuracy {(n - k) / n:.4f} ({n - k}/{n}), balanced accuracy 0.500, "
             f"recall 0, F1 0, AUC 0.500, AP = prevalence {k / n:.3f}"),
            f"  one test positive = {100 / k:.1f} points of recall; one test posting = {100 / n:.2f} points of accuracy"]


def fit_binary(texts, y, train, test) -> dict:
    """C and the two alphas by CV on the training rows, then every model scored once on the test rows."""
    ytr, yte = y.iloc[train], y.iloc[test]
    lr_pipe = Pipeline([("vec", nb.make_vectorizer()),
                        ("lr", LogisticRegression(class_weight="balanced", max_iter=10000))])
    c_table, best_c = tune(lr_pipe, "lr__C", C_GRID, texts, y, train, prefer_small=True)
    balanced = lr_pipe.set_params(lr__C=best_c).fit(texts.iloc[train], ytr)
    plain = Pipeline([("vec", nb.make_vectorizer()), ("lr", LogisticRegression(C=best_c, max_iter=10000))])
    plain.fit(texts.iloc[train], ytr)

    results = {"logistic, balanced": class_metrics(yte, balanced.predict_proba(texts.iloc[test])[:, 1]),
               "logistic, unbalanced (default)": class_metrics(yte, plain.predict_proba(texts.iloc[test])[:, 1])}
    nb_tuning = []
    for name, (cls, binary) in nb.MODELS.items():
        pipe = Pipeline([("vec", nb.make_vectorizer(binary)), ("nb", cls())])
        a_table, alpha = tune(pipe, "nb__alpha", nb.CV_ALPHAS, texts, y, train, prefer_small=False)
        nb_tuning.append((name, a_table, alpha))
        model = pipe.set_params(nb__alpha=alpha).fit(texts.iloc[train], ytr)
        results[f"{name}, alpha {alpha:g}"] = class_metrics(yte, model.predict_proba(texts.iloc[test])[:, 1])
    return {"c_table": c_table, "best_c": best_c, "balanced": balanced, "plain": plain, "results": results,
            "nb_tuning": nb_tuning}


def tuning_lines(fit: dict) -> list[str]:
    lines = ["", f"C by stratified 5-fold CV on training rows, scored on {CV_SCORE} (class_weight='balanced'):",
             fit["c_table"].to_string(float_format="{:.4f}".format), f"selected C = {fit['best_c']:g}"]
    for name, a_table, alpha in fit["nb_tuning"]:
        lines += ["", f"{name} alpha by the same CV ({CV_SCORE}):", a_table.to_string(float_format="{:.4f}".format),
                  f"selected alpha = {alpha:g}"]
    return lines


def test_lines(results: dict, yte: pd.Series) -> list[str]:
    k, n = int(yte.sum()), len(yte)
    cols = ["TP", "FN", "FP", "TN", "accuracy", "balanced_acc", "precision", "recall", "F1", "AUC", "AP"]
    table = pd.DataFrame({name: {c: v[c] for c in cols} for name, v in results.items()}).T
    lines = ["", f"TEST SET ({n} postings, {k} {POSITIVE}); threshold 0.5; majority-baseline accuracy {(n - k) / n:.4f}",
             "'unbalanced (default)' is the same C without class weights, so the gap is the effect of balancing.",
             table.to_string(formatters={**{c: "{:.0f}".format for c in cols[:4]},
                                         **{c: "{:.4f}".format for c in cols[4:]}})]
    for name, r in results.items():
        ci = binomtest(int(r["TP"]), k).proportion_ci(method="wilson")
        lines.append(f"  {name}: confusion [[TN {r['TN']}, FP {r['FP']}], [FN {r['FN']}, TP {r['TP']}]]; "
                     f"recall {r['TP']}/{k}, 95% interval {ci.low:.3f}-{ci.high:.3f}")
    return lines


def separation_lines(results: dict, yte: pd.Series) -> list[str]:
    k = int(yte.sum())
    lines = ["", "HOW MANY TEST POSTINGS SEPARATE THE MODELS (exact McNemar test on the postings they disagree on)"]
    names = list(results)
    yv = yte.to_numpy()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = results[a], results[b]
            right_a, right_b = ra["pred"] == yv, rb["pred"] == yv
            only_a, only_b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
            p = binomtest(min(only_a, only_b), only_a + only_b).pvalue if only_a + only_b else 1.0
            pos = yv == 1
            pa, pb = int((right_a & ~right_b & pos).sum()), int((~right_a & right_b & pos).sum())
            pp = binomtest(min(pa, pb), pa + pb).pvalue if pa + pb else 1.0
            lines.append(f"  {a} vs {b}: postings classified differently {int((ra['pred'] != rb['pred']).sum())}; "
                         f"right only by the first {only_a}, only by the second {only_b} (p = {p:.3f}); "
                         f"{POSITIVE} found {ra['TP']} vs {rb['TP']} ({int(ra['TP'] - rb['TP']):+d} postings, "
                         f"{100 * (ra['TP'] - rb['TP']) / k:+.1f} recall points; on the {k} positives alone "
                         f"{pa} vs {pb} found by only one, p = {pp:.3f}); false positives {ra['FP']} vs {rb['FP']}")
    lines.append("  (p is the exact McNemar test: the chance of a split at least this uneven among the postings the two "
                 "models disagree on, if neither model were better)")
    return lines


def word_lines(balanced: Pipeline, texts, y, train) -> list[str]:
    words = word_table(balanced, texts, y, train)
    fmt = {"coef": "{:+.3f}".format, "OR_per_1.0": "{:.3g}".format, "OR_per_0.1": "{:.3f}".format,
           "typical_weight": "{:.3f}".format, "OR_at_typical": "{:.3f}".format}
    shown = pd.concat([words.nlargest(N_WORDS, "coef"), words.nsmallest(N_WORDS, "coef")])
    rare = shown[shown.iloc[:, -2] + shown.iloc[:, -1] < 10]
    return ["", (f"BALANCED LOGISTIC COEFFICIENTS as odds ratios. A TF-IDF weight of 1.0 is the whole posting; the "
                 f"typical weight is the median weight among training postings that use the word, and OR_at_typical "
                 f"is the odds ratio for a posting that uses the word at that weight. Intercept "
                 f"{balanced.named_steps['lr'].intercept_[0]:+.3f}."),
            f"{N_WORDS} highest (toward {POSITIVE}):",
            words.nlargest(N_WORDS, "coef").to_string(index=False, formatters=fmt),
            f"{N_WORDS} lowest (toward solutions or software_eng):",
            words.nsmallest(N_WORDS, "coef").to_string(index=False, formatters=fmt),
            ("words above used by fewer than 10 training postings in all (a weight resting on a handful of "
             f"postings): {', '.join(rare['word']) or 'none'}")]


def run_binary(subset: pd.DataFrame | None = None) -> list[str]:
    texts, y, train, test = binary_data(subset)
    fit = fit_binary(texts, y, train, test)
    yte = y.iloc[test]
    return (baseline_lines(y.iloc[train], yte) + tuning_lines(fit) + test_lines(fit["results"], yte)
            + separation_lines(fit["results"], yte) + word_lines(fit["balanced"], texts, y, train))


# ---------------------------------------------------------------- figures

TEXT_COLOR, MUTED = "#1C1C1A", "#6B6B66"
CLASS_COLORS = {c: PALETTE[i] for i, c in enumerate(nb.CLASSES)}  # the Module 3 class colours


def plot_pred_vs_actual(preds: pd.DataFrame, table: pd.DataFrame, fit: str = "all features"):
    """Predicted against actual midpoint on the test rows, coloured by role family, marker by salary source."""
    pred = preds[PRED_COLUMN[fit]] / 1000
    actual = preds["actual_mid"] / 1000
    scores = table.set_index("fit")
    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    lo, hi = 50, max(actual.max(), pred.max()) * 1.06
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.3, ls="--", zorder=2)
    ax.text(68, 80, "predicted = actual", fontsize=9.5, color=MUTED, rotation=45, rotation_mode="anchor")
    for source, marker in [("structured", "o"), ("text", "^")]:
        for cls in nb.CLASSES:
            m = (preds["source"] == source) & (preds["role_family"] == cls)
            ax.scatter(actual[m], pred[m], s=46, marker=marker, color=CLASS_COLORS[cls], alpha=0.8, edgecolor="white",
                       lw=0.6, zorder=3)
    class_handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLORS[c], ms=8,
                                label=f"{c} ({int((preds['role_family'] == c).sum())})") for c in nb.CLASSES]
    source_handles = [plt.Line2D([], [], marker=mk, ls="", color=MUTED, ms=8,
                                 label=f"{lab} ({int((preds['source'] == src).sum())})")
                      for src, mk, lab in [("structured", "o", "structured salary band"),
                                           ("text", "^", "pay range in the description")]]
    first = ax.legend(handles=class_handles, title="role family (colour)", loc="upper left", frameon=False,
                      fontsize=9.5, bbox_to_anchor=(1.02, 1.0), alignment="left")
    ax.add_artist(first)
    ax.legend(handles=source_handles, title="where the salary came from (marker)", loc="upper left", frameon=False,
              fontsize=9.5, bbox_to_anchor=(1.02, 0.74), alignment="left")
    s, b = scores.loc[fit], scores.loc["baseline: training mean"]
    ax.text(1.04, 0.0, f"{fit} fit, test set\nR² {s['R2']:.3f}\nRMSE ${s['RMSE']:,.0f}\nMAE ${s['MAE']:,.0f}\n\n"
            f"predicting the training mean:\nRMSE ${b['RMSE']:,.0f}, R² {b['R2']:.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=TEXT_COLOR, linespacing=1.5,
            bbox={"facecolor": "white", "edgecolor": "#DEDEDA", "pad": 6})
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual salary midpoint ($ thousands)")
    ax.set_ylabel("Predicted salary midpoint ($ thousands)")
    ax.set_title(f"Linear regression, test set: the {len(preds)} of 232 test postings that carry a salary,\n"
                 "coloured by role family (16 employers are too many to tell apart by colour)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_confusion(cm: pd.DataFrame, title: str, notes: list[str]):
    """One confusion matrix in the Module 3 style (counts with row % in brackets, coloured by row %), with notes."""
    fig, (ax, side) = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    pct = cm.div(cm.sum(axis=1), axis=0) * 100
    annot = cm.astype(str) + "\n(" + pct.round(1).astype(str) + "%)"
    sns.heatmap(pct, annot=annot.values, fmt="", cmap=CMAP, vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "% of the true class"}, annot_kws={"size": 10}, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    plt.setp(ax.get_yticklabels(), rotation=0)
    side.axis("off")
    side.text(0, 0.5, "\n".join(notes), fontsize=10, color=TEXT_COLOR, va="center", linespacing=1.6)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_confusion_logistic(res: dict, y_test: pd.Series, c: float):
    """The balanced logistic model's 2 x 2 confusion matrix, with the majority baseline beside it."""
    labels = ["FD", "rest"]
    cm = pd.DataFrame([[res["TP"], res["FN"]], [res["FP"], res["TN"]]], index=pd.Index(labels, name="true"),
                      columns=pd.Index(labels, name="predicted")).astype(int)
    k, n = int(y_test.sum()), len(y_test)
    notes = [f"accuracy {res['accuracy']:.1%}",
             f"majority baseline (always 'rest'): {(n - k) / n:.1%}",
             f"balanced accuracy {res['balanced_acc']:.3f}",
             f"precision {res['precision']:.3f}, recall {res['recall']:.3f}",
             f"F1 {res['F1']:.3f}, AUC {res['AUC']:.3f}, AP {res['AP']:.3f}",
             f"C = {c:g}, class_weight balanced, threshold 0.5",
             "rest = solutions and software_eng"]
    return plot_confusion(cm, f"Logistic regression, forward_deployed (FD) against the rest: test set "
                              f"(n = {n}, {k} FD)", notes)


# ---------------------------------------------------------------- report

def cleanup_lines(df: pd.DataFrame) -> list[str]:
    pr = pd.read_csv(xp.OUT_CSV, dtype={"jobId": str})
    inferred = pr[pr["interval_converted"].eq(True) & pr["interval_source"].eq("inferred")
                  & pr["text_currency"].eq("USD") & ~pr["text_placeholder"].eq(True) & ~pr["text_suspect"].eq(True)
                  & pr["text_basis"].ne("ote")]
    stated = pr[pr["text_usable_usd"] & pr["interval_converted"].eq(True)]
    lines = ["hourly figures whose interval was only inferred from magnitude, usable before and now dropped:"]
    lines += [f"  {r.company_clean} jobId {r.jobId}: {r.text_span!r} under '{r.text_label}' "
              f"(would have been {r.text_annual_min:,.0f}-{r.text_annual_max:,.0f}); in the subset: "
              f"{r.jobId in set(df['jobId'])}" for r in inferred.itertuples()]
    lines += [f"stated hourly figures kept: {len(stated)} ({stated['company_clean'].value_counts().to_dict()})",
              (f"multi_zone flag: {int(pr['multi_zone'].sum())} postings in the full set, "
              f"{int(pr['multi_zone'][pr['jobId'].isin(df['jobId'])].sum())} in the subset; it marks a pick made among "
              f"two or more different USD base-pay ranges. Only rows whose salary comes from the text use the pick."),
              ""] + EXTRACTION_AUDIT
    return lines


def main() -> None:
    df = salary_frame()
    lin_lines, preds = run_linear(df)
    report = ["MODULE 4: LINEAR AND LOGISTIC REGRESSION",
              ("Module 3 subset (1,159 postings), Module 3 split (927 training, 232 test); fitted on training rows "
              "only, scored once on the test rows."),
              ("Never used as features: title, seniority, is_near_dup; for salary also the description text, which "
              "the target is partly taken from."),
              "", "=" * 100, "1. CLEANUPS BEFORE MODELLING", *cleanup_lines(df),
              "", "=" * 100, "2. LINEAR REGRESSION: salary midpoint (dollars)", *lin_lines,
              "", "=" * 100, "3-4. LOGISTIC REGRESSION AND NAIVE BAYES: forward_deployed vs rest", *run_binary()]
    REPORT_PATH.write_text("\n".join(report) + "\n")
    preds.to_csv(PRED_PATH, index=False)
    print("\n".join(report))
    print(f"\nsaved {REPORT_PATH.relative_to(tc.ROOT)} and {PRED_PATH.relative_to(tc.ROOT)}")


if __name__ == "__main__":
    main()
