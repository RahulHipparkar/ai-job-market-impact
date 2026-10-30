"""Decision trees on the forward_deployed / solutions / software_eng postings.

Same 1,159 postings and the same 80/20 split as text_nb, so every number is comparable
with Naive Bayes. Four feature sets: all tabular columns; department removed;
department and company removed; and the masked-text TF-IDF matrix that the masked
MultinomialNB uses. Encoders, imputers and the vectoriser are fitted on training rows
only (inside the pipeline, so also per CV fold).
"""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

try:  # imported as src.role_dt from the notebook, or as role_dt from src/ scripts
    from src import text_nb as nb
    from src.plotstyle import CMAP, PALETTE
except ImportError:
    import text_nb as nb
    from plotstyle import CMAP, PALETTE

CLASSES = nb.CLASSES
REPORT_PATH = nb.tc.ROOT / "results" / "dt_report.txt"
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]  # None = grow until leaves are pure
CRITERIA = ["gini", "entropy"]
TREE_SEED = 42
PLOT_DEPTH = 2  # tree diagrams are drawn to this depth for legibility

# Tabular features. Location: the project's cleaned location plus location count; the raw
# city / region / country fields repeat it with more gaps (country is 59.5% missing, with
# unharmonised spellings). Work mode: workplaceType (is_remote is derived from it).
CATEGORICAL = ["department", "company_clean", "location_clean", "workplaceType"]
SALARY = ["salary_annual_min", "salary_annual_max"]
NUMERIC = SALARY + ["desc_len", "location_count", "is_multi_location"]
TABULAR = CATEGORICAL + NUMERIC
# Never used: the target, the title, and columns parsed from the title.
EXCLUDED = {"role_family": "the target",
            "title": "the label is parsed from it",
            "seniority": "parsed from the title by the same keyword rule",
            "is_near_dup": "near-duplicate flag; its key includes the normalised title"}

TREES = {
    "tree1": ("all tabular features", TABULAR),
    "tree2": ("department removed", [c for c in TABULAR if c != "department"]),
    "tree3": ("department and company removed",
              [c for c in TABULAR if c not in ("department", "company_clean")]),
    "tree4": ("masked text TF-IDF", None),
}


def tabular_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The tabular feature columns; suspect salaries are treated as missing."""
    X = df[TABULAR].copy()
    X[CATEGORICAL] = X[CATEGORICAL].astype(object)
    X.loc[df["salary_suspect"].eq(True), SALARY] = np.nan
    X["is_multi_location"] = X["is_multi_location"].astype(int)
    return X


def tabular_encoder(columns: list[str]) -> ColumnTransformer:
    """Categoricals: missing becomes its own "missing" category, then one-hot; a category
    unseen in training encodes as all zeros. Numerics: median imputation, plus one
    indicator for missing salary (min and max are missing on the same rows)."""
    cats = [c for c in CATEGORICAL if c in columns]
    nums = [c for c in NUMERIC if c in columns]
    return ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), cats),
        ("num", SimpleImputer(strategy="median"), nums),
        ("salary_missing", MissingIndicator(features="all"), ["salary_annual_min"]),
    ])


def make_model(tree: str, criterion: str, depth=None) -> Pipeline:
    features = nb.make_vectorizer() if tree == "tree4" else tabular_encoder(TREES[tree][1])
    return Pipeline([("features", features),
                     ("tree", DecisionTreeClassifier(criterion=criterion, max_depth=depth,
                                                     random_state=TREE_SEED))])


def feature_names(model: Pipeline) -> np.ndarray:
    """Readable encoded feature names: "department = Engineering", "salary missing"."""
    names = model.named_steps["features"].get_feature_names_out()
    out = []
    for n in names:
        if n.startswith("cat__"):
            rest = n[len("cat__"):]
            col = next(c for c in CATEGORICAL if rest.startswith(c + "_"))
            out.append(f"{col} = {rest[len(col) + 1:]}")
        elif n.startswith("num__"):
            out.append(n[len("num__"):])
        elif n.startswith("salary_missing__"):
            out.append("salary missing")
        else:
            out.append(n)
    return np.array(out)


def source_feature(name: str) -> str:
    """The input column an encoded feature came from."""
    if name == "salary missing":
        return "salary missing (indicator)"
    return name.split(" = ")[0]


def depth_label(depth) -> str:
    return "none" if depth is None else str(depth)


def cv_depth(X, y: pd.Series, train: np.ndarray, tree: str, criterion: str) -> tuple[pd.DataFrame, object]:
    """Stratified k-fold CV accuracy for each max_depth, on the training rows only.

    The highest mean wins; a tie goes to the shallower tree (None counts as deepest).
    """
    folds = StratifiedKFold(nb.CV_FOLDS, shuffle=True, random_state=nb.SPLIT_SEED)
    search = GridSearchCV(make_model(tree, criterion), {"tree__max_depth": DEPTHS}, cv=folds,
                          scoring="accuracy", refit=False, return_train_score=True)
    search.fit(X.iloc[train], y.iloc[train])
    res = search.cv_results_
    table = pd.DataFrame({"cv_mean": res["mean_test_score"], "cv_sd": res["std_test_score"],
                          "cv_train_mean": res["mean_train_score"]},
                         index=pd.Index([depth_label(d) for d in res["param_tree__max_depth"]],
                                        name="max_depth"))
    best = np.flatnonzero(np.isclose(table["cv_mean"], table["cv_mean"].max()))[0]
    return table, DEPTHS[best]


def run_tree(X, y: pd.Series, train: np.ndarray, test: np.ndarray, tree: str) -> dict:
    """Both criteria for one feature set: CV depth, final fit on all training rows, test metrics."""
    out = {}
    for criterion in CRITERIA:
        cv_table, depth = cv_depth(X, y, train, tree, criterion)
        model = make_model(tree, criterion, depth).fit(X.iloc[train], y.iloc[train])
        res = nb.evaluate(model, X.iloc[test], y.iloc[test])
        fitted = model.named_steps["tree"]
        res.update(model=model, cv=cv_table, depth=depth, criterion=criterion,
                   train_accuracy=accuracy_score(y.iloc[train], model.predict(X.iloc[train])),
                   macro_f1=f1_score(y.iloc[test], res["pred"], labels=CLASSES, average="macro"),
                   n_features=fitted.n_features_in_, fitted_depth=fitted.get_depth(),
                   n_leaves=fitted.get_n_leaves())
        out[criterion] = res
    return out


def identity_probe(X, target: pd.Series, train: np.ndarray, test: np.ndarray, tree: str) -> dict:
    """How well one tree's feature set identifies another label (company or department).

    Same encoder, same split, max_depth by the same CV procedure (stratified on the probed
    label), criterion gini. Labels with fewer training rows than folds make StratifiedKFold
    warn; the warning is silenced and the smallest label count reported instead.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        cv_table, depth = cv_depth(X, target, train, tree, "gini")
    model = make_model(tree, "gini", depth).fit(X.iloc[train], target.iloc[train])
    pred = model.predict(X.iloc[test])
    y_test = target.iloc[test]
    majority = target.iloc[train].value_counts().idxmax()
    per_label = (pd.DataFrame({"label": y_test.values, "correct": pred == y_test.values})
                 .groupby("label")["correct"].agg(n_test="size", recall="mean")
                 .sort_values("n_test", ascending=False))
    return {"cv": cv_table, "depth": depth, "model": model,
            "train_accuracy": accuracy_score(target.iloc[train], model.predict(X.iloc[train])),
            "accuracy": accuracy_score(y_test, pred),
            "baseline": (majority, float((y_test == majority).mean())),
            "labels_in_train": target.iloc[train].nunique(),
            "smallest_train_label": int(target.iloc[train].value_counts().min()),
            "test_rows_unseen_label": int((~y_test.isin(set(target.iloc[train]))).sum()),
            "per_label": per_label, "importance": importances(model)[1]}


def probe_lines(name: str, res: dict) -> list[str]:
    return [f"{name}",
            (f"  labels in training rows: {res['labels_in_train']} (smallest has {res['smallest_train_label']} rows); "
             f"test rows whose label never occurs in training: {res['test_rows_unseen_label']}"),
            "  CV accuracy by max_depth: " + ", ".join(f"{d}: {r.cv_mean:.4f}" for d, r in res["cv"].iterrows()),
            (f"  selected max_depth: {depth_label(res['depth'])}; train accuracy {res['train_accuracy']:.4f}; "
             f"TEST ACCURACY {res['accuracy']:.4f}; majority baseline ({res['baseline'][0]}) {res['baseline'][1]:.4f}"),
            "  importance summed per input column: "
            + ", ".join(f"{k} {v:.4f}" for k, v in res["importance"].items())]


def largest_share(rows: pd.Series, by: pd.Series, top: int = 10) -> pd.DataFrame:
    """For the most common values of rows, the share of postings from the single largest value of by."""
    ct = pd.crosstab(rows, by)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index[:top]]
    return pd.DataFrame({"postings": ct.sum(axis=1), "largest": ct.idxmax(axis=1),
                         "largest n": ct.max(axis=1), "share": ct.max(axis=1) / ct.sum(axis=1)})


def fixed_depth_comparison(X_by_tree: dict, y: pd.Series, train: np.ndarray, test: np.ndarray,
                           results: dict) -> pd.DataFrame:
    """Both criteria refitted at the gini-selected max_depth of each tree."""
    rows = {}
    for tree, by_criterion in results.items():
        depth = by_criterion["gini"]["depth"]
        X = X_by_tree[tree]
        for criterion in CRITERIA:
            model = make_model(tree, criterion, depth).fit(X.iloc[train], y.iloc[train])
            pred = model.predict(X.iloc[test])
            rows[(tree, criterion)] = {
                "max_depth (gini-selected)": depth_label(depth),
                "fitted depth": model.named_steps["tree"].get_depth(),
                "train accuracy": accuracy_score(y.iloc[train], model.predict(X.iloc[train])),
                "test accuracy": accuracy_score(y.iloc[test], pred),
                "macro F1": f1_score(y.iloc[test], pred, labels=CLASSES, average="macro")}
    out = pd.DataFrame(rows).T
    out.index.names = ["tree", "criterion"]
    return out


def summary_row(res: dict) -> dict:
    return {"max_depth (CV)": depth_label(res["depth"]), "fitted depth": res["fitted_depth"],
            "leaves": res["n_leaves"], "features": res["n_features"],
            "train accuracy": res["train_accuracy"], "test accuracy": res["accuracy"],
            "train - test": res["train_accuracy"] - res["accuracy"], "macro F1": res["macro_f1"]}


def result_lines(tree: str, res: dict, baseline: tuple[str, float]) -> list[str]:
    """Every number for one tree x criterion, as printable report lines."""
    return [f"--- {tree} ({TREES[tree][0]}) x criterion = {res['criterion']} ---",
            f"features after encoding: {res['n_features']}",
            f"{nb.CV_FOLDS}-fold stratified CV on the training rows (accuracy, mean +/- sd; mean training-fold accuracy):",
            *[f"  max_depth {d:<5}: {r.cv_mean:.4f} +/- {r.cv_sd:.4f}   (train folds {r.cv_train_mean:.4f})"
              for d, r in res["cv"].iterrows()],
            (f"selected max_depth: {depth_label(res['depth'])} (fitted depth {res['fitted_depth']}, "
             f"{res['n_leaves']} leaves)"),
            (f"TRAIN ACCURACY: {res['train_accuracy']:.4f}   TEST ACCURACY: {res['accuracy']:.4f}   "
             f"gap: {res['train_accuracy'] - res['accuracy']:.4f}   macro F1: {res['macro_f1']:.4f}"),
            f"majority-class baseline ({baseline[0]}): {baseline[1]:.4f}",
            *nb.evaluation_lines(res)]


def transformed(model: Pipeline, X):
    return model.named_steps["features"].transform(X)


def node_class_counts(model: Pipeline, X_train, y_train: pd.Series) -> pd.DataFrame:
    """Training postings of each class reaching every node (nodes x CLASSES)."""
    paths = model.named_steps["tree"].decision_path(transformed(model, X_train))
    return pd.DataFrame({c: np.asarray(paths[(y_train == c).to_numpy()].sum(axis=0)).ravel()
                         for c in CLASSES})


def node_depths(fitted: DecisionTreeClassifier) -> np.ndarray:
    t = fitted.tree_
    depth = np.zeros(t.node_count, dtype=int)
    for node in range(t.node_count):  # children always have larger ids than their parent
        for child in (t.children_left[node], t.children_right[node]):
            if child != -1:
                depth[child] = depth[node] + 1
    return depth


def _gini_terms(counts) -> tuple[str, float]:
    n = sum(counts)
    terms = " + ".join(f"({k}/{n})^2" for k in counts)
    value = 1 - sum((k / n) ** 2 for k in counts)
    return f"1 - [{terms}] = 1 - {sum((k / n) ** 2 for k in counts):.4f} = {value:.4f}", value


def _entropy_terms(counts) -> tuple[str, float]:
    n = sum(counts)
    parts = [f"({k}/{n})log2({k}/{n})" if k else "0" for k in counts]
    value = -sum((k / n) * np.log2(k / n) for k in counts if k)
    return f"-[{' + '.join(parts)}] = {value:.4f}", value


def split_arithmetic(model: Pipeline, counts: pd.DataFrame, node: int) -> list[str]:
    """Gini and entropy before and after one split, with the arithmetic written out."""
    fitted = model.named_steps["tree"]
    t = fitted.tree_
    left, right = t.children_left[node], t.children_right[node]
    name = feature_names(model)[t.feature[node]]
    rows = {"parent": node, "left": left, "right": right}
    c = {k: counts.loc[v, CLASSES].astype(int).tolist() for k, v in rows.items()}
    n = {k: sum(v) for k, v in c.items()}
    lines = [f"node {node} (depth {node_depths(fitted)[node]}): split on '{name}' <= {t.threshold[node]:.4g}",
             "  (one-hot feature: left = postings without this value, right = postings with it)"
             if " = " in name or name == "salary missing" else "",
             f"  class counts in the order {CLASSES}:",
             *[f"    {k:6s} (node {v}): {c[k]}  n = {n[k]}" for k, v in rows.items()]]
    for label, fn in [("GINI", _gini_terms), ("ENTROPY (bits)", _entropy_terms)]:
        text = {k: fn(c[k]) for k in c}
        w_l, w_r = n["left"] / n["parent"], n["right"] / n["parent"]
        after = w_l * text["left"][1] + w_r * text["right"][1]
        lines += [f"  {label}",
                  *[f"    {k:6s}: {text[k][0]}" for k in c],
                  (f"    after the split (weighted): ({n['left']}/{n['parent']}) x {text['left'][1]:.4f} + "
                   f"({n['right']}/{n['parent']}) x {text['right'][1]:.4f} = {after:.4f}"),
                  (f"    decrease = {text['parent'][1]:.4f} - {after:.4f} = {text['parent'][1] - after:.4f}"
                   + ("   <- information gain" if label.startswith("ENTROPY") else ""))]
    if fitted.criterion == "gini":
        assert np.isclose(t.impurity[node], _gini_terms(c["parent"])[1]), "hand Gini disagrees with the tree"
    return [line for line in lines if line]


def department_decisions(model: Pipeline, X_test) -> dict[str, float]:
    """How much of each test posting's path through the tree splits on department.

    only: every split on the path is a department column; any: at least one is;
    last: the split that sends the posting into its leaf is; share: mean fraction of
    the path's splits that are department columns.
    """
    fitted = model.named_steps["tree"]
    names = feature_names(model)
    paths = fitted.decision_path(transformed(model, X_test)).toarray().astype(bool)
    is_dept = np.array([source_feature(names[f]) == "department" if f >= 0 else False
                        for f in fitted.tree_.feature])
    internal = fitted.tree_.feature >= 0
    only, any_dept, last, share = [], [], [], []
    for row in paths:
        used = np.flatnonzero(row & internal)  # node ids increase along a path
        only.append(len(used) > 0 and is_dept[used].all())
        any_dept.append(is_dept[used].any())
        last.append(len(used) > 0 and is_dept[used[-1]])
        share.append(is_dept[used].mean() if len(used) else 0.0)
    return {"only department splits on the path": float(np.mean(only)),
            "at least one department split on the path": float(np.mean(any_dept)),
            "final split before the leaf is department": float(np.mean(last)),
            "mean share of path splits on department": float(np.mean(share)),
            "test postings": len(paths)}


def importances(model: Pipeline) -> tuple[pd.Series, pd.Series]:
    """Impurity-based importance per encoded column, and summed per input column."""
    imp = pd.Series(model.named_steps["tree"].feature_importances_, index=feature_names(model))
    by_source = imp.groupby(imp.index.map(source_feature)).sum().sort_values(ascending=False)
    return imp.sort_values(ascending=False), by_source


def _short(label: str) -> str:
    return label.replace("_", " ")


def plot_confusion_dt(res_by_criterion: dict, title: str):
    """Counts with row % in brackets, coloured by row %, one panel per criterion."""
    fig, axes = plt.subplots(1, len(res_by_criterion), figsize=(12.5, 4.8), gridspec_kw={"wspace": 0.45})
    ticks = [_short(c).replace(" ", "\n") for c in CLASSES]
    for ax, (criterion, res) in zip(axes, res_by_criterion.items()):
        cm = res["confusion"]
        pct = cm.div(cm.sum(axis=1), axis=0) * 100
        annot = cm.astype(str) + "\n(" + pct.round(1).astype(str) + "%)"
        sns.heatmap(pct, annot=annot.values, fmt="", cmap=CMAP, vmin=0, vmax=100, linewidths=0.5,
                    linecolor="white", cbar_kws={"label": "% of the true class"}, annot_kws={"size": 9},
                    ax=ax, xticklabels=ticks, yticklabels=ticks)
        ax.set_title(f"{criterion}, max_depth = {depth_label(res['depth'])}\n"
                     f"(accuracy {res['accuracy']:.1%}, macro F1 {res['macro_f1']:.3f})", fontsize=11)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.grid(False)
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=0)
    fig.suptitle(f"Decision tree confusion matrices, test set (n = {int(cm.values.sum())}): {title}")
    fig.subplots_adjust(top=0.8)
    return fig


def plot_cv_depth(results: dict):
    """CV accuracy against max_depth for every tree, one panel per tree, both criteria."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    x = np.arange(len(DEPTHS))
    for ax, (tree, by_criterion) in zip(axes.ravel(), results.items()):
        for j, (criterion, res) in enumerate(by_criterion.items()):
            cv = res["cv"]
            ax.plot(x, cv["cv_mean"], marker="o", ms=4, lw=2, color=PALETTE[j], label=criterion)
            ax.fill_between(x, cv["cv_mean"] - cv["cv_sd"], cv["cv_mean"] + cv["cv_sd"],
                            color=PALETTE[j], alpha=0.12, lw=0)
            sel = DEPTHS.index(res["depth"])
            ax.plot(x[sel], cv["cv_mean"].iloc[sel], marker="o", ms=10, mfc="none", mec=PALETTE[j], mew=2)
        ax.set_xticks(x, [depth_label(d) for d in DEPTHS])
        ax.set_title(f"{tree}: {TREES[tree][0]}", fontsize=11)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("CV accuracy (mean ± SD)")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"Choosing max_depth: {nb.CV_FOLDS}-fold stratified CV on the training set "
                 "(circled = selected)")
    fig.tight_layout()
    return fig


def plot_importance(model: Pipeline, title: str, n: int = 15):
    """Importance summed per input column (left) and the top encoded columns (right)."""
    imp, by_source = importances(model)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    s = by_source[by_source > 0][::-1]
    a1.barh(s.index, s.values, color=PALETTE[0], height=0.65, zorder=3)
    a1.set_title("Summed per input column")
    top = imp.head(n)
    top = top[top > 0][::-1]
    a2.barh(top.index, top.values, color=PALETTE[0], height=0.65, zorder=3)
    a2.set_title(f"Top {len(top)} encoded columns")
    for a in (a1, a2):
        a.set_xlabel("Impurity-based importance (sums to 1)")
        a.grid(axis="y", visible=False)
        a.tick_params(axis="y", labelsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_tree_diagram(model: Pipeline, title: str, max_depth: int = PLOT_DEPTH):
    """The fitted tree drawn to max_depth; node boxes tinted by majority class.

    One-hot split lines are broken into feature, category and threshold, so long
    category names do not run into the neighbouring node.
    """
    fitted = model.named_steps["tree"]
    truncated = fitted.get_depth() > max_depth
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    annotations = plot_tree(fitted, max_depth=max_depth, feature_names=list(feature_names(model)),
                            class_names=list(fitted.classes_), impurity=True, filled=False,
                            rounded=True, fontsize=10, ax=ax)
    edge = {c: PALETTE[CLASSES.index(c)] for c in CLASSES}
    # opaque light tint (80% white), so the arrows do not show through the node text
    fill = {c: tuple(0.8 + 0.2 * v for v in to_rgb(e)) for c, e in edge.items()}
    for a in annotations:
        a.set_text(re.sub(r"^(\w+) = (.+) (<= \S+)\n", r"\1 =\n\2\n\3\n", a.get_text()))
        m = re.search(r"class = (\w+)", a.get_text())
        if m and a.get_bbox_patch() is not None:
            a.get_bbox_patch().set_facecolor(fill[m.group(1)])
            a.get_bbox_patch().set_edgecolor(edge[m.group(1)])
    ax.legend(handles=[Patch(facecolor=fill[c], edgecolor=edge[c], label=c) for c in CLASSES],
              loc="upper right", frameon=False, title="majority class")
    shown = (f"drawn to depth {max_depth} of {fitted.get_depth()} (deeper nodes truncated, shown as (...))"
             if truncated else f"full tree, depth {fitted.get_depth()}")
    ax.set_title(f"{title}\n{shown}\nvalue = training postings per class in the order "
                 f"{list(fitted.classes_)}", fontsize=12)
    return fig, truncated
