"""AdaBoost on the Module 3 role-family task, compared with the Part 3 decision trees.

Same three classes, the same 1,159 postings and 927/232 split, and the same tabular features as
tree 1, encoded and imputed by role_dt's own pipeline (role_dt.tabular_frame and
role_dt.tabular_encoder). title, seniority and is_near_dup are never used. The base estimator is
a decision stump (max_depth=1).

n_estimators and learning_rate are tuned together by stratified 5-fold CV on the training rows
(the Module 3 folds), scored on accuracy as the trees were. Each fold's booster is grown once to
the largest round count and scored after every round (staged predictions), which gives CV
accuracy at every round count for every learning rate; a booster stopped after k rounds is the
same as the first k rounds of a longer one, so the grid points are read off these curves. The
selected pair is the best grid point; ties go to fewer rounds, then the smaller learning rate.

A second, smaller comparison runs AdaBoost on the Module 3 masked TF-IDF, against tree 4 and the
masked-text Naive Bayes models.

Output: results/adaboost_report.txt, and results/adaboost_cv_curve.csv with CV accuracy (and
training-fold accuracy) at every round count for every learning rate.

usage: python src/role_adaboost.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import binomtest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from src import regression_models as rm
    from src import role_dt as dt
    from src import text_nb as nb
    from src.plotstyle import PALETTE
except ImportError:
    import regression_models as rm
    import role_dt as dt
    import text_nb as nb
    from plotstyle import PALETTE

REPORT_PATH = nb.tc.ROOT / "results" / "adaboost_report.txt"
CURVE_PATH = nb.tc.ROOT / "results" / "adaboost_cv_curve.csv"
CLASSES = nb.CLASSES  # forward_deployed, solutions, software_eng: the row order used throughout
SHORT = {"forward_deployed": "FD", "solutions": "Sol", "software_eng": "SWE"}
SEED = dt.TREE_SEED

TABULAR_RATES = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5]
TABULAR_ROUNDS = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500]
TEXT_RATES = [0.25, 0.5, 1.0]
TEXT_ROUNDS = [25, 50, 100, 200, 300, 500, 750, 1000]
PLATEAU_TOL = 0.005  # a curve has plateaued once it is within this of its own maximum
TABULAR, TEXT = "tabular (tree 1)", "masked TF-IDF (tree 4)"  # feature-set labels in the curve file
MUTED, TEXT_COLOR = "#6B6B66", "#1C1C1A"

# The Module 3 models compared against, with their CV-chosen settings and published test accuracy
TREE1 = ("tree1", "gini", None, 0.9224)
TREE4 = ("tree4", "gini", 12, 0.8879)
NB_MASKED = {"MultinomialNB": (MultinomialNB, False, 0.1, 0.9138), "BernoulliNB": (BernoulliNB, True, 0.5, 0.9353)}


def booster(n_estimators: int, learning_rate: float) -> AdaBoostClassifier:
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=SEED), n_estimators=n_estimators,
                              learning_rate=learning_rate, random_state=SEED)


def tabular_model(n_estimators: int, learning_rate: float) -> Pipeline:
    """role_dt's tree-1 encoder followed by AdaBoost on stumps."""
    return Pipeline([("features", dt.tabular_encoder(dt.TREES["tree1"][1])),
                     ("ada", booster(n_estimators, learning_rate))])


def text_model(n_estimators: int, learning_rate: float) -> Pipeline:
    """The Module 3 masked-text TF-IDF vectoriser followed by AdaBoost on stumps."""
    return Pipeline([("features", nb.make_vectorizer()), ("ada", booster(n_estimators, learning_rate))])


def staged_cv(make, X, y: pd.Series, train: np.ndarray, rates: list[float], max_rounds: int) -> pd.DataFrame:
    """CV accuracy and training-fold accuracy after every round, for every learning rate."""
    folds = StratifiedKFold(nb.CV_FOLDS, shuffle=True, random_state=nb.SPLIT_SEED)
    Xt, yt = X.iloc[train], y.iloc[train]
    rows = []
    for rate in rates:
        val, fit = [], []
        for f_tr, f_va in folds.split(Xt, yt):
            model = make(max_rounds, rate).fit(Xt.iloc[f_tr], yt.iloc[f_tr])
            ada = model.named_steps["ada"]
            for part, idx, store in [("val", f_va, val), ("fit", f_tr, fit)]:
                feats = model.named_steps["features"].transform(Xt.iloc[idx])
                acc = [accuracy_score(yt.iloc[idx], p) for p in ada.staged_predict(feats)]
                store.append(acc + [acc[-1]] * (max_rounds - len(acc)))  # a stopped booster keeps its last answer
        val, fit = np.array(val), np.array(fit)
        rows += [{"learning_rate": rate, "n_estimators": k + 1, "cv_mean": val[:, k].mean(), "cv_sd": val[:, k].std(),
                  "cv_train_mean": fit[:, k].mean()} for k in range(max_rounds)]
    return pd.DataFrame(rows)


def select(curve: pd.DataFrame, grid: list[int]) -> tuple[float, int]:
    """Best grid point by mean CV accuracy; ties go to fewer rounds, then the smaller learning rate."""
    g = curve[curve["n_estimators"].isin(grid)]
    best = g[np.isclose(g["cv_mean"], g["cv_mean"].max())].sort_values(["n_estimators", "learning_rate"]).iloc[0]
    return float(best["learning_rate"]), int(best["n_estimators"])


def surface_lines(curve: pd.DataFrame, grid: list[int]) -> list[str]:
    g = curve[curve["n_estimators"].isin(grid)]
    out = []
    for col, title in [("cv_mean", "mean CV accuracy"), ("cv_sd", "CV accuracy, sd across the 5 folds"),
                       ("cv_train_mean", "training-fold accuracy (for the gap)")]:
        piv = g.pivot(index="n_estimators", columns="learning_rate", values=col)
        out += [f"{title} (rows n_estimators, columns learning_rate):", piv.to_string(float_format="{:.4f}".format)]
    out.append("each curve over every round count:")
    for rate, c in curve.groupby("learning_rate"):
        peak = c.loc[c["cv_mean"].idxmax()]
        plateau = c[c["cv_mean"] >= peak["cv_mean"] - PLATEAU_TOL].iloc[0]
        last = c.iloc[-1]
        out.append(f"  learning_rate {rate:g}: peak {peak['cv_mean']:.4f} at {int(peak['n_estimators'])} rounds; within "
                   f"{PLATEAU_TOL} of the peak from {int(plateau['n_estimators'])} rounds; at {int(last['n_estimators'])} "
                   f"rounds {last['cv_mean']:.4f} (training folds {last['cv_train_mean']:.4f})")
    return out


def evaluate(model, X, y: pd.Series, train: np.ndarray, test: np.ndarray) -> dict:
    res = nb.evaluate(model, X.iloc[test], y.iloc[test])
    p, r, f, s = precision_recall_fscore_support(y.iloc[test], res["pred"], labels=CLASSES, zero_division=0)
    res.update(train_accuracy=accuracy_score(y.iloc[train], model.predict(X.iloc[train])),
               macro_f1=f1_score(y.iloc[test], res["pred"], labels=CLASSES, average="macro"),
               per_class=pd.DataFrame({"precision": p, "recall": r, "F1": f, "support": s}, index=CLASSES))
    return res


def result_lines(name: str, res: dict) -> list[str]:
    cm = res["confusion"].rename(index=SHORT, columns=SHORT)
    return [(f"{name}: test accuracy {res['accuracy']:.4f}, macro F1 {res['macro_f1']:.4f}, train accuracy "
            f"{res['train_accuracy']:.4f}, gap {res['train_accuracy'] - res['accuracy']:.4f}"),
            "confusion matrix (rows true, columns predicted):", cm.to_string(),
            res["per_class"].rename(index=SHORT).to_string(formatters={"precision": "{:.4f}".format,
                                                                        "recall": "{:.4f}".format,
                                                                        "F1": "{:.4f}".format})]


def comparison_lines(results: dict, y_test: pd.Series) -> list[str]:
    """One table, then how many test postings separate each pair, overall and by role family."""
    table = pd.DataFrame({name: {"test accuracy": r["accuracy"], "macro F1": r["macro_f1"],
                                 **{f"recall {SHORT[c]}": r["per_class"].loc[c, "recall"] for c in CLASSES},
                                 "train accuracy": r["train_accuracy"], "gap": r["train_accuracy"] - r["accuracy"],
                                 "correct (of 232)": int((r["pred"] == y_test.to_numpy()).sum())}
                          for name, r in results.items()}).T
    out = [table.to_string(formatters={c: ("{:.0f}".format if c.startswith("correct") else "{:.4f}".format)
                                       for c in table.columns})]
    names, yv = list(results), y_test.to_numpy()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = results[a]["pred"] == yv, results[b]["pred"] == yv
            only_a, only_b = int((ra & ~rb).sum()), int((~ra & rb).sum())
            p = binomtest(min(only_a, only_b), only_a + only_b).pvalue if only_a + only_b else 1.0
            by_class = "; ".join(f"{SHORT[c]} {int((ra & ~rb & (yv == c)).sum())} vs {int((~ra & rb & (yv == c)).sum())}"
                                 for c in CLASSES)
            out += [(f"{a} vs {b}: {only_a + only_b} test postings separate them; right only by {a} {only_a}, only by "
                    f"{b} {only_b}; net {only_a - only_b:+d} postings = {100 * (only_a - only_b) / len(yv):+.2f} accuracy "
                    f"points (one posting = {100 / len(yv):.2f}); exact McNemar p = {p:.3f}"),
                    f"  by role family (right only by {a} vs only by {b}): {by_class}"]
    out.append("(exact McNemar p: the chance of a split at least this uneven among the postings the two models "
               "disagree on, if neither model were better)")
    return out


def importance_lines(model: Pipeline, n: int = 15) -> list[str]:
    names = dt.feature_names(model)
    imp = pd.Series(model.named_steps["ada"].feature_importances_, index=names)
    by_source = imp.groupby([dt.source_feature(c) for c in imp.index]).sum().sort_values(ascending=False)
    used = int((imp > 0).sum())
    return [(f"feature importance (share of the weighted impurity decrease across all stumps); {used} of {len(imp)} "
            f"encoded columns are used by at least one stump"),
            "by input column: " + ", ".join(f"{c} {v:.3f}" for c, v in by_source.items() if v > 0),
            f"top {n} encoded columns:", imp.sort_values(ascending=False).head(n).to_string(float_format="{:.4f}".format)]


def module3_tree(tree: tuple, X, y, train, test) -> dict:
    name, criterion, depth, published = tree
    res = evaluate(dt.make_model(name, criterion, depth).fit(X.iloc[train], y.iloc[train]), X, y, train, test)
    assert round(res["accuracy"], 4) == published, (name, res["accuracy"])
    return res


def load_curves() -> dict[str, pd.DataFrame]:
    """The saved CV curves (results/adaboost_cv_curve.csv), one frame per feature set."""
    curves = pd.read_csv(CURVE_PATH)
    return {label: c.drop(columns="features").reset_index(drop=True) for label, c in curves.groupby("features", sort=False)}


def grid_search(df, y, train) -> dict[str, pd.DataFrame]:
    """The full CV grid for both feature sets (about four and a half minutes); main() saves it to CURVE_PATH."""
    return {TABULAR: staged_cv(tabular_model, dt.tabular_frame(df), y, train, TABULAR_RATES, max(TABULAR_ROUNDS)),
            TEXT: staged_cv(text_model, nb.build_variants(df)["masked"], y, train, TEXT_RATES, max(TEXT_ROUNDS))}


def fit_tabular(df, y, train, test, curve: pd.DataFrame) -> dict:
    """The selected tabular booster refitted on the training rows, and tree 1 beside it."""
    X = dt.tabular_frame(df)
    rate, rounds = select(curve, TABULAR_ROUNDS)
    model = tabular_model(rounds, rate).fit(X.iloc[train], y.iloc[train])
    picked = curve[(curve["learning_rate"] == rate) & (curve["n_estimators"] == rounds)]["cv_mean"].iloc[0]
    return {"X": X, "rate": rate, "rounds": rounds, "picked": picked, "model": model,
            "ada": evaluate(model, X, y, train, test), "tree1": module3_tree(TREE1, X, y, train, test)}


def setup_lines() -> list[str]:
    return ["features: tree 1's nine columns " + ", ".join(dt.TREES["tree1"][1]) + "; encoded by role_dt.tabular_encoder "
            "inside the pipeline, so every CV fold refits it",
            (f"base estimator: decision stump; grid learning_rate {TABULAR_RATES} x n_estimators {TABULAR_ROUNDS}, "
             f"stratified 5-fold CV on the 927 training rows, accuracy")]


def selection_line(fit: dict) -> str:
    return (f"SELECTED: learning_rate {fit['rate']:g}, n_estimators {fit['rounds']} (CV accuracy {fit['picked']:.4f}); "
            f"final model fitted on all 927 training rows with {len(fit['model'].named_steps['ada'].estimators_)} stumps")


def tree_fold_models(fit: dict) -> dict:
    return {"AdaBoost": tabular_model(fit["rounds"], fit["rate"]), "tree 1": dt.make_model(*TREE1[:3])}


def run_tabular(df, y, train, test, curves: list) -> list[str]:
    curve = staged_cv(tabular_model, dt.tabular_frame(df), y, train, TABULAR_RATES, max(TABULAR_ROUNDS))
    curves.append(curve.assign(features=TABULAR))
    fit = fit_tabular(df, y, train, test, curve)
    return [*setup_lines(), "", *surface_lines(curve, TABULAR_ROUNDS), "", selection_line(fit),
            "", "TEST SET (232 postings)", *result_lines("AdaBoost (tabular)", fit["ada"]), "",
            *importance_lines(fit["model"]),
            "", "=" * 100, "3. ADABOOST AGAINST TREE 1 (gini, no depth limit, all tabular features)",
            *result_lines("tree 1 (refitted; matches the published 0.9224)", fit["tree1"]), "",
            *comparison_lines({"AdaBoost": fit["ada"], "tree 1": fit["tree1"]}, y.iloc[test]), "",
            *fold_lines(tree_fold_models(fit), fit["X"], y, train)]


def fold_lines(models: dict, X, y: pd.Series, train: np.ndarray) -> list[str]:
    """The same comparison on the five Module 3 training folds, fold by fold."""
    folds = StratifiedKFold(nb.CV_FOLDS, shuffle=True, random_state=nb.SPLIT_SEED)
    scores = {name: cross_val_score(m, X.iloc[train], y.iloc[train], cv=folds, scoring="accuracy")
              for name, m in models.items()}
    (a, sa), (b, sb) = scores.items()
    return ["the same two models on the five Module 3 training folds (CV accuracy; both settings were chosen on "
            "these folds): " + "; ".join(f"{n} {s.mean():.4f} +/- {s.std():.4f}" for n, s in scores.items()),
            f"  fold by fold ({a} minus {b}): " + ", ".join(f"{d:+.4f}" for d in sa - sb)
            + f"; {a} ahead in {int((sa > sb).sum())} of {len(sa)} folds"]


def fit_text(df, y, train, test, curve: pd.DataFrame) -> dict:
    """The selected masked-text booster, tree 4 and the two masked-text Naive Bayes models."""
    texts = nb.build_variants(df)["masked"]
    rate, rounds = select(curve, TEXT_ROUNDS)
    ada = evaluate(text_model(rounds, rate).fit(texts.iloc[train], y.iloc[train]), texts, y, train, test)
    results = {"AdaBoost (text)": ada, "tree 4": module3_tree(TREE4, texts, y, train, test)}
    for name, (cls, binary, alpha, published) in NB_MASKED.items():
        model = Pipeline([("features", nb.make_vectorizer(binary)), ("nb", cls(alpha=alpha))])
        res = evaluate(model.fit(texts.iloc[train], y.iloc[train]), texts, y, train, test)
        assert round(res["accuracy"], 4) == published, (name, res["accuracy"])
        results[f"{name} (alpha {alpha:g})"] = res
    return {"rate": rate, "rounds": rounds, "ada": ada, "results": results}


def text_setup_line() -> str:
    return (f"smaller grid to keep it cheap: learning_rate {TEXT_RATES} x n_estimators {TEXT_ROUNDS}; the "
            f"vectoriser is refitted inside every fold")


def run_text(df, y, train, test, curves: list) -> list[str]:
    curve = staged_cv(text_model, nb.build_variants(df)["masked"], y, train, TEXT_RATES, max(TEXT_ROUNDS))
    curves.append(curve.assign(features=TEXT))
    fit = fit_text(df, y, train, test, curve)
    return [text_setup_line(), "", *surface_lines(curve, TEXT_ROUNDS), "",
            f"SELECTED: learning_rate {fit['rate']:g}, n_estimators {fit['rounds']}", "",
            *result_lines("AdaBoost (masked text)", fit["ada"]), "",
            ("against tree 4 (gini, depth 12) and the Module 3 masked-text Naive Bayes models, all refitted and "
             "matching their published accuracy:"), *comparison_lines(fit["results"], y.iloc[test])]


# ---------------------------------------------------------------- figures

def rate_colors(rates) -> dict[float, tuple]:
    """One step of a light-to-dark ramp of the project indigo per learning rate (small light, large dark)."""
    rates = sorted(set(TABULAR_RATES) | set(rates))
    ramp = LinearSegmentedColormap.from_list("rates", ["#B9B8E4", PALETTE[0], "#1E1D52"])
    return {r: ramp(i / (len(rates) - 1)) for i, r in enumerate(rates)}


def plot_cv_curve(curves: dict[str, pd.DataFrame], selected: dict[str, tuple[float, int]],
                  reference: dict[str, tuple[str, float]] | None = None, points: int = 90):
    """CV accuracy against rounds, one line per learning rate, a panel per feature set.

    Lines are drawn at about `points` log-spaced round counts (every round is in the curve file).
    The selected learning rate also gets its training-fold accuracy (dashed), so the gap and the
    underfitting show; reference draws a dotted horizontal line, such as tree 1's CV accuracy.
    """
    colors = rate_colors([r for c in curves.values() for r in c["learning_rate"].unique()])
    fig, axes = plt.subplots(1, len(curves), figsize=(11, 5.6), sharey=True)
    callout = {TABULAR: (12, 0.965), TEXT: (60, 0.80)}
    for ax, (label, curve) in zip(np.atleast_1d(axes), curves.items()):
        rate, rounds = selected[label]
        keep = set(np.unique(np.geomspace(1, curve["n_estimators"].max(), points).round().astype(int))) | {rounds}
        shown = curve[curve["n_estimators"].isin(keep)]
        for r, c in shown.groupby("learning_rate"):
            ax.plot(c["n_estimators"], c["cv_mean"], color=colors[r], lw=2.4 if r == rate else 1.6)
        sel = shown[shown["learning_rate"] == rate]
        ax.plot(sel["n_estimators"], sel["cv_train_mean"], color=colors[rate], lw=1.6, ls="--")
        point = sel[sel["n_estimators"] == rounds].iloc[0]
        ax.scatter([rounds], [point["cv_mean"]], s=150, marker="*", color=TEXT_COLOR, zorder=5)
        ax.annotate(f"selected: rate {rate:g}, {rounds} rounds\nCV accuracy {point['cv_mean']:.3f}",
                    xy=(rounds, point["cv_mean"]), xytext=callout.get(label, (10, 0.95)), fontsize=9.5,
                    color=TEXT_COLOR, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
        if reference and label in reference:
            name, value = reference[label]
            ax.axhline(value, color=MUTED, ls=":", lw=1.3)
            ax.text(1.15, value + 0.005, f"{name} {value:.3f}", fontsize=9, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlim(1, curve["n_estimators"].max())
        ax.set_xlabel("Boosting rounds (stumps), log scale")
        ax.set_title(f"{label}: learning rates {', '.join(f'{r:g}' for r in sorted(curve['learning_rate'].unique()))}",
                     fontsize=11)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy, mean over the 5 CV folds")
    handles = [plt.Line2D([], [], color=colors[r], lw=2.2, label=f"{r:g}") for r in sorted(colors)]
    handles += [plt.Line2D([], [], color=MUTED, lw=1.6, label="solid: CV folds"),
                plt.Line2D([], [], color=MUTED, lw=1.6, ls="--", label="dashed: training folds (selected rate)"),
                plt.Line2D([], [], marker="*", ls="", color=TEXT_COLOR, ms=12, label="selected")]
    fig.legend(handles=handles, title="learning rate", loc="lower center", ncols=len(handles), frameon=False,
               fontsize=9, title_fontsize=9.5, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0, handlelength=1.8)
    fig.suptitle("AdaBoost: cross-validated accuracy as rounds are added (927 training postings)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def plot_confusion(res: dict, title: str, notes: list[str]):
    """One confusion matrix in the Module 3 style, FD/Sol/SWE row order (regression_models.plot_confusion)."""
    return rm.plot_confusion(res["confusion"].rename(index=SHORT, columns=SHORT), title, notes)


def save_curves(curves: dict[str, pd.DataFrame]) -> None:
    """Write grid_search()'s curves to CURVE_PATH in the format main() writes."""
    pd.concat(c.assign(features=label) for label, c in curves.items())[
        ["features", "learning_rate", "n_estimators", "cv_mean", "cv_sd", "cv_train_mean"]].to_csv(
        CURVE_PATH, index=False, float_format="%.6f")


def plot_vs_tree(ada: dict, tree: dict, y_test: pd.Series):
    """AdaBoost against tree 1: accuracy and recall per class, and the postings only one gets right."""
    names, colors = ["tree 1", "AdaBoost"], {"tree 1": PALETTE[0], "AdaBoost": PALETTE[3]}
    res = {"tree 1": tree, "AdaBoost": ada}
    metrics = ["accuracy", *[f"recall {SHORT[c]}" for c in CLASSES]]
    values = {n: [r["accuracy"], *[r["per_class"].loc[c, "recall"] for c in CLASSES]] for n, r in res.items()}
    yv = y_test.to_numpy()
    right = {n: r["pred"] == yv for n, r in res.items()}
    only = {n: [int((right[n] & ~right[o] & (yv == c)).sum()) for c in CLASSES]
            for n, o in [("tree 1", "AdaBoost"), ("AdaBoost", "tree 1")]}
    a, b = sum(only["tree 1"]), sum(only["AdaBoost"])
    p = binomtest(min(a, b), a + b).pvalue

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.35, 1]})
    x, w = np.arange(len(metrics)), 0.36
    for j, n in enumerate(names):
        bars = a1.bar(x + (j - 0.5) * w, values[n], w, color=colors[n], label=n, zorder=3)
        for bar, v in zip(bars, values[n]):
            a1.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=9,
                    color=TEXT_COLOR)
    a1.set_xticks(x, metrics)
    a1.set_ylim(0, 1.12)
    a1.set_ylabel(f"Test-set score ({len(yv)} postings)")
    a1.set_title("Accuracy and recall by role family", fontsize=11.5)
    a1.legend(frameon=False, loc="upper left", ncols=2)
    a1.grid(axis="x", visible=False)

    x2 = np.arange(len(CLASSES))
    for j, n in enumerate(names):
        bars = a2.bar(x2 + (j - 0.5) * w, only[n], w, color=colors[n], label=f"right only by {n}", zorder=3)
        for bar, v in zip(bars, only[n]):
            a2.text(bar.get_x() + bar.get_width() / 2, v + 0.25, str(v), ha="center", fontsize=10, color=TEXT_COLOR)
    a2.set_xticks(x2, [SHORT[c] for c in CLASSES])
    a2.yaxis.set_major_locator(MaxNLocator(integer=True))
    a2.set_ylim(0, max(max(v) for v in only.values()) * 1.3 + 1)
    a2.set_ylabel("Test postings")
    a2.set_title(f"Postings only one model gets right: tree 1 {a}, AdaBoost {b}\n"
                 f"net {a - b} postings ({100 * (a - b) / len(yv):.1f} points), exact McNemar p = {p:.3f}",
                 fontsize=11)
    a2.legend(frameon=False, loc="upper right")
    a2.grid(axis="x", visible=False)
    fig.suptitle(f"AdaBoost against the Part 3 decision tree (tree 1): same tabular features, same {len(yv)} "
                 "test postings")
    fig.tight_layout()
    return fig


def main() -> None:
    df, _ = nb.load_subset()
    y = df["role_family"]
    train, test = nb.split_rows(y)
    curves = []
    report = ["ADABOOST: role family (forward_deployed / solutions / software_eng)",
              "Module 3 subset (1,159 postings), Module 3 split (927 training, 232 test); fitted on training rows "
              "only, scored once on the test rows. Never used as features: "
              + ", ".join(c for c in dt.EXCLUDED if c != "role_family") + ".",
              "", "=" * 100, "1. MULTI-ZONE RESIDUAL CHECK (linear regression, all-features fit)",
              *rm.multizone_residual_lines(rm.salary_frame(df)),
              "", "=" * 100, "2. ADABOOST ON TREE 1'S TABULAR FEATURES", *run_tabular(df, y, train, test, curves),
              "", "=" * 100, "4. ADABOOST ON THE MASKED TEXT (second comparison)", *run_text(df, y, train, test, curves)]
    REPORT_PATH.write_text("\n".join(report) + "\n")
    pd.concat(curves)[["features", "learning_rate", "n_estimators", "cv_mean", "cv_sd", "cv_train_mean"]].to_csv(
        CURVE_PATH, index=False, float_format="%.6f")
    print("\n".join(report))
    print(f"\nsaved {REPORT_PATH.relative_to(nb.tc.ROOT)} and {CURVE_PATH.relative_to(nb.tc.ROOT)}")


if __name__ == "__main__":
    main()
