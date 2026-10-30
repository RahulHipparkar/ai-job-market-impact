"""Naive Bayes on posting text: forward_deployed vs solutions vs software_eng.

The role label is parsed from the title, and descriptions repeat the same role phrases,
so a model on the unmasked text partly re-runs the labelling rule. Three inputs are
compared on one shared split: raw, company boilerplate stripped (the Module 2 rule), and
stripped with the posting's own title and the multi-word role phrases masked out.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline

try:  # imported as src.text_nb from the notebook, or as text_nb from src/ scripts
    from src import text_clusters as tc
    from src.plotstyle import CMAP, PALETTE
except ImportError:
    import text_clusters as tc
    from plotstyle import CMAP, PALETTE

CLASSES = ["forward_deployed", "solutions", "software_eng"]
REPORT_PATH = tc.ROOT / "results" / "nb_report.txt"
TEST_SIZE, SPLIT_SEED = 0.2, 42
# alpha is chosen by stratified CV on the training rows; the test-set sweep is sensitivity only
CV_ALPHAS, CV_FOLDS = [0.01, 0.05, 0.1, 0.5, 1.0], 5
TEST_SWEEP_ALPHAS = [0.1, 0.5, 1.0]
VARIANTS = ["raw", "stripped", "masked"]
LABEL = {"raw": "raw descriptions", "stripped": "company boilerplate removed",
         "masked": "boilerplate removed, role phrases masked"}
# The Module 2 parameters, but every token must contain a letter: pure numbers (employer
# pay bands such as "$182,000-$250,208") are dropped. Otherwise the default tokeniser
# (\b\w\w+\b), so hyphenated words still split as before.
VECTOR_PARAMS = {**tc.TFIDF_PARAMS, "token_pattern": r"(?u)\b(?=\w*[^\W\d_])\w\w+\b"}
# MultinomialNB on TF-IDF weights; BernoulliNB on word presence (binary counts)
MODELS = {"MultinomialNB": (MultinomialNB, False), "BernoulliNB": (BernoulliNB, True)}

ROLE_PHRASES = ["forward deployed", "forward-deployed", "fde", "solutions architect",
                "solutions engineer", "solution architect", "field engineer", "field engineering",
                "software engineer", "software engineering", "deployment strategist",
                "deployment engineer",
                # Palantir's abbreviation for the forward-deployed title
                "fdse", "fdses", "forward deployed software engineer",
                "forward deployed software engineers",
                # Databricks' abbreviations for the (delivery / specialist) solutions architect title
                "sa", "sas", "dsa", "dsas", "delivery solutions architect",
                "delivery solutions architects", "ssa", "ssas"]
# Floors for the lead list (training rows only). LEAD_MIN_CLASS_DF: postings of the class
# containing the word. LEAD_MIN_COMPANIES: distinct companies among those postings; one
# employer's template text is exclusive to its class by construction, so a word must be
# used by several employers' postings of the class to count as a cue for the role.
LEAD_MIN_CLASS_DF, LEAD_MIN_COMPANIES = 10, 3
# Whole phrases only, with an optional plural "s"; a space or hyphen may join the words.
# The single words "solutions", "engineer" and "deployed" are left in place, and short
# entries such as "sa" match only as whole tokens, never inside a word.
def phrase_pattern(phrase: str) -> str:
    return re.sub(r"[\s-]+", r"[\\s-]+", phrase)


PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(map(phrase_pattern, sorted(ROLE_PHRASES, key=len, reverse=True)))
    + r")s?\b", re.I)


def load_subset() -> tuple[pd.DataFrame, list[str]]:
    """The three-class postings, with the Module 2 stripped text alongside.

    Boilerplate is found across all 3,718 deduplicated postings, exactly as in Module 2,
    and only then are the three classes kept.
    """
    postings, log = tc.load_postings()
    stripped, _ = tc.strip_boilerplate(postings)
    keep = postings["role_family"].isin(CLASSES)
    df = postings.loc[keep].copy()
    df["stripped"] = stripped[keep]
    return df.reset_index(drop=True), log


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def mask_role_phrases(text: str, title: str) -> str:
    """Remove the posting's own title and the role phrases, case-insensitively."""
    if isinstance(title, str) and title.strip():
        text = re.sub(r"\s+".join(map(re.escape, title.split())), " ", text, flags=re.I)
    return collapse(PHRASE_RE.sub(" ", text))


def build_variants(df: pd.DataFrame) -> dict[str, pd.Series]:
    """raw, stripped and masked text for the same rows, aligned to df.index."""
    masked = pd.Series([mask_role_phrases(t, ti) for t, ti in zip(df["stripped"], df["title"])],
                       index=df.index)
    return {"raw": df["description_clean"], "stripped": df["stripped"], "masked": masked}


def masking_summary(stripped: pd.Series, masked: pd.Series) -> dict[str, float]:
    """Rows the mask changed and characters it removed, against whitespace-collapsed input."""
    before = stripped.map(collapse)
    removed = before.str.len() - masked.str.len()
    changed = before != masked
    return {"rows_changed": int(changed.sum()), "rows": len(masked),
            "median_chars_removed_changed_rows": float(removed[changed].median()),
            "median_chars_removed_all_rows": float(removed.median()),
            "median_pct_removed_changed_rows": float((100 * removed / before.str.len())[changed].median())}


def phrase_hits(texts: pd.Series, rows: pd.DataFrame, phrases=ROLE_PHRASES) -> pd.DataFrame:
    """Postings containing each phrase exactly (whole tokens, no plural), by class and company."""
    out = []
    for p in phrases:
        hit = texts.str.contains(re.compile(rf"\b{phrase_pattern(p)}\b", re.I))
        by_class = rows.loc[hit, "role_family"].value_counts().reindex(CLASSES, fill_value=0)
        companies = rows.loc[hit, "company_clean"].value_counts()
        out.append({"phrase": p, "postings": int(hit.sum()), **by_class.to_dict(),
                    "companies": ", ".join(f"{c} {n}" for c, n in companies.items())})
    return pd.DataFrame(out).set_index("phrase")


def split_rows(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """One stratified 80/20 split of row positions, shared by every variant and model."""
    return train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y,
                            random_state=SPLIT_SEED)


def make_vectorizer(binary: bool = False):
    """TF-IDF, or binary word presence, with VECTOR_PARAMS."""
    return (CountVectorizer(binary=True, **VECTOR_PARAMS) if binary
            else TfidfVectorizer(**VECTOR_PARAMS))


def vectorize(texts: pd.Series, train: np.ndarray, test: np.ndarray, binary: bool = False):
    """Fit on the training rows only; the test rows are transformed with that vocabulary."""
    vec = make_vectorizer(binary)
    X_train = vec.fit_transform(texts.iloc[train])
    return X_train, vec.transform(texts.iloc[test]), vec


def cv_alpha(texts: pd.Series, y: pd.Series, train: np.ndarray,
             model: str) -> tuple[pd.DataFrame, float]:
    """Stratified k-fold CV accuracy for each alpha, on the training rows only.

    The vectoriser sits inside the pipeline, so every fold refits it on its own training
    part. The highest mean accuracy wins; a tie goes to the larger alpha.
    """
    cls, binary = MODELS[model]
    pipe = Pipeline([("vec", make_vectorizer(binary)), ("nb", cls())])
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    search = GridSearchCV(pipe, {"nb__alpha": CV_ALPHAS}, cv=folds, scoring="accuracy",
                          refit=False).fit(texts.iloc[train], y.iloc[train])
    res = search.cv_results_
    table = pd.DataFrame({"cv_mean": res["mean_test_score"], "cv_sd": res["std_test_score"]},
                         index=pd.Index(res["param_nb__alpha"].astype(float), name="alpha"))
    best = table.index[np.isclose(table["cv_mean"], table["cv_mean"].max())].max()
    return table, float(best)


def majority_baseline(y: pd.Series, train: np.ndarray, test: np.ndarray) -> tuple[str, float]:
    """The most common training class, and its accuracy when predicted for every test row."""
    majority = y.iloc[train].value_counts().idxmax()
    return majority, float((y.iloc[test] == majority).mean())


def evaluate(model, X_test, y_test: pd.Series) -> dict:
    """Accuracy, labelled confusion matrix (rows true, columns predicted) and per-class report."""
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    return {"accuracy": accuracy_score(y_test, pred),
            "confusion": pd.DataFrame(cm, index=pd.Index(CLASSES, name="true"),
                                      columns=pd.Index(CLASSES, name="predicted")),
            "report": classification_report(y_test, pred, labels=CLASSES, digits=3,
                                             zero_division=0),
            "pred": pred}


def run_variant(texts: pd.Series, y: pd.Series, train: np.ndarray, test: np.ndarray) -> dict:
    """Both NB models on one text variant.

    Headline results use the CV-selected alpha. Also kept: the alpha = 1.0 (sklearn
    default) results, and test accuracy over TEST_SWEEP_ALPHAS as a sensitivity check.
    """
    out = {}
    for name, (cls, binary) in MODELS.items():
        X_train, X_test, vec = vectorize(texts, train, test, binary)
        y_train, y_test = y.iloc[train], y.iloc[test]
        cv_table, alpha = cv_alpha(texts, y, train, name)
        model = cls(alpha=alpha).fit(X_train, y_train)
        res = evaluate(model, X_test, y_test)
        res["default"] = evaluate(cls(alpha=1.0).fit(X_train, y_train), X_test, y_test)
        res["test_sweep"] = {a: accuracy_score(y_test, cls(alpha=a).fit(X_train, y_train).predict(X_test))
                             for a in TEST_SWEEP_ALPHAS}
        res.update(alpha=alpha, cv=cv_table, model=model, vectorizer=vec, X_test=X_test,
                   train_shape=X_train.shape, test_shape=X_test.shape)
        out[name] = res
    return out


def top_words(model, vec, n: int = 15) -> dict[str, pd.Series]:
    """Highest log P(word | class) for each class."""
    terms = vec.get_feature_names_out()
    pos = {c: i for i, c in enumerate(model.classes_)}
    return {c: pd.Series(model.feature_log_prob_[pos[c]], index=terms).nlargest(n) for c in CLASSES}


def class_doc_freq(vec, texts: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Postings of each class containing each vocabulary term (terms x classes)."""
    present = (vec.transform(texts) > 0).astype(int)
    return pd.DataFrame({c: np.asarray(present[(y == c).to_numpy()].sum(axis=0)).ravel()
                         for c in CLASSES}, index=vec.get_feature_names_out())


def class_company_count(vec, texts: pd.Series, y: pd.Series, companies: pd.Series) -> pd.DataFrame:
    """Distinct companies whose postings of each class contain each term (terms x classes)."""
    present = (vec.transform(texts) > 0).astype(int)
    out = {}
    for c in CLASSES:
        in_class = (y == c).to_numpy()
        per_company = [np.asarray(present[in_class & (companies == co).to_numpy()].sum(axis=0)).ravel() > 0
                       for co in companies[in_class].unique()]
        out[c] = np.sum(per_company, axis=0)
    return pd.DataFrame(out, index=vec.get_feature_names_out())


def lead_table(model, vec, doc_freq: pd.DataFrame, company_count: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every term's log-probability lead over the next most likely class, per class,
    with its within-class document frequency and company count (training rows)."""
    terms = vec.get_feature_names_out()
    lp = model.feature_log_prob_
    out = {c: pd.DataFrame({"lead": lp[i] - np.delete(lp, i, axis=0).max(axis=0),
                            "class_df": doc_freq.loc[terms, c].values,
                            "companies": company_count.loc[terms, c].values}, index=terms)
           for i, c in enumerate(model.classes_)}
    return {c: out[c] for c in CLASSES}


def distinctive_words(leads: dict[str, pd.DataFrame], min_class_df: int = 0,
                      min_companies: int = 0, n: int = 15) -> dict[str, pd.DataFrame]:
    """The n largest leads per class among words meeting both floors."""
    return {c: t[(t["class_df"] >= min_class_df) & (t["companies"] >= min_companies)].nlargest(n, "lead")
            for c, t in leads.items()}


def confident_errors(model, X_test, rows: pd.DataFrame, texts: pd.Series, n: int = 10,
                     chars: int = 150) -> pd.DataFrame:
    """The n misclassified test postings with the highest predicted-class probability."""
    proba = model.predict_proba(X_test)
    out = rows[["jobId", "company_clean", "title", "role_family"]].rename(
        columns={"role_family": "true"}).reset_index(drop=True)
    out["predicted"] = model.classes_[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1)
    out["text"] = texts.str.slice(0, chars).values
    return out[out["true"] != out["predicted"]].nlargest(n, "confidence").reset_index(drop=True)


def evaluation_lines(res: dict) -> list[str]:
    """Confusion matrix and classification report as indented lines."""
    return ["confusion matrix (rows = true, columns = predicted):",
            *[f"  {line}" for line in res["confusion"].to_string().splitlines()],
            "classification report:",
            *[f"  {line}" for line in res["report"].rstrip().splitlines()]]


def result_lines(variant: str, model: str, res: dict, baseline: tuple[str, float]) -> list[str]:
    """Every number for one variant x model, headline (CV-selected alpha) first."""
    return [f"--- {variant} ({LABEL[variant]}) x {model} ---",
            (f"features: train {res['train_shape'][0]} x {res['train_shape'][1]}, "
             f"test {res['test_shape'][0]} x {res['test_shape'][1]}"),
            f"{CV_FOLDS}-fold stratified CV on the training rows (accuracy, mean +/- sd over folds):",
            *[f"  alpha {a:<5}: {r.cv_mean:.4f} +/- {r.cv_sd:.4f}" for a, r in res["cv"].iterrows()],
            f"selected alpha: {res['alpha']}",
            f"TEST ACCURACY at selected alpha {res['alpha']}: {res['accuracy']:.4f}",
            f"majority-class baseline ({baseline[0]}): {baseline[1]:.4f}",
            *evaluation_lines(res),
            f"for comparison, alpha = 1.0 (sklearn default): test accuracy {res['default']['accuracy']:.4f}",
            *[f"  {line}" for line in evaluation_lines(res["default"])]]


def company_accuracy(res: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Test accuracy per company, with its test rows by true class."""
    t = pd.DataFrame({"company": rows["company_clean"].values, "true": rows["role_family"].values,
                      "correct": res["pred"] == rows["role_family"].values})
    out = pd.crosstab(t["company"], t["true"]).reindex(columns=CLASSES, fill_value=0)
    out.insert(0, "n_test", out.sum(axis=1))
    out["accuracy"] = t.groupby("company")["correct"].mean()
    return out.sort_values("n_test", ascending=False)


def token_postings(texts: pd.Series, rows: pd.DataFrame, token: str,
                   n_examples: int = 3, window: int = 50) -> tuple[pd.Series, list[str]]:
    """Which postings contain a vectoriser token, and example snippets around it."""
    rx = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")
    hit = texts.str.contains(rx)
    examples = []
    for i in texts.index[hit][:n_examples]:
        m = rx.search(texts[i])
        examples.append(f"[{rows.at[i, 'company_clean']} | {rows.at[i, 'role_family']} | "
                        f"{rows.at[i, 'title']}] ...{texts[i][max(0, m.start() - window):m.end() + window]}...")
    return hit, examples


def _short(label: str) -> str:
    return label.replace("_", " ")


def plot_confusion(results: dict, label: str):
    """Counts and row-normalised confusion matrices, one row of panels per NB model."""
    fig, axes = plt.subplots(len(results), 2, figsize=(12.5, 4.6 * len(results)),
                             gridspec_kw={"wspace": 0.45})
    ticks = [_short(c).replace(" ", "\n") for c in CLASSES]
    for row, (name, res) in zip(axes, results.items()):
        cm = res["confusion"]
        pct = cm.div(cm.sum(axis=1), axis=0) * 100
        for ax, data, fmt, title, cbar in [
                (row[0], cm, "d", "counts", "Postings"),
                (row[1], pct, ".1f", "row %  (recall on the diagonal)", "% of the true class")]:
            sns.heatmap(data, annot=True, fmt=fmt, cmap=CMAP, linewidths=0.5, linecolor="white",
                        cbar_kws={"label": cbar}, annot_kws={"size": 10}, ax=ax,
                        xticklabels=ticks, yticklabels=ticks)
            ax.set_title(f"{name}, alpha = {res['alpha']}: {title}\n(accuracy {res['accuracy']:.1%})",
                         fontsize=11)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.grid(False)
            plt.setp(ax.get_yticklabels(), rotation=0)
            plt.setp(ax.get_xticklabels(), rotation=0)
    fig.suptitle(f"Naive Bayes confusion matrices, test set (n = {int(cm.values.sum())}), {label}\n"
                 f"alpha chosen by {CV_FOLDS}-fold CV on the training set")
    fig.subplots_adjust(hspace=0.45)
    return fig


def plot_accuracy(acc: pd.DataFrame, alphas: pd.DataFrame, baseline: tuple[str, float]):
    """Test accuracy per text variant (rows of acc) for each NB model (columns), at the
    CV-selected alpha shown under each value."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(acc.index))
    width = 0.36
    for j, model in enumerate(acc.columns):
        offset = (j - (len(acc.columns) - 1) / 2) * (width + 0.02)
        bars = ax.bar(x + offset, acc[model], width, color=PALETTE[j], label=model, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.1%}\nα = {a}" for v, a in zip(acc[model], alphas[model])],
                     padding=3, fontsize=9, color="#1C1C1A")
    ax.axhline(baseline[1], color="#8A8A84", ls="--", lw=1.2, zorder=4,
               label=f"Majority-class baseline, always {_short(baseline[0])} ({baseline[1]:.1%})")
    ax.set_xticks(x, [LABEL[v].replace(", ", ",\n") for v in acc.index])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Test-set accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Naive Bayes test accuracy by text variant (forward_deployed / solutions / software_eng)\n"
                 f"alpha chosen by {CV_FOLDS}-fold CV on the training set", fontsize=12)
    fig.tight_layout()
    return fig


def plot_top_words(words: dict[str, pd.Series], label: str, alpha: float):
    """Top words per class by P(word | class) for MultinomialNB, one panel per class."""
    fig, axes = plt.subplots(1, len(words), figsize=(13, 5.5), sharex=True)
    for ax, (c, s), color in zip(axes, words.items(), PALETTE):
        p = np.exp(s[::-1])
        ax.barh(p.index, p.values, color=color, height=0.7, zorder=3)
        ax.set_title(_short(c))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("P(word | class)")
    fig.suptitle(f"MultinomialNB (alpha = {alpha}): 15 most probable words per class ({label})")
    fig.tight_layout()
    return fig


def plot_lead_words(lead: dict[str, pd.DataFrame], label: str, alpha: float, min_companies: int):
    """Largest log-probability lead per class; each word labelled with its within-class
    posting count and company count."""
    fig, axes = plt.subplots(1, len(lead), figsize=(13.5, 5.5), sharex=True)
    for ax, (c, t), color in zip(axes, lead.items(), PALETTE):
        t = t[::-1]
        ax.barh([f"{w} ({d}, {k})" for w, d, k in zip(t.index, t["class_df"], t["companies"])],
                t["lead"], color=color, height=0.7, zorder=3)
        ax.set_title(_short(c))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("log P(word | class) minus the next-highest class")
    fig.suptitle(f"MultinomialNB (alpha = {alpha}): words that most favour each class ({label})\n"
                 f"only words used by at least {min_companies} companies' training postings of the class; "
                 "(postings, companies) in brackets")
    fig.tight_layout()
    return fig
