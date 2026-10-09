"""TF-IDF construction, k-means / hierarchical clustering and SVD helpers for the ATS postings.

Two text inputs are supported: `description_clean` as-is ("raw"), and the same text
with per-company boilerplate removed ("stripped"). On the raw text, sentences every
posting from one employer shares (mission statements, benefits, EEO notices) dominate
the TF-IDF weights, so clusters track the employer rather than the role.
"""

import itertools
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, cut_tree, linkage, to_tree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_samples, silhouette_score)

try:  # imported as src.text_clusters from the notebook, or as text_clusters from src/ scripts
    from src.clean_ats_stage2 import normalize_title
except ImportError:
    from clean_ats_stage2 import normalize_title

ROOT = Path(__file__).resolve().parent.parent
ATS_FINAL = ROOT / "data" / "processed" / "ats_final.csv"
CLUSTERS_CSV = ROOT / "data" / "processed" / "ats_clusters.csv"
REPORT_PATH = ROOT / "results" / "clustering_report.txt"

TFIDF_PARAMS = dict(min_df=5, max_df=0.8, stop_words="english")
RANDOM_STATE = 0
# A sentence is boilerplate when it recurs in at least this share of one company's
# postings, and in at least BOILERPLATE_MIN of them (so tiny employers are not gutted).
BOILERPLATE_SHARE, BOILERPLATE_MIN = 0.10, 3
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def dedup_key(df: pd.DataFrame) -> pd.Series:
    """The key is_near_dup was built from in clean_ats_stage2.py: company, normalised
    title and the first 300 characters of the description."""
    return (df["company_clean"].astype(str) + "|" + df["title"].map(normalize_title) + "|"
            + df["description_clean"].fillna("").str.slice(0, 300))


def load_postings(path: Path = ATS_FINAL) -> tuple[pd.DataFrame, list[str]]:
    """Read ats_final.csv, keep the first copy of each near-duplicate group, drop junk rows.

    is_near_dup flags every posting in a repeated group, the first copy included, so
    dropping the flagged rows would remove those jobs entirely. Keeping the first copy
    on the same key removes only the extra copies.
    """
    df = pd.read_csv(path)
    log = [f"rows before dedup: {len(df)}"]
    key = dedup_key(df)
    group_size = key.map(key.value_counts())
    # the key must reproduce the stored flag exactly, or the two definitions have drifted
    assert (group_size > 1).equals(df["is_near_dup"]), "dedup key does not match is_near_dup"
    groups = key[group_size > 1].value_counts()
    log.append(f"near-duplicate groups: {len(groups)} ({int(groups.sum())} rows flagged is_near_dup)")
    log.append("group size distribution (copies per group: number of groups):")
    log += [f"  {size}: {n}" for size, n in groups.value_counts().sort_index().items()]
    keep = ~key.duplicated(keep="first")
    log.append(f"removed as extra copies: {int((~keep).sum())}")
    df = df.loc[keep]
    log.append(f"rows after dedup: {len(df)}")
    if "is_junk" in df.columns:
        n_junk = int(df["is_junk"].sum())
        df = df[~df["is_junk"]]
        log.append(f"dropped is_junk: {n_junk} -> {len(df)} rows")
    else:
        log.append("is_junk: column not present in ats_final.csv, nothing to drop")
    return df.reset_index(drop=True), log


def split_sentences(text: str) -> list[str]:
    """Split a description on sentence-ending punctuation and line breaks."""
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def strip_boilerplate(df: pd.DataFrame, text_col: str = "description_clean",
                      company_col: str = "company_clean",
                      share: float = BOILERPLATE_SHARE,
                      min_count: int = BOILERPLATE_MIN) -> tuple[pd.Series, pd.DataFrame]:
    """Drop sentences repeated across a company's postings, then remove company names.

    Returns the stripped text (aligned to df.index) and a per-company summary.
    """
    sentences = df[text_col].map(split_sentences)
    stripped = pd.Series(index=df.index, dtype=object)
    rows = []
    for company, idx in df.groupby(company_col).groups.items():
        counts = Counter(s for i in idx for s in set(sentences[i]))
        threshold = max(min_count, share * len(idx))
        boiler = {s for s, c in counts.items() if c >= threshold}
        for i in idx:
            stripped[i] = " ".join(s for s in sentences[i] if s not in boiler)
        before = df.loc[idx, text_col].str.len()
        after = stripped[idx].str.len()
        rows.append({"company": company, "postings": len(idx),
                     "boilerplate_sentences": len(boiler),
                     "median_chars_before": int(before.median()),
                     "median_chars_after": int(after.median()),
                     "pct_chars_removed": round(100 * (1 - after.sum() / before.sum()), 1)})
    stripped = remove_company_names(stripped, df[company_col].unique())
    summary = pd.DataFrame(rows).sort_values("postings", ascending=False).reset_index(drop=True)
    return stripped, summary


def remove_company_names(texts: pd.Series, companies) -> pd.Series:
    """Blank out every company name (and its possessive) as a whole phrase."""
    names = sorted(companies, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")('s)?\b", re.I)
    return texts.map(lambda t: pattern.sub(" ", t))


def build_tfidf(texts: pd.Series):
    """Fit the shared TF-IDF vectoriser; rows come out L2-normalised."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def kmeans_sweep(X, ks=range(2, 16), seeds=range(5)) -> pd.DataFrame:
    """Inertia, silhouette and seed stability for each k.

    Inertia is from the RANDOM_STATE fit; silhouette is averaged over all seeds;
    stability is the mean pairwise adjusted Rand index between the seeds' partitions.
    """
    rows = []
    for k in ks:
        fits = {s: KMeans(k, n_init=10, random_state=s).fit(X) for s in seeds}
        sil = [silhouette_score(X, f.labels_) for f in fits.values()]
        ari = [adjusted_rand_score(a.labels_, b.labels_)
               for a, b in itertools.combinations(fits.values(), 2)]
        rows.append({"k": k, "inertia": fits[RANDOM_STATE].inertia_,
                     "silhouette": np.mean(sil), "silhouette_sd": np.std(sil),
                     "stability_ari": np.mean(ari)})
    return pd.DataFrame(rows).set_index("k")


def fit_kmeans(X, k: int) -> KMeans:
    """Final k-means fit with the project seed."""
    return KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(X)


def top_terms(centers: np.ndarray, terms: np.ndarray, n: int = 15) -> dict[int, list[str]]:
    """Highest-weighted terms in each cluster centroid."""
    return {c: terms[np.argsort(row)[::-1][:n]].tolist() for c, row in enumerate(centers)}


def order_by_dominant(ct: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by the column holding their largest share, then by that share."""
    pos = ct.values.argmax(axis=1)
    peak = ct.values.max(axis=1)
    return ct.iloc[np.lexsort((-peak, pos))]


def role_crosstab(labels, roles: pd.Series, prefix: str = "K") -> pd.DataFrame:
    """Cluster x role_family row percentages; columns by family size with 'other' last."""
    ct = pd.crosstab(pd.Series(labels, index=roles.index, name="cluster"), roles,
                     normalize="index") * 100
    order = [f for f in roles.value_counts().index if f != "other"] + ["other"]
    ct = ct[[f for f in order if f in ct.columns]]
    sizes = pd.Series(labels).value_counts()
    ct.index = [f"{prefix}{c} (n={sizes[c]})" for c in ct.index]
    return order_by_dominant(ct)


def family_landing(labels, roles: pd.Series, prefix: str = "K") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where each role family's postings land: row percentages and counts by cluster."""
    clusters = pd.Series(labels, index=roles.index, name="cluster")
    counts = pd.crosstab(roles, clusters)
    counts.columns = [f"{prefix}{c}" for c in counts.columns]
    return counts.div(counts.sum(axis=1), axis=0).mul(100).round(1), counts


def agreement(labels, df: pd.DataFrame) -> dict[str, float]:
    """How much of the company and of the role family a labeling recovers."""
    return {"nmi_company": normalized_mutual_info_score(df["company_clean"], labels),
            "nmi_role_family": normalized_mutual_info_score(df["role_family"], labels),
            "ari_role_family": adjusted_rand_score(df["role_family"], labels)}


def cosine_distances(X) -> np.ndarray:
    """Square matrix of cosine distances (1 - cosine similarity) between TF-IDF rows.

    Rows are L2-normalised, so cosine similarity is the sparse Gram matrix itself.
    """
    dist = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(dist, 0)
    return dist


def cosine_linkage(dist: np.ndarray, method: str) -> np.ndarray:
    """Hierarchical linkage on a square cosine-distance matrix."""
    return linkage(squareform(dist, checks=False), method=method)


def cut_k(Z: np.ndarray, k: int) -> np.ndarray:
    """Zero-based labels from cutting the tree into exactly k clusters.

    Cuts by merge order (scipy cut_tree) rather than by height (fcluster): postings that
    share no terms sit at cosine distance exactly 1, so the top complete-linkage merges
    can tie at 1.0, and a height cut cannot separate tied merges.
    """
    return cut_tree(Z, n_clusters=k).ravel()


def cut_height(Z: np.ndarray, k: int) -> float:
    """Merge height midway between the k-cluster and (k-1)-cluster cuts."""
    return float((Z[-k, 2] + Z[-(k - 1), 2]) / 2)


def cophenetic_corr(Z: np.ndarray, dist: np.ndarray) -> float:
    return float(cophenet(Z, squareform(dist, checks=False))[0])


def hier_k_scan(Z: np.ndarray, dist: np.ndarray, ks=range(2, 16)) -> pd.DataFrame:
    """What k the tree suggests: merge-height gap and cosine silhouette for each cut.

    merge_gap is how much higher the merge that would take k clusters down to k - 1 sits
    than the merge before it; a large gap means k clusters are well separated.
    """
    ks = list(ks)
    labels = cut_tree(Z, n_clusters=ks)
    h = Z[:, 2]
    rows = []
    for j, k in enumerate(ks):
        rows.append({"k": k, "merge_gap": h[-(k - 1)] - h[-k],
                     "silhouette_cosine": silhouette_score(dist, labels[:, j], metric="precomputed"),
                     "largest_cluster": int(np.bincount(labels[:, j]).max())})
    return pd.DataFrame(rows).set_index("k")


def kmeans_compare(X, ks, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Fit k-means at several k and summarise each: silhouette, sizes, agreement."""
    rows, labels = [], {}
    for k in ks:
        labels[k] = fit_kmeans(X, k).labels_
        sizes = np.bincount(labels[k])
        rows.append({"k": k, "silhouette": silhouette_score(X, labels[k]),
                     "pct_negative_silhouette": 100 * (silhouette_samples(X, labels[k]) < 0).mean(),
                     "smallest": sizes.min(), "largest": sizes.max(),
                     "sizes": sorted(sizes.tolist(), reverse=True),
                     **agreement(labels[k], df)})
    return pd.DataFrame(rows).set_index("k"), labels


def truncated_leaf_labels(Z: np.ndarray, p: int, roles: pd.Series,
                          clusters: np.ndarray) -> dict[int, str]:
    """Label each leaf of a lastp-truncated dendrogram by its cluster and majority family."""
    n = len(roles)
    _, nodes = to_tree(Z, rd=True)
    # the last p - 1 merges are undone, so leaves are exactly the nodes alive before them
    merged = {int(i) for i in Z[: n - p, :2].ravel()}
    alive = [i for i in range(n + (n - p)) if i not in merged]
    labels = {}
    for node_id in alive:
        members = nodes[node_id].pre_order()
        counts = roles.iloc[members].value_counts()
        share = 100 * counts.iloc[0] / len(members)
        labels[node_id] = (f"H{clusters[members[0]]} · {counts.index[0]} "
                           f"{share:.0f}% · n={len(members)}")
    return labels


def aligned_crosstab(a, b, a_prefix: str = "H", b_prefix: str = "K") -> pd.DataFrame:
    """Count cross-tab of two labelings, columns permuted to best match the rows."""
    ct = pd.crosstab(pd.Series(a, name="a"), pd.Series(b, name="b"))
    rows, cols = linear_sum_assignment(-ct.values)
    col_order = list(ct.columns[cols[np.argsort(rows)]])
    col_order += [c for c in ct.columns if c not in col_order]
    ct = ct[col_order]
    ct.index = [f"{a_prefix}{i}" for i in ct.index]
    ct.columns = [f"{b_prefix}{j}" for j in ct.columns]
    return ct


def fit_svd(X, n_components: int = 100) -> tuple[TruncatedSVD, np.ndarray]:
    """TruncatedSVD (works on sparse input; does not mean-centre) and the projected rows."""
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    coords = svd.fit_transform(X)
    return svd, coords


def top_loadings(svd: TruncatedSVD, terms: np.ndarray, n_components: int = 4,
                 n: int = 10) -> pd.DataFrame:
    """Top positive and negative loading terms for the first n_components.

    Uncentred SVD on non-negative TF-IDF gives a first component with no negative
    loadings, so a component only gets a '-' row when it has negative loadings.
    """
    rows = []
    for i, comp in enumerate(svd.components_[:n_components]):
        order = np.argsort(comp)
        rows.append({"component": i + 1, "direction": "+",
                     "terms": ", ".join(terms[order[::-1][:n]])})
        if (comp < 0).any():
            rows.append({"component": i + 1, "direction": "-",
                         "terms": ", ".join(terms[order[:n]])})
    return pd.DataFrame(rows)
