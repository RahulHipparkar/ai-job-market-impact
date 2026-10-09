"""Figures drawn once per text variant (raw / stripped) in the clustering notebook."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.metrics import silhouette_samples

from src.plotstyle import CMAP, PALETTE, SEQUENTIAL

MUTED, FAINT = "#6B6B66", "#C9C9C4"
SEQ_MID = SEQUENTIAL[2]


def plot_k_sweep(sweep: pd.DataFrame, k: int, label: str):
    """Elbow (inertia) and silhouette against k, side by side, with the chosen k marked."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    a1.plot(sweep.index, sweep["inertia"], marker="o", ms=5, lw=2, color=PALETTE[0])
    a1.set_title("Inertia (elbow)")
    a1.set_ylabel("Within-cluster sum of squares")
    a2.plot(sweep.index, sweep["silhouette"], marker="o", ms=5, lw=2, color=PALETTE[0])
    a2.fill_between(sweep.index, sweep["silhouette"] - sweep["silhouette_sd"],
                    sweep["silhouette"] + sweep["silhouette_sd"],
                    color=PALETTE[0], alpha=0.15, lw=0)
    a2.set_title("Silhouette score (mean ± SD over 5 seeds)")
    a2.set_ylabel("Silhouette score")
    for a in (a1, a2):
        a.axvline(k, color="#8A8A84", ls="--", lw=1)
        a.set_xlabel("Number of clusters (k)")
        a.set_xticks(sweep.index)
    a2.text(k + 0.2, a2.get_ylim()[0] + 0.05 * np.ptp(a2.get_ylim()), f"chosen k = {k}",
            fontsize=9, color=MUTED)
    fig.suptitle(f"Choosing k for k-means on TF-IDF of postings ({label})")
    fig.tight_layout()
    return fig


def plot_heatmap(ct: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                 cbar_label: str, fmt: str = ".0f", figsize=(9, 5), color_by=None):
    """Annotated heatmap in the project sequential colormap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ct if color_by is None else color_by, annot=ct.values, fmt=fmt, cmap=CMAP,
                linewidths=0.5, linecolor="white", cbar_kws={"label": cbar_label},
                annot_kws={"size": 8}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return fig


def draw_silhouette(ax, values: np.ndarray, labels: np.ndarray, label_clusters: bool = True,
                    prefix: str = "K"):
    """Per-sample silhouette diagram: one band per cluster, samples sorted within it."""
    gap = max(10, len(values) // 150)
    y = 0
    for c in range(labels.max() + 1):
        v = np.sort(values[labels == c])
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, color=PALETTE[0], lw=0)
        if label_clusters:
            ax.text(-0.02, y + len(v) / 2, f"{prefix}{c}", ha="right", va="center",
                    fontsize=8, color=MUTED, transform=ax.get_yaxis_transform())
        y += len(v) + gap
    mean = values.mean()
    ax.axvline(mean, color=PALETTE[1], ls="--", lw=1.2)
    ax.axvline(0, color="#8A8A84", lw=0.8)
    ax.set_ylim(-gap, y)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    return mean


def plot_silhouette(X, labels: np.ndarray, k: int, label: str):
    """Per-sample silhouette diagram for one k-means fit."""
    values = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(9, 6))
    mean = draw_silhouette(ax, values, labels)
    ax.text(mean, 1.01, f" mean = {mean:.3f}", transform=ax.get_xaxis_transform(),
            fontsize=9, color=MUTED, va="bottom")
    ax.set_title(f"Per-posting silhouette values, k-means with k = {k} ({label})", pad=18)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Postings, grouped by cluster and sorted", labelpad=28)
    return fig


def plot_silhouette_compare(X, labels_by_k: dict[int, np.ndarray], chosen: int, label: str):
    """Silhouette diagrams side by side for several k, sharing the x axis."""
    ks = list(labels_by_k)
    fig, axes = plt.subplots(1, len(ks), figsize=(11, 4.8), sharex=True)
    for ax, k in zip(axes, ks):
        values = silhouette_samples(X, labels_by_k[k])
        mean = draw_silhouette(ax, values, labels_by_k[k], label_clusters=False)
        tag = " (chosen)" if k == chosen else ""
        ax.set_title(f"k = {k}{tag}\nmean = {mean:.3f}", fontsize=10,
                     fontweight="bold" if k == chosen else "normal")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Postings, grouped by cluster")
    fig.supxlabel("Silhouette coefficient", fontsize=10)
    fig.suptitle(f"Per-posting silhouette values at different k ({label})")
    fig.tight_layout()
    return fig


def plot_hier_k_scan(scan: pd.DataFrame, kmeans_k: int, label: str):
    """Merge-height gap and cut silhouette against k for the hierarchical tree."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    best_gap, best_sil = scan["merge_gap"].idxmax(), scan["silhouette_cosine"].idxmax()
    a1.bar(scan.index, scan["merge_gap"],
           color=[PALETTE[0] if k == best_gap else SEQ_MID for k in scan.index], width=0.75)
    a1.set_title("Gap in merge height above each cut")
    a1.set_ylabel("Merge height gap (cosine distance)")
    a1.grid(axis="x", visible=False)
    a2.plot(scan.index, scan["silhouette_cosine"], marker="o", ms=5, lw=2, color=PALETTE[0])
    a2.plot(best_sil, scan.loc[best_sil, "silhouette_cosine"], marker="o", ms=10,
            mfc="none", mec=PALETTE[1], mew=2)
    a2.set_title("Silhouette of each cut (cosine)")
    a2.set_ylabel("Silhouette score")
    for a in (a1, a2):
        a.axvline(kmeans_k, color="#8A8A84", ls="--", lw=1)
        a.set_xlabel("Number of clusters (k)")
        a.set_xticks(scan.index)
    a1.text(kmeans_k + 0.2, a1.get_ylim()[1] * 0.97, f"k-means k = {kmeans_k}",
            fontsize=9, color=MUTED, va="top")
    fig.suptitle(f"Which k the complete-linkage tree suggests ({label})")
    fig.tight_layout()
    return fig


def plot_dendrogram(Z: np.ndarray, threshold: float, leaf_labels: dict[int, str], p: int,
                    k: int, label: str):
    """lastp-truncated complete-linkage dendrogram, branches below the k-cluster cut coloured."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    # two alternating hues mark where one cluster ends and the next begins; the
    # H-number in each leaf label carries the cluster identity
    set_link_color_palette([PALETTE[0], PALETTE[2]])
    dendrogram(Z, truncate_mode="lastp", p=p, orientation="left",
               color_threshold=threshold, above_threshold_color=FAINT,
               leaf_label_func=lambda i: leaf_labels[i], leaf_font_size=8,
               show_contracted=True, ax=ax)
    set_link_color_palette(None)
    # complete-linkage merges near the top of the tree all sit close to distance 1, so
    # zoom the axis to the span of the merges actually drawn
    lo, hi = Z[-(p - 1), 2], Z[-1, 2]
    pad = 0.08 * (hi - lo)
    ax.set_xlim(hi + pad, lo - pad)
    ax.axvline(threshold, color="#8A8A84", ls="--", lw=1)
    ax.text(threshold, ax.get_ylim()[1], f" cut: {k} clusters", fontsize=9,
            color=MUTED, va="bottom")
    ax.set_title(f"Complete-linkage dendrogram on cosine distance, last {p} merges ({label})",
                 pad=18)
    ax.set_xlabel(f"Cosine distance at merge (axis zoomed to the last {p} merges)")
    ax.set_ylabel("Merged group: hierarchical cluster · majority role family · size")
    ax.grid(axis="y", visible=False)
    return fig


def plot_scree(svd, label: str, n_show: int = 30):
    """Per-component and cumulative explained variance for a TruncatedSVD."""
    ratio = svd.explained_variance_ratio_ * 100
    comps = np.arange(1, len(ratio) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    a1.bar(comps[:n_show], ratio[:n_show], color=PALETTE[0], width=0.8)
    a1.set_title(f"Per component (first {n_show})")
    a1.set_ylabel("Variance explained (%)")
    a1.grid(axis="x", visible=False)
    a2.plot(comps, ratio.cumsum(), lw=2, color=PALETTE[0])
    a2.set_title(f"Cumulative ({len(ratio)} components)")
    a2.set_ylabel("Cumulative variance explained (%)")
    a2.set_ylim(0, None)
    for a in (a1, a2):
        a.set_xlabel("Component")
    fig.suptitle(f"Explained variance of truncated SVD on TF-IDF ({label})")
    fig.tight_layout()
    return fig


def plot_projection_facets(coords: np.ndarray, groups: pd.Series, order: list,
                           ncols: int, title: str, axis_labels: tuple[str, str],
                           panel_name=str, figsize=(11, 5.5)):
    """One panel per group: that group in colour over every posting in grey.

    Small multiples keep each group legible without needing more than one hue.
    """
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    g = np.asarray(groups)
    for ax, name in zip(axes, order):
        mask = g == name
        ax.scatter(coords[:, 0], coords[:, 1], s=2, color=FAINT, alpha=0.4, lw=0,
                   rasterized=True)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, color=PALETTE[0], alpha=0.7,
                   lw=0, rasterized=True)
        ax.set_title(f"{panel_name(name)} (n={mask.sum()})", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    fig.supxlabel(axis_labels[0], fontsize=10)
    fig.supylabel(axis_labels[1], fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
