"""Draw the two explanatory figures for the PCA page from small made-up data.

Output: figures/pca_concept_components.png (eigenvectors and eigenvalues on correlated
points, and the same points reduced to one dimension) and figures/pca_concept_dimensions.png
(distances between random points bunching together as dimensions grow), each with a
thumbnail in figures/thumbs/.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from plotstyle import PALETTE, SEQUENTIAL, apply_style, save

MUTED = "#6B6B66"


def components_figure():
    """Two correlated made-up features, their principal components, and the 1D projection."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    pts = np.column_stack([x, 0.8 * x + rng.normal(0, 0.45, 200)])
    pts = (pts - pts.mean(0)) / pts.std(0)
    pca = PCA().fit(pts)
    share = pca.explained_variance_ratio_ * 100

    v1, v2 = pca.components_
    angle = np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1)))
    print(f"eigenvectors: PC1 ({v1[0]:.4f}, {v1[1]:.4f}), PC2 ({v2[0]:.4f}, {v2[1]:.4f})")
    print(f"angle between them: {angle:.4f} degrees")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    a1.scatter(*pts.T, s=14, color=SEQUENTIAL[2], lw=0)
    ends = pca.components_ * 2 * np.sqrt(pca.explained_variance_)[:, None]
    # labels sit in empty space beside the diagonal cloud; PC2's is joined to its short arrow
    # by a thin leader that drops from above, clear of the outlying point just past the tip
    label_at = [(2.65, -0.65), (-0.9, 2.3)]
    for i, (end, val) in enumerate(zip(ends, pca.explained_variance_)):
        a1.annotate("", xy=end, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE[i], lw=2.6))
        leader = dict(arrowstyle="-", color=PALETTE[i], lw=0.8, shrinkA=0, shrinkB=2) if i == 1 else None
        a1.annotate(f"PC{i + 1}\neigenvalue {val:.2f}\n{share[i]:.0f}% of variance",
                    xy=end, xytext=label_at[i], arrowprops=leader,
                    color=PALETTE[i], fontsize=9, ha="center", va="center", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    # one symmetric range on both axes, so equal scaling shows the 90 degree angle as 90 degrees
    lim = np.ceil(max(np.abs(pts).max(), np.abs(ends).max()) * 1.1 * 4) / 4
    a1.set_xlim(-lim, lim)
    a1.set_ylim(-lim, lim)
    a1.set_aspect("equal")
    inside = (np.abs(pts) < lim).all(axis=1).sum()
    print(f"left panel: xlim {tuple(map(float, a1.get_xlim()))}, ylim {tuple(map(float, a1.get_ylim()))}")
    print(f"left panel: {inside} of {len(pts)} points inside the axes")
    a1.set_title("Eigenvectors give the directions")
    a1.set_xlabel("Feature 1 (standardised)")
    a1.set_ylabel("Feature 2 (standardised)")

    scores = pca.transform(pts)[:, 0]
    a2.scatter(scores, rng.uniform(-0.15, 0.15, len(scores)), s=14, color=PALETTE[0], lw=0)
    a2.set_ylim(-1, 1)
    a2.set_yticks([])
    a2.grid(axis="y", visible=False)
    a2.set_title(f"Kept: PC1 only, {share[0]:.0f}% of the variance")
    a2.set_xlabel("Position along PC1")
    fig.suptitle("PCA on 200 made-up points: two features reduced to one")
    fig.tight_layout()
    fig.canvas.draw()
    box = a1.get_window_extent()
    print(f"left panel: get_aspect() = {a1.get_aspect()}; plotted box {box.width:.1f} x {box.height:.1f} px, "
          f"{'square' if abs(box.width - box.height) < 1 else 'not square'}")
    save(fig, "pca_concept_components")


def dimensions_figure():
    """Pairwise distances between random points, scaled by their mean, at four dimensions."""
    rng = np.random.default_rng(0)
    dims = [2, 10, 100, 1000]
    # each panel keeps its own y-scale: shared, the wide 2-dimension spread is squashed flat
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharex=True, sharey=False)
    for ax, d in zip(axes, dims):
        dist = pdist(rng.uniform(size=(300, d)))
        rel = dist / dist.mean()
        # every panel has the same 44,850 pairs, so counts compare across panels as density did
        counts, _, _ = ax.hist(rel, bins=np.linspace(0, 2.5, 51), color=PALETTE[0])
        ax.set_ylim(0, counts.max() * 1.4)  # headroom so the note above the bars stays clear
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.set_title(f"{d:,} dimensions", fontsize=10)
        ratio = dist.max() / dist.min()
        ax.text(0.97, 0.95, f"farthest pair is\n{ratio:,.{0 if ratio >= 100 else 1}f}x the nearest",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        ax.grid(axis="y", visible=False)
    axes[0].set_ylabel("Number of pairs")
    fig.supxlabel("Distance between two points, as a multiple of the average distance", fontsize=10)
    fig.suptitle("As dimensions grow, every point ends up about the same distance from every other")
    fig.tight_layout()
    for ax, d in zip(axes, dims):
        tallest = max(patch.get_height() for patch in ax.patches)
        print(f"{d:>5} dimensions: y-axis upper limit {ax.get_ylim()[1]:,.0f}, tallest bar {tallest:,.0f} "
              f"({tallest / ax.get_ylim()[1]:.0%} of the panel height)")
    save(fig, "pca_concept_dimensions")


if __name__ == "__main__":
    apply_style()
    components_figure()
    dimensions_figure()
    print("saved pca_concept_components.png and pca_concept_dimensions.png")
