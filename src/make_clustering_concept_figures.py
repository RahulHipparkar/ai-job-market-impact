"""Draw the two explanatory figures for the Clustering page from small made-up data.

Output: figures/clustering_concept_methods.png (k-means against a dendrogram on the same
points) and figures/clustering_concept_distance.png (straight-line distance against
cosine), each with a thumbnail in figures/thumbs/.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

from plotstyle import CONTROL, PALETTE, apply_style, save

MUTED, FAINT = "#6B6B66", "#8A8A84"


def methods_figure():
    """The same 15 made-up points split by k-means and drawn as a dendrogram."""
    rng = np.random.default_rng(0)
    centres = np.array([[1.0, 1.0], [4.0, 1.5], [2.5, 4.0]])
    points = np.vstack([c + rng.normal(0, 0.45, size=(5, 2)) for c in centres])
    km = KMeans(3, n_init=10, random_state=0).fit(points)

    counts = {c + 1: int(n) for c, n in enumerate(np.bincount(km.labels_))}
    print(f"{len(points)} points; per k-means group: {counts}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for c in range(3):
        m = km.labels_ == c
        a1.scatter(points[m, 0], points[m, 1], s=60, color=PALETTE[c], edgecolor="white",
                   lw=1.5, zorder=3, label=f"Group {c + 1}")
    # hollow centres drawn under the points, so a point sitting on a centre stays visible
    a1.scatter(*km.cluster_centers_.T, marker="X", s=140, facecolor="white",
               edgecolor="#1C1C1A", lw=2, zorder=2, label="Group centre")
    # numbered as the dendrogram leaves are, so a point can be followed between panels
    # points 1, 8 and 15 sit beside a centre, 12 beside 15 and 9 beside 6, so those labels
    # move to the side away from their neighbour
    nudges = {0: (3, -13), 7: (-4, -13), 8: (5, -11), 11: (-14, 5), 14: (-3, -13)}
    labels = [a1.annotate(str(i + 1), (x, y), xytext=nudges.get(i, (5, 5)), textcoords="offset points",
                          fontsize=8, color=MUTED, zorder=5) for i, (x, y) in enumerate(points)]
    a1.set_title("Partitional: k-means with k = 3")
    a1.set_xlabel("Feature 1")
    a1.set_ylabel("Feature 2")
    a1.legend(frameon=False, fontsize=9, loc="upper right")

    Z = linkage(points, method="complete")
    tree = dendrogram(Z, ax=a2, color_threshold=0, above_threshold_color=PALETTE[0],
                      labels=[str(i + 1) for i in range(len(points))], leaf_font_size=9)
    assert tree["ivl"] == [str(i + 1) for i in tree["leaves"]]
    print("point  (x, y)          k-means group")
    for i, (x, y) in enumerate(points):
        print(f"{i + 1:>5}  ({x:5.2f}, {y:5.2f})  {km.labels_[i] + 1}")
    print("dendrogram leaf order:", " ".join(tree["ivl"]))
    cut = (Z[-3, 2] + Z[-2, 2]) / 2
    a2.axhline(cut, color=FAINT, ls="--", lw=1)
    # x in axes units, so the label stays inside the right edge whatever the leaf count
    a2.text(0.98, cut, "cut here: 3 groups", transform=a2.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=9, color=MUTED)
    a2.set_title("Hierarchical: a tree of merges")
    a2.set_xlabel("Point")
    a2.set_ylabel("Distance when merged")
    a2.grid(axis="x", visible=False)
    fig.suptitle("Two ways to group the same 15 made-up points")
    fig.tight_layout()
    check_labels(fig, a1, labels, points, km)
    save(fig, "clustering_concept_methods")


def check_labels(fig, ax, labels, points, km):
    """Print where each point label lands and check it is nearer its own point than any other marker."""
    fig.canvas.draw()
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    markers = np.vstack([points, km.cluster_centers_])
    names = [str(i + 1) for i in range(len(points))] + [f"centre {c + 1}" for c in range(3)]
    print("label  label centre (data)  its point (data)  px to own point  nearest other marker (px)")
    for i, label in enumerate(labels):
        box = label.get_window_extent()
        centre = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
        d = np.linalg.norm(to_px(markers) - centre, axis=1)
        other = np.delete(np.arange(len(markers)), i)
        near = other[d[other].argmin()]
        lx, ly = to_data(centre)
        flag = "" if d[i] < d[near] else "  <-- AMBIGUOUS"
        print(f"{i + 1:>5}  ({lx:4.2f}, {ly:4.2f})         ({points[i, 0]:4.2f}, {points[i, 1]:4.2f})"
              f"      {d[i]:5.1f}           {names[near]:>8}: {d[near]:5.1f}{flag}")
        assert d[i] < d[near], f"label {i + 1} is nearer {names[near]} than its own point"


def distance_figure():
    """Why cosine suits text: a long and a short posting with the same word mix."""
    a, b, c = np.array([2.0, 1.0]), np.array([6.0, 3.0]), np.array([1.0, 3.0])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # B first, so the shorter A, which lies along the same line, is drawn on top of it
    for vec, name, color, lw in [(b, "Posting B (long, same mix as A)", PALETTE[2], 2.2),
                                 (a, "Posting A (short)", PALETTE[0], 3.4),
                                 (c, "Posting C (different mix)", PALETTE[3], 2.2)]:
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
        ax.plot([], [], color=color, lw=lw, label=name)
        label_at = {"Posting A": a + [-0.85, -0.6]}.get(name.split(" (")[0], vec + [0.12, -0.25])
        ax.text(*label_at, name.split(" (")[0], fontsize=10, color="#1C1C1A")
    # A and B lie on one line, so their straight-line gap is drawn just below that line
    offset = np.array([0.12, -0.24])
    ax.annotate("", xy=b + offset, xytext=a + offset,
                arrowprops=dict(arrowstyle="|-|", color=CONTROL, lw=1.4, mutation_scale=4))
    ax.text(4.0, 1.2, f"straight-line distance {np.linalg.norm(b - a):.1f}\nangle 0°",
            fontsize=9, color=MUTED)
    ax.plot(*zip(a, c), color=CONTROL, ls="--", lw=1.4)
    angle = np.degrees(np.arccos(a @ c / (np.linalg.norm(a) * np.linalg.norm(c))))
    ax.text(1.65, 2.25, f"straight-line distance {np.linalg.norm(c - a):.1f}\nangle {angle:.0f}°",
            fontsize=9, color=MUTED)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4)
    ax.set_xlabel("How often word 1 is used")
    ax.set_ylabel("How often word 2 is used")
    ax.set_title("Straight-line distance against cosine (the angle between postings)")
    # arrows are drawn B first for layering, but the legend reads A, B, C
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
              loc="upper left")
    save(fig, "clustering_concept_distance")


if __name__ == "__main__":
    apply_style()
    methods_figure()
    distance_figure()
    print("saved clustering_concept_methods.png and clustering_concept_distance.png")
