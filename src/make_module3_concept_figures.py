"""Draw the explanatory figures for the Naive Bayes and Decision Trees pages from made-up data.

Output: figures/nb_concept_1.png (one document read as counts and as presence),
figures/nb_concept_2.png (scoring a document, and what smoothing does),
figures/dt_concept_1.png (Gini and entropy against class proportion) and
figures/dt_concept_2.png (one split and its information gain), each with a thumbnail in
figures/thumbs/. Every number is invented and printed so the arithmetic can be checked.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from plotstyle import PALETTE, apply_style, save

MUTED, FAINT, TEXT = "#6B6B66", "#8A8A84", "#1C1C1A"
CLASS_A, CLASS_B = PALETTE[0], PALETTE[1]


def nb_counts_figure():
    """The same made-up document as word counts (Multinomial) and word presence (Bernoulli)."""
    vocab = ["data", "pipeline", "customer", "deploy", "sales", "quota"]
    doc = "Move the data through the pipeline, check the data, then share the data with the customer."
    words = [w.strip(",.").lower() for w in doc.split()]
    counts = np.array([words.count(v) for v in vocab])
    present = (counts > 0).astype(int)
    print("document:", doc)
    print("vocabulary word  count  present")
    for v, c, p in zip(vocab, counts, present):
        print(f"  {v:10s}      {c}       {p}")

    x = np.arange(len(vocab))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))

    a1.bar(x, counts, width=0.6, color=CLASS_A, zorder=3)
    # P stands for P(word | c), the word under the bar
    for i, c in enumerate(counts):
        if c:
            a1.text(i, c + 0.1, rf"$\times\,P^{{{c}}}$", ha="center", fontsize=11, color=TEXT)
        else:
            a1.text(i, 0.1, "× P⁰ = 1\n(drops out)", ha="center", fontsize=8.5, color=FAINT)
    a1.annotate("repeats count:\n3 uses, so its P is cubed", xy=(0.32, 2.6), xytext=(1.3, 3.2),
                fontsize=9.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    a1.set_ylim(0, 4)
    a1.set_yticks(range(5))
    a1.set_ylabel("Times the word appears in the document")
    a1.set_title("Multinomial NB: how often each word appears")

    a2.bar(x[present == 1], present[present == 1], width=0.6, color=CLASS_A, zorder=3)
    a2.scatter(x[present == 0], np.zeros((present == 0).sum()), s=90, facecolor="white",
               edgecolor=CLASS_B, lw=2, zorder=4, clip_on=False)
    for i, p in enumerate(present):
        if p:
            a2.text(i, 1.04, r"$\times\,P$", ha="center", fontsize=11, color=TEXT)
        else:
            a2.text(i, 0.08, r"$\times\,(1 - P)$", ha="center", fontsize=10.5, color=CLASS_B)
    a2.annotate("3 uses or 1 use:\nthe same 'present'", xy=(0.32, 0.85), xytext=(1.0, 1.3),
                fontsize=9.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    a2.text(4, 0.45, "absent words are evidence too:\neach one multiplies in 1 − P",
            ha="center", fontsize=9.5, color=CLASS_B)
    a2.set_ylim(0, 1.55)
    a2.set_yticks([0, 1], ["0 (absent)", "1 (present)"])
    a2.set_ylabel("Does the word appear?")
    a2.set_title("Bernoulli NB: whether each word appears")

    for a in (a1, a2):
        a.set_xticks(x, vocab)
        a.set_xlabel("Word in a six-word vocabulary")
        a.grid(axis="x", visible=False)
    fig.suptitle(f"One made-up document, read two ways (P = P(word | c), the chance of that word "
                 f"in the class c being scored)\n“{doc}”", fontsize=12)
    fig.tight_layout()
    save(fig, "nb_concept_1")


def nb_scoring_figure():
    """Log prior plus log likelihoods per class, and a word never seen in one class."""
    vocab = ["goal", "match", "vote", "team", "poll"]
    counts = {"A": np.array([6, 8, 0, 5, 1]), "B": np.array([1, 2, 7, 1, 9])}
    prior = {"A": 0.6, "B": 0.4}
    doc = ["goal", "match", "vote"]
    alpha, V = 1, len(vocab)
    idx = [vocab.index(w) for w in doc]
    p_smooth = {c: (n + alpha) / (n.sum() + alpha * V) for c, n in counts.items()}
    p_raw = {c: n / n.sum() for c, n in counts.items()}

    print(f"\nvocabulary {vocab}; training words per class: { {c: int(n.sum()) for c, n in counts.items()} }")
    print(f"document {doc}; priors {prior}; alpha = {alpha}")
    steps = {}
    for c in counts:
        terms = [np.log(prior[c])] + [np.log(p_smooth[c][i]) for i in idx]
        steps[c] = np.cumsum(terms)
        print(f"class {c}, smoothed: ln {prior[c]} " + " ".join(
            f"+ ln({counts[c][i]}+1)/({counts[c].sum()}+{V})" for i in idx)
            + " = " + " ".join(f"{t:+.3f}" for t in terms) + f" = {steps[c][-1]:.3f}")
    raw_total = {}
    for c in counts:
        with np.errstate(divide="ignore"):
            raw_total[c] = np.log(prior[c]) + sum(np.log(p_raw[c][i]) for i in idx)
        print(f"class {c}, no smoothing: total = {raw_total[c]:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(doc) + 1)
    labels = ["log prior", *[f"+ log\nP({w} | c)" for w in doc]]
    for c, color, dy in [("A", CLASS_A, 10), ("B", CLASS_B, -16)]:
        a1.plot(x, steps[c], marker="o", ms=7, lw=2.2, color=color, label=f"Class {c} (prior {prior[c]})", zorder=3)
        for xi, value in zip(x, steps[c]):
            # A's last label sits to the right, clear of the steep segment arriving from above
            last_a = c == "A" and xi == x[-1]
            a1.annotate(f"{value:.2f}", (xi, value), xytext=(9, -4) if last_a else (0, dy),
                        textcoords="offset points", ha="left" if last_a else "center", fontsize=9, color=color)
    winner = max(steps, key=lambda c: steps[c][-1])
    a1.annotate(f"highest total wins:\nclass {winner}\n({steps['A'][-1]:.2f} > {steps['B'][-1]:.2f})",
                xy=(x[-1] + 0.03, steps[winner][-1] + 0.08), xytext=(x[-1] + 0.12, -4.2),
                fontsize=9.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    a1.set_xticks(x, labels, fontsize=9.5)
    a1.set_xlim(-0.3, x[-1] + 1.0)
    a1.set_ylim(-7.4, 0.3)
    a1.set_xlabel("Term added to the running total")
    a1.set_ylabel("Running log-probability (natural log)")
    a1.set_title(f"Scoring “{' '.join(doc)}” with smoothing (α = {alpha})")
    a1.legend(frameon=False, loc="lower left", fontsize=9)

    xb = np.arange(len(doc))
    w = 0.36
    raw_a, smooth_a = p_raw["A"][idx], p_smooth["A"][idx]
    a2.bar(xb - w / 2, raw_a, w, color=FAINT, label=f"no smoothing: count / {counts['A'].sum()}", zorder=3)
    a2.bar(xb + w / 2, smooth_a, w, color=CLASS_A,
           label=f"smoothed: (count + 1) / ({counts['A'].sum()} + {V})", zorder=3)
    for xi, r, s, i in zip(xb, raw_a, smooth_a, idx):
        a2.text(xi - w / 2, r + 0.01, f"{counts['A'][i]}/{counts['A'].sum()}", ha="center", fontsize=8.5, color=MUTED)
        a2.text(xi + w / 2, s + 0.01, f"{counts['A'][i] + 1}/{counts['A'].sum() + V}", ha="center", fontsize=8.5,
                color=CLASS_A)
    a2.annotate("never seen in class A:\nP = 0, log P = −∞,\nso A can never win", xy=(2 - w / 2, 0.04),
                xytext=(1.5, 0.22), fontsize=10, color=CLASS_B,
                arrowprops=dict(arrowstyle="->", color=CLASS_B, lw=1))
    a2.text(0.98, 0.79, f"totals without smoothing:  A = −∞,  B = {raw_total['B']:.2f}  → B\n"
                        f"totals with smoothing:  A = {steps['A'][-1]:.2f},  B = {steps['B'][-1]:.2f}  → A",
            transform=a2.transAxes, ha="right", va="top", fontsize=10, color=TEXT)
    a2.set_xticks(xb, doc)
    a2.set_ylim(0, 0.7)
    a2.set_xlabel("Word in the document")
    a2.set_ylabel("P(word | class A)")
    a2.set_title("A word never seen in class A")
    a2.legend(frameon=False, loc="upper left", fontsize=10)
    a2.grid(axis="x", visible=False)
    fig.suptitle(f"Naive Bayes on made-up counts: two classes, {counts['A'].sum()} training words each, "
                 f"vocabulary of {V}")
    fig.tight_layout()
    save(fig, "nb_concept_2")


def gini(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return 1 - p ** 2 - (1 - p) ** 2


def entropy(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.nan_to_num(h)


def impurity_curves_figure():
    """Gini and entropy against the proportion of one class in a two-class node."""
    p = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(p, entropy(p), lw=2.5, color=CLASS_A, label="Entropy  −p log₂ p − (1 − p) log₂(1 − p)")
    ax.plot(p, gini(p), lw=2.5, color=CLASS_B, label="Gini  1 − p² − (1 − p)²")
    ax.plot(p, 2 * gini(p), lw=1.3, ls="--", color=CLASS_B, alpha=0.6,
            label="2 × Gini (rescaled to compare shape)")
    for pt in (0, 1):
        ax.scatter([pt], [0], s=70, color=TEXT, zorder=5, clip_on=False)
    # one label under both curves, pointing at both pure ends
    label_at = (0.5, 0.17)
    ax.text(*label_at, "pure nodes (all one class):\nimpurity 0 for both measures", ha="center",
            va="center", fontsize=9.5, color=MUTED)
    for end, start in [(0, (0.34, 0.15)), (1, (0.66, 0.15))]:
        ax.annotate("", xy=(end, 0), xytext=start, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.scatter([0.5, 0.5], [1.0, 0.5], s=70, color=[CLASS_A, CLASS_B], zorder=5)
    ax.annotate("50 / 50, maximally impure:\nentropy 1.0 bit", xy=(0.5, 1.0), xytext=(0.6, 1.04),
                fontsize=9.5, color=CLASS_A, arrowprops=dict(arrowstyle="->", color=CLASS_A, lw=1))
    ax.annotate("Gini 0.5", xy=(0.5, 0.5), xytext=(0.58, 0.42), fontsize=9.5, color=CLASS_B,
                arrowprops=dict(arrowstyle="->", color=CLASS_B, lw=1))
    for pt in (0.2, 0.8):
        print(f"p = {pt}: entropy {entropy(pt):.3f}, gini {gini(pt):.3f}, 2 x gini {2 * gini(pt):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Proportion of the node in class A (p); the rest is class B")
    ax.set_ylabel("Impurity")
    ax.set_title("Two measures of how mixed a two-class node is")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "dt_concept_1")


def _node(ax, x, y, title, a, b, width=0.4, height=0.28):
    """One tree node: its class counts, a bar of the class mix, and both impurities."""
    n = a + b
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="white",
                                edgecolor=FAINT, lw=1.2, transform=ax.transAxes))
    ax.text(x, y + height / 2 - 0.045, f"{title} (n = {n})", ha="center", va="center", fontsize=12,
            fontweight="bold", color=TEXT, transform=ax.transAxes)
    bar_w, bar_y = width - 0.06, y + 0.005
    ax.add_patch(Rectangle((x - bar_w / 2, bar_y), bar_w * a / n, 0.045, color=CLASS_A, transform=ax.transAxes))
    ax.add_patch(Rectangle((x - bar_w / 2 + bar_w * a / n, bar_y), bar_w * b / n, 0.045, color=CLASS_B,
                           transform=ax.transAxes))
    ax.text(x, bar_y - 0.04, f"A {a}  ·  B {b}", ha="center", va="center", fontsize=12, color=TEXT,
            transform=ax.transAxes)
    ax.text(x, y - height / 2 + 0.04, f"Gini {gini(a / n):.3f} · entropy {entropy(a / n):.3f}",
            ha="center", va="center", fontsize=11, color=TEXT, transform=ax.transAxes)


def split_figure():
    """A made-up parent node split into two children, and the drop in impurity it buys."""
    parent, left, right = (50, 50), (40, 0), (10, 50)
    n, n_l, n_r = sum(parent), sum(left), sum(right)
    before = {"Gini": gini(parent[0] / n), "Entropy (bits)": entropy(parent[0] / n)}
    child = {"Gini": (gini(left[0] / n_l), gini(right[0] / n_r)),
             "Entropy (bits)": (entropy(left[0] / n_l), entropy(right[0] / n_r))}
    after = {k: n_l / n * l + n_r / n * r for k, (l, r) in child.items()}
    print(f"\nparent {parent}, left {left}, right {right}")
    for k in before:
        print(f"{k}: before {before[k]:.3f}; after = ({n_l}/{n}) x {child[k][0]:.3f} + ({n_r}/{n}) x "
              f"{child[k][1]:.3f} = {after[k]:.3f}; decrease {before[k] - after[k]:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    a1.axis("off")
    _node(a1, 0.5, 0.83, "Parent", *parent)
    _node(a1, 0.22, 0.4, "Left child", *left)
    _node(a1, 0.78, 0.4, "Right child", *right)
    for x_end, answer in [(0.22, "yes"), (0.78, "no")]:
        a1.annotate("", xy=(x_end, 0.54), xytext=(0.5, 0.69), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.4))
        a1.text((x_end + 0.5) / 2 + (-0.06 if answer == "yes" else 0.06), 0.6, answer, ha="center",
                fontsize=11, color=MUTED, transform=a1.transAxes)
    a1.text(0.5, 0.615, "feature ≤ threshold?", ha="center", va="center", fontsize=11, color=TEXT,
            transform=a1.transAxes, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    # the weighting written out, one line per measure (weights = share of the parent in each child)
    wl, wr = n_l / n, n_r / n
    a1.text(0.5, 0.11, f"after = {wl:.1f} × left + {wr:.1f} × right\n"
                       f"Gini:  {wl:.1f} × {child['Gini'][0]:.3f} + {wr:.1f} × {child['Gini'][1]:.3f} = "
                       f"{after['Gini']:.3f}\n"
                       f"entropy:  {wl:.1f} × {child['Entropy (bits)'][0]:.3f} + {wr:.1f} × "
                       f"{child['Entropy (bits)'][1]:.3f} = {after['Entropy (bits)']:.3f}",
            ha="center", va="center", fontsize=11, color=TEXT, linespacing=1.5, transform=a1.transAxes)
    a1.legend(handles=[Rectangle((0, 0), 1, 1, color=CLASS_A, label="class A"),
                       Rectangle((0, 0), 1, 1, color=CLASS_B, label="class B")],
              loc="upper left", frameon=False, fontsize=10.5)
    a1.set_title("One split of a made-up node")

    x = np.arange(len(before)) * 1.8  # wide gaps leave room for the gain labels
    w = 0.34
    a2.bar(x - w / 2, list(before.values()), w, color=FAINT, label="before: parent", zorder=3)
    a2.bar(x + w / 2, list(after.values()), w, color=CLASS_A, label="after: weighted children", zorder=3)
    for xi, k in zip(x, before):
        a2.text(xi - w / 2, before[k] + 0.02, f"{before[k]:.3f}", ha="center", fontsize=10.5, color=MUTED)
        a2.text(xi + w / 2, after[k] + 0.02, f"{after[k]:.3f}", ha="center", fontsize=10.5, color=CLASS_A)
        a2.annotate("", xy=(xi + w / 2 + 0.2, after[k]), xytext=(xi + w / 2 + 0.2, before[k]),
                    arrowprops=dict(arrowstyle="<->", color=CLASS_B, lw=1.6))
        gain = "information gain" if k.startswith("Entropy") else "Gini decrease"
        a2.text(xi + w / 2 + 0.25, (before[k] + after[k]) / 2, f"{gain}\n{before[k]:.3f} − {after[k]:.3f}\n"
                f"= {before[k] - after[k]:.3f}", fontsize=10.5, color=CLASS_B, va="center")
    a2.set_xticks(x, list(before))
    a2.set_xlim(-0.5, x[-1] + 1.25)
    a2.set_ylim(0, 1.2)
    a2.set_xlabel("Impurity measure")
    a2.set_ylabel("Impurity")
    a2.set_title("Impurity before and after the split")
    a2.legend(frameon=False, loc="upper left", fontsize=10.5)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "dt_concept_2")


if __name__ == "__main__":
    apply_style()
    nb_counts_figure()
    nb_scoring_figure()
    impurity_curves_figure()
    split_figure()
    print("saved nb_concept_1.png, nb_concept_2.png, dt_concept_1.png and dt_concept_2.png")
