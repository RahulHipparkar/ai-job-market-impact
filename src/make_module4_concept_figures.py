"""Draw the explanatory figures for the Regression and AdaBoost pages from made-up data.

Output: figures/reg_concept_1.png (linear against logistic regression),
figures/reg_concept_2.png (the sigmoid, the 0.5 threshold, and what moving it does),
figures/ab_concept_1.png (bagging against boosting) and
figures/ab_concept_2.png (AdaBoost worked through on ten points), each with a thumbnail in
figures/thumbs/. Every number is invented and printed so the arithmetic can be checked.
"""

from fractions import Fraction
from math import lcm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import minimize

from plotstyle import PALETTE, apply_style, save

MUTED, FAINT, TEXT = "#6B6B66", "#8A8A84", "#1C1C1A"
POS, NEG, WRONG = PALETTE[0], PALETTE[1], PALETTE[3]

# ---------------------------------------------------------------- regression

HOURS = np.arange(1, 13)
SCORE = np.array([52, 55, 61, 64, 70, 72, 79, 83, 86, 91, 93, 97])  # exam score out of 100
PASSED = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])  # 1 = passed


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def least_squares(x, y) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def logistic_fit(x, y) -> tuple[float, float]:
    """Unpenalised logistic regression by maximum likelihood."""
    def nll(b):
        z = b[0] + b[1] * x
        return np.sum(np.log1p(np.exp(z)) - y * z)
    b = minimize(nll, x0=np.zeros(2), method="BFGS").x
    return float(b[0]), float(b[1])


def linear_vs_logistic_figure():
    """The same input predicting a number, and predicting a yes-or-no outcome."""
    a, b = least_squares(HOURS, SCORE)
    la, lb = least_squares(HOURS, PASSED)
    ga, gb = logistic_fit(HOURS, PASSED)
    print("hours studied:", HOURS.tolist())
    print("exam score:   ", SCORE.tolist())
    print("passed (1/0): ", PASSED.tolist())
    print(f"linear fit to the score: score = {a:.2f} + {b:.2f} x hours; at 14 hours {a + b * 14:.1f}")
    print(f"straight line fitted to pass/fail: {la:.3f} + {lb:.3f} x hours; at 0 hours {la:.3f}, "
          f"at 14 hours {la + lb * 14:.3f}")
    print(f"logistic fit: P(pass) = 1 / (1 + exp(-({ga:.3f} + {gb:.3f} x hours))); at 0 hours "
          f"{sigmoid(ga):.3f}, at 14 hours {sigmoid(ga + gb * 14):.3f}; 0.5 crossed at {-ga / gb:.2f} hours")

    grid = np.linspace(0, 14, 200)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5))
    a1.scatter(HOURS, SCORE, s=55, color=POS, zorder=4, label="students (made up)")
    a1.plot(grid, a + b * grid, color=TEXT, lw=2, label="least-squares line")
    a1.annotate(f"predicts a number:\n{a + b * 14:.0f} points at 14 hours", xy=(14, a + b * 14), xytext=(9.3, 56),
                fontsize=10, color=MUTED, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
    a1.set_xlim(0, 14.5)
    a1.set_ylim(40, 110)
    a1.set_xlabel("Hours studied")
    a1.set_ylabel("Exam score")
    a1.set_title("Linear regression: a continuous target")
    a1.legend(frameon=False, loc="upper left")

    a2.axhspan(1, 1.4, color=WRONG, alpha=0.08, lw=0)
    a2.axhspan(-0.4, 0, color=WRONG, alpha=0.08, lw=0)
    a2.text(0.3, 1.25, "above 1: not a probability", fontsize=9.5, color=WRONG)
    a2.text(6.2, -0.27, "below 0: not a probability", fontsize=9.5, color=WRONG)
    a2.scatter(HOURS, PASSED, s=55, color=np.where(PASSED == 1, POS, NEG), zorder=4)
    a2.plot(grid, la + lb * grid, color=MUTED, lw=1.8, ls="--", label="straight line fitted to 0/1")
    a2.plot(grid, sigmoid(ga + gb * grid), color=TEXT, lw=2.2, label="logistic curve")
    a2.axhline(0.5, color=FAINT, lw=1, ls=":")
    a2.set_xlim(0, 14.5)
    a2.set_ylim(-0.4, 1.4)
    a2.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0 (failed)", "0.25", "0.5", "0.75", "1 (passed)"])
    a2.set_xlabel("Hours studied")
    a2.set_ylabel("Predicted chance of passing")
    a2.set_title("Logistic regression: a yes-or-no target")
    a2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=9.5)
    fig.suptitle("The same input, two kinds of target (made-up data): a straight line can leave 0 to 1, "
                 "the logistic curve cannot", fontsize=12)
    fig.tight_layout()
    save(fig, "reg_concept_1")


SCORES_POS = np.array([0.95, 0.90, 0.85, 0.80, 0.72, 0.66, 0.58, 0.52, 0.45, 0.38, 0.30, 0.22])
SCORES_NEG = np.array([0.70, 0.60, 0.55, 0.48, 0.42, 0.35, 0.28, 0.20, 0.15, 0.10, 0.06, 0.03])


def counts_at(t: float) -> tuple[int, int, int]:
    tp = int((SCORES_POS >= t).sum())
    fp = int((SCORES_NEG >= t).sum())
    return tp, fp, len(SCORES_POS) - tp


def sigmoid_threshold_figure():
    """The sigmoid turning a score into a probability, and the threshold trading recall for precision."""
    worked = [-2.0, 0.0, 1.0]
    print("sigmoid: p = 1 / (1 + exp(-z))")
    for z in worked:
        print(f"  z = {z:+.1f}: 1 / (1 + exp({-z:+.1f})) = 1 / (1 + {np.exp(-z):.3f}) = {sigmoid(z):.3f}")
    print(f"made-up scored examples, {len(SCORES_POS)} positive and {len(SCORES_NEG)} negative")
    print("  positive scores:", SCORES_POS.tolist())
    print("  negative scores:", SCORES_NEG.tolist())
    marks = [0.3, 0.5, 0.7]
    for t in marks:
        tp, fp, fn = counts_at(t)
        print(f"  threshold {t}: TP {tp}, FP {fp}, FN {fn}; precision {tp}/{tp + fp} = {tp / (tp + fp):.3f}, "
              f"recall {tp}/{tp + fn} = {tp / (tp + fn):.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 5), gridspec_kw={"width_ratios": [1, 1]})
    z = np.linspace(-6, 6, 300)
    a1.plot(z, sigmoid(z), color=TEXT, lw=2.2)
    a1.axhline(0.5, color=FAINT, lw=1, ls=":")
    a1.axvline(0, color=FAINT, lw=1, ls=":")
    a1.fill_between(z, 0, 1, where=z >= 0, color=POS, alpha=0.07, lw=0)
    a1.text(3.2, 0.12, "score above 0:\npredict yes", fontsize=10, color=POS, ha="center")
    a1.text(-3.2, 0.82, "score below 0:\npredict no", fontsize=10, color=NEG, ha="center")
    labels = {-2.0: ("z = −2\np = 1 / (1 + e²) = 0.119", (-5.9, 0.30)),
              0.0: ("z = 0\np = 1 / (1 + e⁰) = 0.500", (1.2, 0.33)),
              1.0: ("z = +1\np = 1 / (1 + e⁻¹) = 0.731", (2.0, 0.60))}
    for zz in worked:
        p = sigmoid(zz)
        text, xy_text = labels[zz]
        assert text.endswith(f"{p:.3f}")
        a1.scatter([zz], [p], s=60, color=TEXT, zorder=5)
        a1.annotate(text, xy=(zz, p), xytext=xy_text, fontsize=9.5, color=TEXT,
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})
    a1.text(-5.8, 0.53, "threshold 0.5", fontsize=9.5, color=MUTED)
    a1.set_ylim(-0.03, 1.03)
    a1.set_xlabel("Linear score z = b0 + b1·x1 + b2·x2 + …")
    a1.set_ylabel("Probability p")
    a1.set_title("The sigmoid turns any score into a probability")

    ts = np.linspace(0.02, SCORES_POS.max(), 200)  # above the top score nothing is predicted yes
    prec = [counts_at(t)[0] / (counts_at(t)[0] + counts_at(t)[1]) for t in ts]
    rec = [counts_at(t)[0] / len(SCORES_POS) for t in ts]
    a2.plot(ts, rec, color=POS, lw=2.2, drawstyle="steps-post", label="recall (yeses found)")
    a2.plot(ts, prec, color=NEG, lw=2.2, drawstyle="steps-post", label="precision (predicted yeses right)")
    rows = []
    for t in marks:
        tp, fp, fn = counts_at(t)
        a2.axvline(t, color=FAINT, lw=1, ls=":")
        a2.text(t, 1.02, f"t = {t}", ha="center", fontsize=9, color=MUTED)
        a2.scatter([t, t], [tp / (tp + fn), tp / (tp + fp)], s=36, color=[POS, NEG], zorder=5)
        rows.append(f"t = {t}: TP {tp}, FP {fp}, FN {fn} → precision {tp / (tp + fp):.2f}, recall {tp / (tp + fn):.2f}")
    a2.text(0.02, 0.04, "\n".join(rows), fontsize=9, color=TEXT, va="bottom", linespacing=1.5,
            bbox={"facecolor": "white", "edgecolor": "#DEDEDA", "pad": 4})
    a2.set_ylim(0, 1.25)
    a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.set_xlim(0, 1)
    a2.set_xlabel("Threshold t: predict yes when p ≥ t")
    a2.set_ylabel("Precision or recall")
    a2.set_title(f"Moving the threshold ({len(SCORES_POS)} yes, {len(SCORES_NEG)} no, made up)")
    a2.legend(frameon=False, loc="upper center", fontsize=9.5, ncols=2, bbox_to_anchor=(0.5, 1.01),
              handlelength=1.5, columnspacing=1.2)
    fig.suptitle("Logistic regression: score, probability, threshold (made-up numbers)", fontsize=12)
    fig.tight_layout()
    save(fig, "reg_concept_2")


# ---------------------------------------------------------------- AdaBoost

X10 = np.arange(10)
Y10 = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])


def best_stump(w: np.ndarray) -> tuple[float, int, float]:
    """The threshold and side (+1: predict + left of it) with the lowest weighted error."""
    best = None
    for t in np.arange(-0.5, 10, 1.0):
        for side in (1, -1):
            pred = np.where(X10 < t, side, -side)
            err = float(w[pred != Y10].sum())
            if best is None or err < best[2] - 1e-12:
                best = (float(t), side, err)
    return best


def stump_predict(t: float, side: int, x=X10) -> np.ndarray:
    return np.where(x < t, side, -side)


def adaboost_rounds(n_rounds: int = 3) -> list[dict]:
    """Discrete AdaBoost on the ten points: alpha = 1/2 ln((1 - e) / e), w <- w exp(-alpha y h) / Z.

    The weights are also kept as exact fractions, which the figure prints. With this alpha the update leaves
    half the total weight on the wrong points and half on the right ones, so a wrong point's weight becomes
    w / (2e) and a right point's w / (2(1 - e)).
    """
    w = np.full(10, 0.1)
    w_frac = [Fraction(1, 10)] * 10
    rounds = []
    for m in range(n_rounds):
        t, side, err = best_stump(w)
        pred = stump_predict(t, side)
        wrong = pred != Y10
        err_frac = sum(f for f, bad in zip(w_frac, wrong) if bad)
        assert abs(float(err_frac) - err) < 1e-12
        alpha = 0.5 * np.log((1 - err) / err)
        new = w * np.exp(-alpha * Y10 * pred)
        rounds.append({"m": m + 1, "t": t, "side": side, "error": err, "alpha": alpha, "w": w.copy(),
                       "wrong": wrong, "w_next": new / new.sum(), "w_frac": w_frac, "error_frac": err_frac})
        w = new / new.sum()
        w_frac = [f / (2 * err_frac) if bad else f / (2 * (1 - err_frac)) for f, bad in zip(w_frac, wrong)]
        assert np.allclose(w, [float(f) for f in w_frac])
    return rounds


def fraction(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def error_and_alpha(r: dict) -> tuple[str, str]:
    """ε as the sum of the wrong points' weights and α from the same fractions, in the form of the table on the
    AdaBoost page: "3 × 1/14 = 0.214" and "½ ln((11/14) / (3/14)) = 0.650"."""
    weights = [f for f, bad in zip(r["w_frac"], r["wrong"]) if bad]
    groups = {f: weights.count(f) for f in dict.fromkeys(weights)}  # weight: how many wrong points carry it
    terms = " + ".join(f"{n} × {fraction(f)}" if n > 1 else fraction(f) for f, n in groups.items())
    d = lcm(*(f.denominator for f in groups))
    k = r["error_frac"] * d
    assert k.denominator == 1
    k = k.numerator
    alpha = 0.5 * np.log((d - k) / k)
    assert f"{alpha:.3f}" == f"{r['alpha']:.3f}"
    return (f"weighted error ε = {terms} = {float(r['error_frac']):.3f}",
            f"α = ½ ln(({d - k}/{d}) / ({k}/{d})) = {alpha:.3f}")


def print_rounds(rounds: list[dict]) -> None:
    print("ten points x = 0..9, labels", Y10.tolist())
    for r in rounds:
        rule = f"x < {r['t']:.1f} -> {'+' if r['side'] == 1 else '-'}, else {'-' if r['side'] == 1 else '+'}"
        print(f"round {r['m']}: weights {np.round(r['w'], 4).tolist()}")
        print(f"  stump {rule}; wrong on x = {X10[r['wrong']].tolist()}; weighted error "
              f"{' + '.join(f'{v:.4f}' for v in r['w'][r['wrong']])} = {r['error']:.4f}; "
              f"alpha = 1/2 ln((1 - {r['error']:.4f}) / {r['error']:.4f}) = {r['alpha']:.4f}")
        print(f"  next weights {np.round(r['w_next'], 4).tolist()}")
        print(f"  as fractions: weights {[fraction(f) for f in r['w_frac']]}; " + "; ".join(error_and_alpha(r)))
    score = sum(r["alpha"] * stump_predict(r["t"], r["side"]) for r in rounds)
    print(f"final score sum(alpha_m h_m(x)) = {np.round(score, 4).tolist()}")
    print(f"sign = {np.sign(score).astype(int).tolist()}; correct {int((np.sign(score) == Y10).sum())} of 10")


def draw_points(ax, w, wrong=None, y=0.0, scale=2600):
    for cls, color, marker in [(1, POS, "o"), (-1, NEG, "s")]:
        m = Y10 == cls
        ax.scatter(X10[m], np.full(m.sum(), y), s=w[m] * scale + 25, color=color, marker=marker, zorder=4,
                   edgecolor="white", lw=0.8)
    if wrong is not None and wrong.any():
        ax.scatter(X10[wrong], np.full(wrong.sum(), y), s=w[wrong] * scale + 260, facecolor="none", edgecolor=WRONG,
                   lw=2, zorder=5)


def round_panel(ax, r: dict, title: str, before: dict | None = None):
    ax.axvspan(-0.8, r["t"], color=POS if r["side"] == 1 else NEG, alpha=0.08, lw=0)
    ax.axvspan(r["t"], 9.8, color=NEG if r["side"] == 1 else POS, alpha=0.08, lw=0)
    ax.axvline(r["t"], color=TEXT, lw=2, ls="--")
    draw_points(ax, r["w"], r["wrong"])
    left, right = ("+", "−") if r["side"] == 1 else ("−", "+")
    ax.text(r["t"] - 0.15, 0.7, f"predict {left}", ha="right", fontsize=10, color=TEXT)
    ax.text(r["t"] + 0.15, 0.7, f"predict {right}", ha="left", fontsize=10, color=TEXT)
    wrong_x = ", ".join(str(v) for v in X10[r["wrong"]])
    lines = [f"stump {r['m']}: split at x = {r['t']:.1f}; wrong on x = {wrong_x} (ringed)", *error_and_alpha(r)]
    if before is not None:
        grew = X10[before["wrong"]]
        lines.insert(0, f"last round's mistakes x = {', '.join(map(str, grew))}: weight "
                        f"{fraction(before['w_frac'][grew[0]])} → {fraction(r['w_frac'][grew[0]])} each")
    ax.text(-0.7, -0.42, "\n".join(lines), fontsize=9.5, color=TEXT, va="top", linespacing=1.45,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
    ax.set_xlim(-0.8, 9.8)
    ax.set_ylim(-1.15, 0.95)
    ax.set_yticks([])
    ax.set_xticks(X10)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=11, loc="left")


def adaboost_rounds_figure(rounds: list[dict]):
    """Three rounds on ten points, weights as dot size, and the weighted vote that classifies all ten."""
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 8.6))
    titles = ["Round 1: every point weighs 1/10",
              "Round 2: round 1's mistakes now weigh more",
              "Round 3: round 2's mistakes now weigh more", "Final weighted vote"]
    for i, (ax, r, title) in enumerate(zip(axes.flat[:3], rounds, titles)):
        round_panel(ax, r, title, rounds[i - 1] if i else None)

    ax = axes.flat[3]
    rows = [(r["alpha"], stump_predict(r["t"], r["side"]), f"α{r['m']} = {r['alpha']:.3f}") for r in rounds]
    score = sum(a * h for a, h, _ in rows)
    for i, (a, h, label) in enumerate(rows):
        yv = 3 - i
        for x, v in zip(X10, h):
            ax.add_patch(FancyBboxPatch((x - 0.42, yv - 0.32), 0.84, 0.64, boxstyle="round,pad=0.02,rounding_size=0.1",
                                        color=POS if v > 0 else NEG, alpha=0.18 + 0.5 * a / max(rr[0] for rr in rows),
                                        lw=0))
            ax.text(x, yv, "+" if v > 0 else "−", ha="center", va="center", fontsize=11, color=TEXT)
        ax.text(-0.75, yv, f"stump {i + 1}\n{label}", ha="right", va="center", fontsize=9.5, color=TEXT)
    for x, s in zip(X10, score):
        ax.text(x, -0.05, f"{s:+.2f}", ha="center", va="center", fontsize=8.6, color=TEXT)
        ok = np.sign(s) == Y10[x]
        ax.text(x, -0.75, ("+" if s > 0 else "−") + (" ✓" if ok else " ✗"), ha="center", va="center", fontsize=10.5,
                color=POS if s > 0 else NEG, fontweight="bold")
    ax.text(-0.75, -0.05, "Σ α·vote", ha="right", va="center", fontsize=9.5, color=TEXT)
    ax.text(-0.75, -0.75, "final", ha="right", va="center", fontsize=9.5, color=TEXT, fontweight="bold")
    for x, yy in zip(X10, Y10):
        ax.text(x, -1.35, "+" if yy > 0 else "−", ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(-0.75, -1.35, "true", ha="right", va="center", fontsize=9.5, color=MUTED)
    ax.set_xlim(-3.2, 9.6)
    ax.set_ylim(-1.75, 3.6)
    ax.axis("off")
    ax.set_title(titles[3] + f": sign(α1·h1 + α2·h2 + α3·h3) gets {int((np.sign(score) == Y10).sum())} of 10 right",
                 fontsize=11, loc="left")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=POS, ms=9, label="class +"),
                        plt.Line2D([], [], marker="s", ls="", color=NEG, ms=9, label="class −"),
                        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=WRONG, mew=2, ms=12,
                                   label="wrong this round")],
               loc="lower center", ncols=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("AdaBoost on ten made-up points: dot size is each point's weight; no single split separates them",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "ab_concept_2")


def box(ax, x, y, w, h, text, color, fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none",
                                alpha=0.16))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=TEXT)


def arrow(ax, start, end, color=MUTED, text=None, offset=(0, 0.13), rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.5,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], text, ha="center",
                fontsize=9, color=color)


def mini_points(ax, x0, y0, weights, wrong=None, width=2.0, scale=900):
    """The ten points as a strip of small markers; marker area follows the weight."""
    xs = x0 + np.linspace(0.1, width - 0.1, 10)
    for i, (x, wv) in enumerate(zip(xs, weights)):
        if wv <= 0:
            ax.scatter([x], [y0], s=10, facecolor="none", edgecolor=FAINT, lw=0.8, zorder=4)
            continue
        ax.scatter([x], [y0], s=wv * scale + 6, color=POS if Y10[i] > 0 else NEG, marker="o" if Y10[i] > 0 else "s",
                   zorder=4, lw=0)
        if wrong is not None and wrong[i]:
            ax.scatter([x], [y0], s=wv * scale + 90, facecolor="none", edgecolor=WRONG, lw=1.4, zorder=5)


def bagging_boosting_figure(rounds: list[dict]):
    """Bagging trains on independent resamples at the same time; boosting trains each model on weights set by
    the one before."""
    rng = np.random.default_rng(4)
    samples = [np.bincount(rng.integers(0, 10, 10), minlength=10) for _ in range(3)]
    print("bagging: three bootstrap samples of the ten points (how many times each point is drawn)")
    for i, s in enumerate(samples, 1):
        print(f"  sample {i}: {s.tolist()}  (points left out: {np.flatnonzero(s == 0).tolist()})")
    print("boosting: the weights each stump trains on are the ones printed for ab_concept_2")

    fig, ax = plt.subplots(figsize=(11.1, 7.4))
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    ax.text(0.15, 7.3, "Bagging (as in a random forest): models built independently, at the same time",
            fontsize=11.5, color=TEXT, fontweight="bold")
    box(ax, 0.15, 4.95, 2.3, 1.05, "", MUTED)
    ax.text(1.3, 5.72, "training data (10 points)", ha="center", fontsize=9.5, color=TEXT)
    mini_points(ax, 0.3, 5.25, np.full(10, 0.1), width=2.0)
    for i, (s, yb) in enumerate(zip(samples, [6.3, 5.3, 4.3])):
        box(ax, 3.2, yb - 0.08, 2.6, 0.82, "", PALETTE[2])
        ax.text(3.3, yb + 0.5, f"resample {i + 1}: drawn with replacement", fontsize=8.8, color=TEXT)
        mini_points(ax, 3.35, yb + 0.16, s / 10, width=2.3)
        arrow(ax, (2.45, 5.47), (3.2, yb + 0.33))
        box(ax, 6.45, yb - 0.02, 1.35, 0.7, f"tree {i + 1}", PALETTE[2])
        arrow(ax, (5.8, yb + 0.33), (6.45, yb + 0.33))
        arrow(ax, (7.8, yb + 0.33), (8.85, 5.47))
    box(ax, 8.85, 4.95, 2.3, 1.05, "majority vote\n(each tree counts once)", PALETTE[2])
    ax.text(5.65, 3.92, "no tree sees another tree's results, so all three can be trained at once "
            "(hollow dots: points a resample left out)", fontsize=9.3, color=MUTED, ha="center")

    ax.plot([0.15, 11.15], [3.62, 3.62], color="#DEDEDA", lw=1)
    ax.text(0.15, 3.3, "Boosting (AdaBoost): models built one after another, each on weights set by the last",
            fontsize=11.5, color=TEXT, fontweight="bold")
    ax.text(0.15, 2.98, "stump 2 cannot be trained until stump 1 has run: the order matters", fontsize=9.5,
            color=MUTED)
    xs = [0.15, 3.05, 5.95]
    for i, (r, x0) in enumerate(zip(rounds, xs)):
        box(ax, x0, 1.85, 2.3, 0.95, "", POS)
        ax.text(x0 + 1.15, 2.55, f"weights for stump {i + 1}", ha="center", fontsize=8.8, color=TEXT)
        mini_points(ax, x0 + 0.15, 2.15, r["w"], wrong=r["wrong"], width=2.0)
        box(ax, x0 + 0.55, 0.75, 1.2, 0.72, f"stump {i + 1}\nα{i + 1} = {r['alpha']:.2f}", POS, fontsize=9.5)
        arrow(ax, (x0 + 1.15, 1.85), (x0 + 1.15, 1.47))
        if i < 2:
            arrow(ax, (x0 + 1.75, 1.11), (xs[i + 1], 2.3), color=WRONG, rad=0.3)
            ax.text(x0 + 2.42, 1.0, "its mistakes\nweigh more", ha="center", fontsize=9, color=WRONG)
    box(ax, 8.85, 0.75, 2.3, 1.35, "weighted vote\nα1·h1 + α2·h2 + α3·h3\n(better stumps count more)", POS,
        fontsize=9.5)
    bus = 0.42  # every stump's vote runs along one line into the vote box
    for x0 in xs:
        ax.plot([x0 + 1.15, x0 + 1.15], [0.75, bus], color=FAINT, lw=1.5)
    ax.plot([xs[0] + 1.15, 9.6], [bus, bus], color=FAINT, lw=1.5)
    arrow(ax, (9.6, bus), (9.6, 0.75), color=FAINT)
    ax.text(5.2, bus - 0.2, "each stump's vote, weighted by its α", ha="center", fontsize=9, color=MUTED)
    fig.suptitle("Two ways to combine many simple models (dots: the ten made-up points of the next figure; "
                 "dot size = weight)", fontsize=11.5)
    fig.tight_layout()
    save(fig, "ab_concept_1")


def main() -> None:
    apply_style()
    print("=== reg_concept_1")
    linear_vs_logistic_figure()
    print("\n=== reg_concept_2")
    sigmoid_threshold_figure()
    rounds = adaboost_rounds()
    print("\n=== ab_concept_2")
    print_rounds(rounds)
    adaboost_rounds_figure(rounds)
    print("\n=== ab_concept_1")
    bagging_boosting_figure(rounds)


if __name__ == "__main__":
    main()
