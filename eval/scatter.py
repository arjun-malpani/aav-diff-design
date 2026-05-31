"""Per-sequence scatter plots: novelty (distance to training set) vs. fitness.

Each point is one generated sequence, placed at
    x = min edit distance to the training set  (novelty; 0 = memorized a training seq)
    y = predicted fitness from the held-out ESM-2 judge.

Two views (saved as separate files):
  - by guidance scale w  -> does turning guidance up push fitness up (and where)?
  - by cohort            -> do conditioned sequences stay fit further from the data
                            than unconditional / random / training?

novelty (vs. the whole training set) is per-sequence and so scatters, unlike
internal pairwise diversity (a set-level number; see diversity_by_cohort).
"""
import numpy as np

from common import save_figure, setup_style, viable_threshold


def _scatter(ax, groups, colors):
    """groups: list of (label, novelty array, fitness array). Plots each as points."""
    for (label, novelty, fitness), color in zip(groups, colors):
        # small horizontal jitter so integer edit distances don't form hard columns
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, size=len(novelty))
        ax.scatter(novelty + jitter, fitness, s=8, alpha=0.35, color=color,
                   edgecolors="none", label=label)
    ax.axhline(viable_threshold(), color="k", ls="--", lw=1, alpha=0.6,
               label=f"viable threshold ({viable_threshold():.2f})")
    ax.set(xlabel="novelty: min edit distance to training set",
           ylabel="predicted fitness (judge)")
    leg = ax.legend(fontsize=9, markerscale=2)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)


def plot_by_w(w_cohorts, out_dir):
    """w_cohorts: dict {w: cohort}. One color per guidance scale."""
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ws = sorted(w_cohorts)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(ws)))
    groups = [(f"w={w}", w_cohorts[w]["novelty"], w_cohorts[w]["fitness"]) for w in ws]
    _scatter(ax, groups, colors)
    ax.set_title("Novelty vs. fitness, by guidance scale (target fitness = 6)")
    fig.tight_layout()
    return save_figure(fig, "scatter_by_w.png", out_dir)


def plot_by_cohort(cohorts, out_dir):
    """cohorts: dict {name: cohort}. One color per cohort."""
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    names = list(cohorts)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(names)))
    groups = [(cohorts[n]["label"], cohorts[n]["novelty"], cohorts[n]["fitness"]) for n in names]
    _scatter(ax, groups, colors)
    ax.set_title("Novelty vs. fitness, by cohort")
    fig.tight_layout()
    return save_figure(fig, "scatter_by_cohort.png", out_dir)
