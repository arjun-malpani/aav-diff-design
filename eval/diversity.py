"""Diversity and novelty evaluation.

Mirrors AAVDiff's diversity analysis (Liu 2024, Fig 1): mutation-count and length
distributions, and novelty as the per-sequence min edit distance to the training
set (0 = the model reproduced a training sequence). Adds a per-cohort internal
diversity bar (mean pairwise edit distance) -- the set-level "how varied are the
sequences among themselves" number, which catches mode collapse (high novelty but
near-identical samples).

Plotting functions consume pre-built cohort dicts (see viability.py for the shape).
"""
import numpy as np

from common import mean_pairwise_edit_distance, save_figure, setup_style


def plot_histograms(cohort, out_dir):
    """Mutation-count, length, and novelty distributions for one cohort."""
    import matplotlib.pyplot as plt
    setup_style()
    muts = np.asarray(cohort["mutations"])
    lengths = np.array([len(s) for s in cohort["seqs"]])
    novelty = np.asarray(cohort["novelty"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(muts, bins=range(0, int(muts.max()) + 2), color="#4C72B0", alpha=0.85)
    axes[0].set(xlabel="mutations from wild type", ylabel="count",
                title="Mutation-count distribution")

    axes[1].hist(lengths, bins=range(int(lengths.min()), int(lengths.max()) + 2),
                 color="#55A868", alpha=0.85)
    axes[1].axvline(28, color="k", ls="--", lw=1, label="WT length (28)")
    axes[1].set(xlabel="sequence length", ylabel="count", title="Length distribution")
    axes[1].legend(fontsize=9)

    axes[2].hist(novelty, bins=range(0, int(novelty.max()) + 2), color="#C44E52", alpha=0.85)
    axes[2].set(xlabel="min edit distance to training set", ylabel="count",
                title="Novelty (0 = memorized)")
    fig.suptitle(f"Diversity & novelty — {cohort['label']}", y=1.02)
    fig.tight_layout()
    return save_figure(fig, "diversity.png", out_dir)


def plot_by_cohort(cohorts, out_dir, seed=0):
    """Bar of internal diversity (mean pairwise edit distance) per cohort."""
    import matplotlib.pyplot as plt
    setup_style()
    names = list(cohorts)
    labels = [cohorts[n]["label"] for n in names]
    diversities = [mean_pairwise_edit_distance(cohorts[n]["seqs"], seed=seed) for n in names]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, diversities, color=colors)
    ax.set(ylabel="mean pairwise edit distance", title="Internal diversity by cohort")
    ax.tick_params(axis="x", rotation=20)
    for i, d in enumerate(diversities):
        ax.text(i, d + 0.05, f"{d:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "diversity_by_cohort.png", out_dir), dict(zip(names, diversities))
