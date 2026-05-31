"""Viability evaluation: how often does the generator produce sequences the
held-out ESM-2 judge predicts as viable, vs. baselines, and how does that vary
with distance from wild type?

Mirrors AAVDiff's headline analysis (Liu 2024, Fig 2): % viable as a function of
mutation count, with a random-mutation baseline that collapses as mutations grow.
"Viable" = predicted fitness above Bryant's recovered viability threshold (see
common.viable_threshold), NOT an arbitrary 0.

Plotting functions consume pre-built cohort dicts (generated once by run_all so
diverse/precise presets share the same reference + seed). A cohort dict has:
    name, label, seqs, canvas, fitness, mutations, novelty
"""
import numpy as np

from common import save_figure, setup_style, viable_threshold


def viable_fraction(fitness, threshold=None):
    threshold = viable_threshold() if threshold is None else threshold
    return float((np.asarray(fitness) > threshold).mean()) if len(fitness) else 0.0


def plot(cohorts, out_dir):
    """cohorts: dict {name: cohort}. Bar of viable fraction + viability vs mutations."""
    import matplotlib.pyplot as plt
    setup_style()
    threshold = viable_threshold()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    names = list(cohorts)
    labels = [cohorts[n]["label"] for n in names]
    fracs = [viable_fraction(cohorts[n]["fitness"], threshold) for n in names]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    ax1.bar(labels, fracs, color=colors)
    ax1.set(ylabel="fraction predicted viable", title="Viability by cohort", ylim=(0, 1))
    ax1.tick_params(axis="x", rotation=20)
    for i, f in enumerate(fracs):
        ax1.text(i, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)

    bins = np.arange(0, 31, 3)
    centers = (bins[:-1] + bins[1:]) / 2
    for name, color in zip(names, colors):
        fitness = np.asarray(cohorts[name]["fitness"])
        muts = np.asarray(cohorts[name]["mutations"])
        if not len(fitness):
            continue
        frac_per_bin = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (muts >= lo) & (muts < hi)
            frac_per_bin.append((fitness[sel] > threshold).mean() if sel.sum() else np.nan)
        ax2.plot(centers, frac_per_bin, "-o", label=cohorts[name]["label"],
                 color=color, markersize=4)
    ax2.set(xlabel="mutations from wild type (edit distance)",
            ylabel="fraction predicted viable", title="Viability vs. divergence", ylim=(0, 1))
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return save_figure(fig, "viability.png", out_dir)
