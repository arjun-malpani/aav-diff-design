"""Distribution-match metrics: does the generator reproduce the statistics of real
viable sequences (without memorizing them)?

Mirrors AAVDiffusion's Fig 4 "intrinsic relationships" battery, restricted to the
parts that are cheap and judge-independent and that our aligned canvas gives for
free (no MSA needed -- the 28 anchor slots are already aligned to wild type):

  - per-anchor Shannon entropy  : does the model conserve the positions nature
                                  conserves and vary the ones it varies?
                                  (reported as Pearson r between the two profiles)
  - per-anchor amino-acid JSD   : is the residue composition right at each position?
                                  (Jensen-Shannon divergence; 0 = identical)
  - t-SNE overlap               : do generated & real occupy the same sequence
                                  space (visual; not a number)

The reference is VIABLE training sequences -- the distribution the generator is
asked to match.
"""
import numpy as np

from common import (pearson, per_anchor_entropy, per_anchor_jsd, save_figure,
                    setup_style, tsne_2d)


def plot_entropy(gen_canvas, real_canvas, out_dir):
    """Per-anchor entropy profiles for generated vs. real, with their correlation."""
    import matplotlib.pyplot as plt
    setup_style()
    gen_h = per_anchor_entropy(gen_canvas)
    real_h = per_anchor_entropy(real_canvas)
    r = pearson(gen_h, real_h)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(1, len(gen_h) + 1)
    ax.plot(positions, real_h, "-o", color="#4C72B0", markersize=4, label="real viable")
    ax.plot(positions, gen_h, "-o", color="#C44E52", markersize=4, label="generated")
    ax.set(xlabel="anchor position (WT residue 1-28)", ylabel="Shannon entropy (nats)",
           title=f"Per-position conservation: generated vs. real  (Pearson r = {r:.3f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "distribution_entropy.png", out_dir), r


def plot_jsd(gen_canvas, real_canvas, out_dir):
    """Per-anchor amino-acid composition divergence (JSD) between generated and real."""
    import matplotlib.pyplot as plt
    setup_style()
    jsd = per_anchor_jsd(gen_canvas, real_canvas)
    mean_jsd = float(jsd.mean())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(1, len(jsd) + 1)
    ax.bar(positions, jsd, color="#55A868")
    ax.axhline(mean_jsd, color="k", ls="--", lw=1, label=f"mean = {mean_jsd:.3f}")
    ax.set(xlabel="anchor position (WT residue 1-28)",
           ylabel="amino-acid JSD (nats; 0 = identical)",
           title="Per-position amino-acid composition divergence (generated vs. real)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "distribution_jsd.png", out_dir), mean_jsd


def plot_tsne(gen_canvas, real_canvas, out_dir, seed=0):
    """t-SNE of anchor one-hots: generated and real overlaid to show overlap."""
    import matplotlib.pyplot as plt
    setup_style()
    # cap each group for speed; t-SNE on a few thousand points is plenty
    cap = 1500
    g = gen_canvas[:cap]
    r = real_canvas[:cap]
    coords, labels = tsne_2d([r, g], seed=seed)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for group_index, (name, color) in enumerate([("real viable", "#4C72B0"),
                                                 ("generated", "#C44E52")]):
        sel = labels == group_index
        ax.scatter(coords[sel, 0], coords[sel, 1], s=6, alpha=0.4, color=color,
                   edgecolors="none", label=name)
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2",
           title="Sequence-space overlap (t-SNE of anchor composition)")
    ax.grid(False)
    leg = ax.legend(fontsize=9, markerscale=2)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    fig.tight_layout()
    return save_figure(fig, "distribution_tsne.png", out_dir)
