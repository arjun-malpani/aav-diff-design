"""t-SNE of sequence space, colored by predicted viability/fitness score.

Complements distribution_tsne.png (which colors by group: generated vs real).
Here each point is colored by its ESM-2 judge fitness, so you can see whether
high-fitness sequences cluster in sequence space -- and whether the generated and
real-viable manifolds carry fitness the same way.

Produces two panels in eval/figures/tsne_score/:
  - generated, colored by predicted fitness
  - real viable, colored by predicted fitness
on a SHARED t-SNE embedding (both sets embedded together) and a SHARED color
scale, so the two panels are directly comparable.

Both cohorts are scored by the same judge for a comparable color scale (real
viable are all viable by label, so their scores cluster high -- expected).

This regenerates one cohort (conditioned, w=2, target=6), scores it, saves the
arrays to eval/data/ so future re-plots need no diffusion run, then plots.

    python tsne_by_score.py -n 2000        # MPS, ~10-12 min for one cohort
    python tsne_by_score.py --smoke
"""
import argparse
import time
from pathlib import Path

import numpy as np

from common import (FIGURES, ROOT, get_device, load_generator, load_judge,
                    load_training, save_figure, score_fitness, setup_style,
                    tokenizer, tsne_2d, viable_threshold)
from config import Config
from denoising import generate

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET_FITNESS = 6.0
GUIDANCE = 2  # the canonical "conditioned" cohort (w=2)


def generate_and_score(n, steps, device, seed):
    """Generate the conditioned cohort + load real viable; score both with the judge.

    Returns a dict of arrays, also persisted to DATA_DIR so re-plots need no rerun.
    """
    config = Config()
    config.sampler.num_steps = steps
    config.sampler.guidance_scale = GUIDANCE
    tok = tokenizer()

    generator = load_generator(device, config)
    judge, judge_tok = load_judge(device)

    gen_ids = generate(generator, n, fitness=TARGET_FITNESS, config=config, seed=seed,
                       return_ids=True).cpu().numpy()
    gen_seqs = [tok.decode(row) for row in gen_ids]
    gen_fitness = score_fitness(gen_seqs, judge, judge_tok, device)

    real_canvas, real_seqs = load_training(viable_only=True, n=n, seed=seed)
    real_fitness = score_fitness(real_seqs, judge, judge_tok, device)

    bundle = {
        "gen_canvas": gen_ids, "gen_fitness": gen_fitness,
        "real_canvas": real_canvas, "real_fitness": real_fitness,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "tsne_cohort.npz", **bundle)
    print(f"  saved arrays -> {DATA_DIR / 'tsne_cohort.npz'}")
    return bundle


# high fitness -> dark/bold so the most-viable points stand out even when sparse
# (plain viridis fades them to hard-to-see yellow); reversed magma runs
# light(low) -> dark(high).
CMAP = "magma_r"


def plot(bundle, out_dir, seed=0):
    """Compute the shared t-SNE embedding once, emit all three views from it."""
    import matplotlib.pyplot as plt
    setup_style()

    cap = min(len(bundle["gen_canvas"]), len(bundle["real_canvas"]))
    gen_c, real_c = bundle["gen_canvas"][:cap], bundle["real_canvas"][:cap]
    gen_f, real_f = bundle["gen_fitness"][:cap], bundle["real_fitness"][:cap]

    coords, labels = tsne_2d([real_c, gen_c], seed=seed)  # real=0, generated=1
    real_xy, gen_xy = coords[labels == 0], coords[labels == 1]
    all_f = np.concatenate([gen_f, real_f])
    vmin, vmax = np.percentile(all_f, [2, 98])  # shared robust color scale

    paths = []
    paths.append(_plot_sidebyside(gen_xy, gen_f, real_xy, real_f, vmin, vmax, out_dir))
    paths.append(_plot_overlap(gen_xy, gen_f, real_xy, real_f, vmin, vmax, out_dir))
    paths.append(_plot_hexbin(gen_xy, gen_f, real_xy, real_f, vmin, vmax, out_dir))
    return paths


def _add_cbar(fig, sc, axes):
    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("predicted fitness (judge)")
    cbar.ax.axhline(viable_threshold(), color="red", lw=1)  # viability boundary
    return cbar


def _plot_sidebyside(gen_xy, gen_f, real_xy, real_f, vmin, vmax, out_dir):
    """Two panels: generated | real, each colored by fitness (shared scale)."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, xy, f, title in [(ax1, gen_xy, gen_f, "Generated (conditioned, w=2)"),
                             (ax2, real_xy, real_f, "Real viable (Bryant)")]:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=f, s=10, alpha=0.7, cmap=CMAP,
                        vmin=vmin, vmax=vmax, edgecolors="none")
        ax.set(xlabel="t-SNE 1", title=title)
        ax.grid(False)
    ax1.set_ylabel("t-SNE 2")
    _add_cbar(fig, sc, [ax1, ax2])
    fig.suptitle("t-SNE by predicted fitness — side by side (shared embedding & scale)", y=1.0)
    return save_figure(fig, "tsne_by_score.png", out_dir)


def _plot_overlap(gen_xy, gen_f, real_xy, real_f, vmin, vmax, out_dir):
    """Single panel: both sets overlaid, color = fitness, marker = which set."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.scatter(real_xy[:, 0], real_xy[:, 1], c=real_f, marker="o", s=22, alpha=0.55,
               cmap=CMAP, vmin=vmin, vmax=vmax, edgecolors="none", label="real viable")
    sc = ax.scatter(gen_xy[:, 0], gen_xy[:, 1], c=gen_f, marker="^", s=22, alpha=0.7,
                    cmap=CMAP, vmin=vmin, vmax=vmax, edgecolors="black", linewidths=0.2,
                    label="generated (w=2)")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2",
           title="t-SNE by predicted fitness — overlap (○ real, △ generated)")
    ax.grid(False)
    ax.legend(fontsize=9, markerscale=1.5)
    _add_cbar(fig, sc, ax)
    fig.tight_layout()
    return save_figure(fig, "tsne_overlap.png", out_dir)


def _plot_hexbin(gen_xy, gen_f, real_xy, real_f, vmin, vmax, out_dir):
    """Two hexbin panels: cell color = MEAN fitness of points inside; empty = gray."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    extent = (min(gen_xy[:, 0].min(), real_xy[:, 0].min()),
              max(gen_xy[:, 0].max(), real_xy[:, 0].max()),
              min(gen_xy[:, 1].min(), real_xy[:, 1].min()),
              max(gen_xy[:, 1].max(), real_xy[:, 1].max()))
    for ax, xy, f, title in [(ax1, gen_xy, gen_f, "Generated (conditioned, w=2)"),
                             (ax2, real_xy, real_f, "Real viable (Bryant)")]:
        ax.set_facecolor("0.85")  # gray where no points fall
        hb = ax.hexbin(xy[:, 0], xy[:, 1], C=f, reduce_C_function=np.mean,
                       gridsize=55, cmap=CMAP, vmin=vmin, vmax=vmax, extent=extent,
                       mincnt=1)
        ax.set(xlabel="t-SNE 1", title=title)
        ax.grid(False)
    ax1.set_ylabel("t-SNE 2")
    _add_cbar(fig, hb, [ax1, ax2])
    fig.suptitle("t-SNE hexbin — cell color = mean predicted fitness (gray = empty)", y=1.0)
    return save_figure(fig, "tsne_hexbin.png", out_dir)


def main():
    p = argparse.ArgumentParser(description="t-SNE colored by predicted fitness (standalone).")
    p.add_argument("-n", "--num", type=int, default=10000, help="sequences per cohort")
    p.add_argument("--steps", type=int, default=256, help="reverse sampling steps")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse", action="store_true",
                   help="skip generation; replot from saved eval/data/tsne_cohort.npz")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    n, steps = args.num, args.steps
    if args.smoke:
        n, steps = 60, 16

    t0 = time.time()
    if args.reuse:
        bundle = dict(np.load(DATA_DIR / "tsne_cohort.npz", allow_pickle=True))
        print(f"reusing saved arrays ({len(bundle['gen_canvas'])} gen, "
              f"{len(bundle['real_canvas'])} real)")
    else:
        device = get_device(args.device)
        print(f"device={device} | n={n} steps={steps} | w={GUIDANCE} target={TARGET_FITNESS}")
        bundle = generate_and_score(n, steps, device, args.seed)

    paths = plot(bundle, FIGURES / "tsne_score", seed=args.seed)
    g, r = bundle["gen_fitness"], bundle["real_fitness"]
    print(f"\ngenerated fitness: mean={g.mean():+.2f} viable={ (g>viable_threshold()).mean():.3f}")
    print(f"real viable fitness: mean={r.mean():+.2f} viable={ (r>viable_threshold()).mean():.3f}")
    print(f"done in {time.time() - t0:.1f}s")
    for path in paths:
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
