"""Conditioning calibration: if we ASK for fitness X, do we GET fitness X?

This is the dose-response test of the conditional generator -- distinct from the
guidance-scale scatter (which varies w at a fixed target). Here we hold the
sampler fixed and sweep the requested FITNESS TARGET, generate a cohort at each,
score with the held-out ESM-2 judge, and plot requested-vs-achieved fitness with
the y=x ideal line. A well-calibrated model tracks y=x; a model that only learned
"viable vs not" saturates into a flat line.

We sweep at a couple of guidance scales so you can see whether more guidance
improves or degrades calibration (the full run showed higher w hurts viability).

Run standalone -- it writes only eval/figures/calibration/ and touches nothing else:

    python calibration.py            # n=1000, 256 steps, w in {1,2}
    python calibration.py --smoke    # tiny fast check
"""
import argparse
import time

import numpy as np

from common import (FIGURES, get_device, load_generator, load_judge, pearson,
                    save_figure, score_fitness, setup_style, viable_threshold)
from config import Config
from denoising import generate

# Requested targets span the data's fitness range (raw viral_selection units;
# the distribution runs ~ -11..+9.5, viable mean ~ +0.6). Standardization clamps
# at +/-4 std, so targets beyond ~ +11 saturate -- keep within the learned range.
TARGETS = [-4, -2, 0, 2, 4, 6, 8]
GUIDANCE_SCALES = [1, 2]


def evaluate(generator, judge, judge_tok, device, config, n, seed):
    """Sweep (target x w); return rows of {target, w, mean_pred, std_pred}."""
    rows = []
    for w in GUIDANCE_SCALES:
        config.sampler.guidance_scale = w
        for target in TARGETS:
            seqs = generate(generator, n, fitness=target, config=config, seed=seed)
            preds = score_fitness(seqs, judge, judge_tok, device)
            rows.append({"target": target, "w": w,
                         "mean_pred": float(np.mean(preds)),
                         "std_pred": float(np.std(preds)),
                         "viable_frac": float((preds > viable_threshold()).mean())})
            print(f"  target={target:+d} w={w}: mean predicted={np.mean(preds):+.3f} "
                  f"viable={(preds > viable_threshold()).mean():.3f}")
    return rows


def plot(rows, out_dir):
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ws = sorted({r["w"] for r in rows})
    colors = plt.cm.plasma(np.linspace(0.15, 0.7, len(ws)))

    for w, color in zip(ws, colors):
        sub = sorted([r for r in rows if r["w"] == w], key=lambda r: r["target"])
        targets = [r["target"] for r in sub]
        means = [r["mean_pred"] for r in sub]
        stds = [r["std_pred"] for r in sub]
        r = pearson(targets, means)
        ax.errorbar(targets, means, yerr=stds, fmt="-o", color=color, capsize=3,
                    markersize=5, label=f"w={w}  (Pearson r={r:.3f})")

    lo = min(r["target"] for r in rows)
    hi = max(r["target"] for r in rows)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x (perfect calibration)")
    ax.axhline(viable_threshold(), color="gray", ls=":", lw=1, alpha=0.7,
               label=f"viable threshold ({viable_threshold():.2f})")
    ax.set(xlabel="requested target fitness (raw viral_selection)",
           ylabel="achieved predicted fitness (judge)",
           title="Conditioning calibration: requested vs. achieved fitness")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "calibration.png", out_dir)


def main():
    p = argparse.ArgumentParser(description="Conditioning calibration sweep (standalone).")
    p.add_argument("-n", "--num", type=int, default=1000, help="sequences per (target, w) cell")
    p.add_argument("--steps", type=int, default=256, help="reverse sampling steps")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny fast check")
    args = p.parse_args()

    n, steps = args.num, args.steps
    if args.smoke:
        n, steps = 16, 12

    device = get_device(args.device)
    config = Config()
    config.sampler.num_steps = steps
    print(f"device={device} | n={n} steps={steps} | targets={TARGETS} | w={GUIDANCE_SCALES}")

    t0 = time.time()
    generator = load_generator(device, config)
    judge, judge_tok = load_judge(device)
    print(f"models loaded ({time.time() - t0:.1f}s)\n")

    rows = evaluate(generator, judge, judge_tok, device, config, n, args.seed)
    path = plot(rows, FIGURES / "calibration")
    print(f"\ndone in {time.time() - t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
