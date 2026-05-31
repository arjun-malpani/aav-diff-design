"""Run the full diffusion eval suite under two sampler presets.

For each preset (diverse, precise) it generates the cohorts ONCE, scores them with
the held-out ESM-2 judge, and produces every figure into eval/figures/<preset>/:

    viability.png            bar + viability-vs-divergence line (AAVDiff Fig 2 style)
    diversity.png            mutation-count / length / novelty histograms
    diversity_by_cohort.png  internal diversity (mean pairwise edit distance) per cohort
    scatter_by_w.png         per-sequence novelty vs fitness, colored by guidance w
    scatter_by_cohort.png    per-sequence novelty vs fitness, colored by cohort
    distribution_entropy.png per-anchor conservation, generated vs real (Pearson r)
    distribution_jsd.png     per-anchor amino-acid composition divergence
    distribution_tsne.png    sequence-space overlap (t-SNE)

The two presets SHARE the training reference, the random baseline, and the RNG
seed -- so the starting noise is identical and the only difference between diverse
and precise is the sampler (decoding rule / commit rule / temperature).

    python run_all.py                 # full run: n=1000, 256 steps, w in {1,2,4,8}
    python run_all.py --smoke         # tiny fast correctness check
    python run_all.py --preset diverse  # just one preset

Compute-heavy: each preset runs 5 generations (w in {1,2,4,8} + unconditional) of
n x 256-step sampling, then the judge. Expect ~45 min/preset on MPS at n=1000.
"""
import argparse
import time

import numpy as np
import torch

import diversity
import distribution
import scatter
import viability
from common import (FIGURES, get_device, load_generator, load_judge, load_training,
                    mutation_counts, novelty_to_training, score_fitness, tokenizer,
                    viable_threshold, wild_type_sequence)
from config import Config
from denoising import generate

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TARGET_FITNESS = 6.0          # the "high fitness" ask, in raw viral_selection units
GUIDANCE_SCALES = [1, 2, 4, 8]
CONDITIONED_WS = [2, 4]       # which w's define the "conditioned" cohorts (reused from the sweep)

PRESETS = {
    "diverse": {"decoding": "random", "commit": "sample", "temperature": 1.0},
    # MaskGIT-style mode-seeking preset: confidence ordering + low temperature is
    # much less diverse than 'diverse', but still stochastic. NOTE: commit='argmax'
    # would be FULLY deterministic here (argmax ignores temperature, and confidence
    # ranking is temperature-invariant) -> every sequence identical -> degenerate
    # diversity/entropy. So we sample at low temperature instead of argmax.
    "precise": {"decoding": "confidence", "commit": "sample", "temperature": 0.8},
}


def random_variants(n, wt, n_mut_range=(1, 20), seed=0):
    """Baseline: wild type with k random substitutions (AAVDiff's random baseline)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(n_mut_range[0], n_mut_range[1] + 1)
        chars = list(wt)
        for p in rng.choice(len(wt), size=min(k, len(wt)), replace=False):
            chars[p] = AMINO_ACIDS[rng.integers(0, 20)]
        out.append("".join(chars))
    return out


def build_cohort(name, label, seqs, canvas, judge, judge_tok, device, wt,
                 train_seqs=None):
    """Score + annotate a set of sequences into a cohort dict."""
    cohort = {
        "name": name, "label": label, "seqs": seqs, "canvas": canvas,
        "fitness": score_fitness(seqs, judge, judge_tok, device),
        "mutations": mutation_counts(seqs, wt),
        "novelty": novelty_to_training(seqs, train_seqs) if train_seqs is not None else None,
    }
    return cohort


def run_preset(preset, generator, judge, judge_tok, device, wt, train_full_seqs,
               train_viable, random_cohort, n, steps, seed):
    """Generate cohorts under one sampler preset and emit all figures."""
    out_dir = FIGURES / preset
    config = Config()
    config.sampler.num_steps = steps
    for key, value in PRESETS[preset].items():
        setattr(config.sampler, key, value)

    tok = tokenizer()

    def gen(fitness, w):
        config.sampler.guidance_scale = w
        ids = generate(generator, n, fitness=fitness, config=config, seed=seed,
                       return_ids=True).cpu().numpy()
        seqs = [tok.decode(row) for row in ids]
        return ids, seqs

    # --- generate the w-sweep (target=6); w=CONDITIONED_W doubles as "conditioned" ---
    w_cohorts = {}
    for w in GUIDANCE_SCALES:
        ids, seqs = gen(TARGET_FITNESS, w)
        w_cohorts[w] = build_cohort(
            f"w{w}", f"w={w}", seqs, ids, judge, judge_tok, device, wt, train_full_seqs)
        print(f"  [{preset}] generated w={w}  viable={(w_cohorts[w]['fitness'] > viable_threshold()).mean():.3f}")

    # --- unconditional ---
    ids, seqs = gen(None, 1.0)
    uncond = build_cohort("unconditional", "unconditional", seqs, ids,
                          judge, judge_tok, device, wt, train_full_seqs)
    print(f"  [{preset}] generated unconditional  viable={(uncond['fitness'] > viable_threshold()).mean():.3f}")

    # conditioned cohorts at the requested guidance scales (reused from the w-sweep).
    # The first (w=2) is the canonical "conditioned" used for single-cohort figures.
    conditioned_cohorts = {}
    for w in CONDITIONED_WS:
        c = dict(w_cohorts[w])
        c["name"] = f"conditioned_w{w}"
        c["label"] = f"conditioned (w={w}, target={int(TARGET_FITNESS)})"
        conditioned_cohorts[c["name"]] = c
    primary = conditioned_cohorts[f"conditioned_w{CONDITIONED_WS[0]}"]

    # cohort groupings (both conditioned w's shown alongside baselines)
    viability_cohorts = {**conditioned_cohorts, "unconditional": uncond,
                         "random": random_cohort, "training": train_viable}
    scatter_cohorts = {**conditioned_cohorts, "unconditional": uncond,
                       "random": random_cohort}  # training excluded: novelty is 0 by definition

    # --- figures ---
    viability.plot(viability_cohorts, out_dir)
    diversity.plot_histograms(primary, out_dir)
    _, div_by_cohort = diversity.plot_by_cohort(viability_cohorts, out_dir, seed=seed)
    scatter.plot_by_w(w_cohorts, out_dir)
    scatter.plot_by_cohort(scatter_cohorts, out_dir)
    _, entropy_r = distribution.plot_entropy(primary["canvas"], train_viable["canvas"], out_dir)
    _, mean_jsd = distribution.plot_jsd(primary["canvas"], train_viable["canvas"], out_dir)
    distribution.plot_tsne(primary["canvas"], train_viable["canvas"], out_dir, seed=seed)

    # --- report ---
    print(f"\n  ===== {preset.upper()} preset summary =====")
    print(f"  viable fraction:  " + "  ".join(
        f"{c['label'].split('(')[0].strip()}={(np.asarray(c['fitness']) > viable_threshold()).mean():.3f}"
        for c in viability_cohorts.values()))
    print(f"  internal diversity (mean pairwise edit): " +
          "  ".join(f"{k}={v:.2f}" for k, v in div_by_cohort.items()))
    print(f"  conditioned (w={CONDITIONED_WS[0]}) novelty: mean min-dist to train = {np.mean(primary['novelty']):.2f}, "
          f"frac novel = {(np.asarray(primary['novelty']) > 0).mean():.3f}")
    print(f"  distribution match (conditioned vs viable training): "
          f"entropy Pearson r = {entropy_r:.3f}, mean per-anchor AA JSD = {mean_jsd:.3f}")
    print(f"  -> figures in {out_dir}\n")


def main():
    p = argparse.ArgumentParser(description="Full AAV diffusion eval suite (diverse + precise).")
    p.add_argument("-n", "--num", type=int, default=1500, help="sequences per cohort")
    p.add_argument("--steps", type=int, default=256, help="reverse sampling steps")
    p.add_argument("--preset", choices=list(PRESETS), default=None, help="run only one preset")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny fast correctness check")
    args = p.parse_args()

    n, steps = args.num, args.steps
    if args.smoke:
        n, steps = 24, 12

    device = get_device(args.device)
    wt = wild_type_sequence()
    print(f"device={device} | n={n} steps={steps} | w-sweep={GUIDANCE_SCALES} | "
          f"target={TARGET_FITNESS} | threshold={viable_threshold():.2f}")
    print(f"wild type: {wt}")

    t0 = time.time()
    generator = load_generator(device, Config())
    judge, judge_tok = load_judge(device)

    # shared across presets: training reference (full for novelty, viable for distribution),
    # and the random baseline (same seed -> identical in both folders)
    _, train_full_seqs = load_training(viable_only=False, n=20000, seed=args.seed)
    viable_canvas, viable_seqs = load_training(viable_only=True, n=n, seed=args.seed)
    train_viable = build_cohort("training", "training (viable)", viable_seqs, viable_canvas,
                                judge, judge_tok, device, wt, train_seqs=None)
    random_seqs = random_variants(n, wt, seed=args.seed)
    random_cohort = build_cohort("random", "random", random_seqs, None,
                                 judge, judge_tok, device, wt, train_full_seqs)
    print(f"models + references loaded ({time.time() - t0:.1f}s)\n")

    presets = [args.preset] if args.preset else list(PRESETS)
    for preset in presets:
        run_preset(preset, generator, judge, judge_tok, device, wt, train_full_seqs,
                   train_viable, random_cohort, n, steps, args.seed)

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
