"""Pre-tokenize the Bryant dataset into fixed-canvas tensors for the diffusion model.

Reads the raw, case-preserving CSV (insertions are lowercase there; the
classifier's processed *_seq.pt tensors were upper-cased and cannot represent
them), encodes every variant onto the L=56 interleaved canvas, and saves aligned
(canvas, score, viable) tensors so training just torch.loads them. Sequences the
current canvas cannot represent (e.g. a leading insertion before the first
anchor, ~0.11% of rows) are skipped and logged.

The diffusion model trains on the FULL cleaned distribution (viable + non-viable)
so the classifier-free-guidance unconditional path sees the whole landscape.

Run:  python preprocess.py     (from the Diffusion/ directory)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from preprocess_data import load_and_clean  # reuse the exact cleaning + dedup

from config import TokenizerConfig
from tokenizer import AAVTokenizer

OUT = ROOT / "data" / "processed" / "bryant" / "diffusion"

# Standardized fitness is clamped to +/- this many std devs before conditioning, so
# the rare tails (scores reach ~-11) don't dominate the Fourier features.
FITNESS_CLAMP = 4.0


def main() -> None:
    df = load_and_clean()  # case-preserved 'sequence', numeric 'viral_selection', deduped
    tokenizer = AAVTokenizer(TokenizerConfig())

    canvases, scores, viables = [], [], []
    skipped = 0
    for seq, score, viable in zip(df["sequence"], df["viral_selection"], df["is_viable"]):
        try:
            canvas_row = tokenizer.encode(seq)
        except ValueError:
            skipped += 1  # canvas can't represent this row (e.g. leading insertion)
            continue
        canvases.append(canvas_row)
        scores.append(float(score))
        viables.append(float(viable))

    canvas = torch.from_numpy(np.stack(canvases)).to(torch.long)
    score = torch.tensor(scores, dtype=torch.float32)
    viable = torch.tensor(viables, dtype=torch.float32)

    # Fitness standardization stats, computed on the diffusion training scores.
    # These are a FROZEN artifact: training and sampling must standardize fitness
    # with identical mean/std or classifier-free guidance drifts. The clamp is
    # applied to the standardized value at use time (see DataLoading.py), not here.
    fitness_stats = {"mean": float(score.mean()), "std": float(score.std()),
                     "clamp": FITNESS_CLAMP}

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(canvas, OUT / "canvas.pt")
    torch.save(score, OUT / "score.pt")
    torch.save(viable, OUT / "viable.pt")
    tokenizer.save(OUT / "tokenizer.json")
    (OUT / "fitness_stats.json").write_text(json.dumps(fitness_stats, indent=2))

    kept = canvas.shape[0]
    print(f"encoded {kept:,} sequences onto canvas {tuple(canvas.shape)} "
          f"(vocab={tokenizer.vocab_size}); skipped {skipped:,} unrepresentable")
    print(f"  score range [{score.min():.3f}, {score.max():.3f}], "
          f"viable_frac={viable.mean():.4f}")
    print(f"  fitness stats: mean={fitness_stats['mean']:.4f} std={fitness_stats['std']:.4f} "
          f"clamp=+/-{FITNESS_CLAMP} std")
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
