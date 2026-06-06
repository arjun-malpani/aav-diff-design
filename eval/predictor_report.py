"""Predictor performance on the held-out test set, with the top-end saturation made
explicit.

This evaluates the ESM-2 + MLP judge ITSELF (not the generator): it scores the
scheme-B predictor_test split and plots predicted vs. true viral_selection. The
point is two-fold:
  1. report overall predictor quality (Pearson / Spearman / RMSE), and
  2. expose the high-fitness CEILING -- the judge saturates near +4.5 because the
     training labels are tail-sparse (only ~0.24% above +4.5) and MSE regresses the
     rare high targets toward the mean. Above that ceiling the predictor loses
     resolution, so any conditioning metric that needs to distinguish "very high"
     from "extremely high" fitness is limited by the JUDGE, not the generator.

Two panels:
  - parity scatter (predicted vs true) with the y=x line, density-shaded
  - Pearson + Spearman computed WITHIN true-fitness bins, showing correlation
    collapsing in the top bin (the saturation, quantified)

Standalone; writes only eval/figures/predictor/. Independent of the generator
(no sampling), so it can run on CPU without contending for the GPU.

    python predictor_report.py                 # scheme B test split
    python predictor_report.py --limit 4000    # subsample for a quick look
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import (FIGURES, get_device, pearson, save_figure, setup_style,
                    viable_threshold, ROOT)
import sys
sys.path.insert(0, str(ROOT / "Classifier"))
from model import EVAL_SPLIT, MODEL_NAME, build_model, get_dataloader  # noqa: E402

PREDICTOR_CKPT = ROOT / "Classifier" / "weights" / "esm35m_b.pt"


def spearman(x, y):
    """Spearman = Pearson on ranks."""
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return pearson(xr, yr)


@torch.no_grad()
def predict_test(scheme, limit, device, batch_size=256):
    """Score the predictor_test split; return (preds, trues) numpy arrays."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    loader = get_dataloader(scheme, EVAL_SPLIT[scheme.lower()], tok, batch_size,
                            shuffle=False, limit=limit)
    model = build_model(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(PREDICTOR_CKPT, map_location=device, weights_only=False))
    model.eval()
    use_amp = device.type == "cuda"
    preds, trues = [], []
    for batch in loader:
        # Keep labels on CPU: round-tripping them through MPS (and the model's loss
        # path) corrupts a handful of values to NaN/huge. The model only needs the
        # input ids/mask for logits, so we never pass labels to the device.
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(**batch)
        preds.append(out.logits.float().squeeze(-1).cpu())
        trues.append(labels)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    # safety net: drop any non-finite labels so correlations aren't poisoned
    finite = np.isfinite(preds) & np.isfinite(trues)
    if not finite.all():
        print(f"  dropped {(~finite).sum()} non-finite rows")
    return preds[finite], trues[finite]


def binned_correlation(preds, trues, edges):
    """Pearson + Spearman + n within each true-fitness bin defined by edges."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (trues >= lo) & (trues < hi)
        n = int(sel.sum())
        rows.append({
            "lo": lo, "hi": hi, "n": n,
            "pearson": pearson(preds[sel], trues[sel]) if n >= 10 else float("nan"),
            "spearman": spearman(preds[sel], trues[sel]) if n >= 10 else float("nan"),
        })
    return rows


def plot(preds, trues, out_dir):
    import matplotlib.pyplot as plt
    setup_style()
    overall_p = pearson(preds, trues)
    overall_s = spearman(preds, trues)
    ceiling = float(np.percentile(preds, 99.5))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- panel 1: parity scatter, density-shaded ---
    ax1.scatter(trues, preds, s=6, alpha=0.15, color="#4C72B0", edgecolors="none")
    lo = float(min(trues.min(), preds.min()))
    hi = float(max(trues.max(), preds.max()))
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (perfect)")
    ax1.axhline(ceiling, color="#C44E52", ls=":", lw=1.5,
                label=f"predicted ceiling (p99.5 = {ceiling:.2f})")
    ax1.axvline(viable_threshold(), color="gray", ls=":", lw=1, alpha=0.6)
    ax1.set(xlabel="true viral_selection", ylabel="predicted fitness (judge)",
            title=f"Predictor parity — Pearson {overall_p:.3f}, Spearman {overall_s:.3f}")
    ax1.legend(fontsize=9)

    # --- panel 2: correlation within true-fitness bins ---
    edges = np.array([-12, -6, -4, -2, 0, 2, 4, 6, 10])
    rows = binned_correlation(preds, trues, edges)
    centers = [(r["lo"] + r["hi"]) / 2 for r in rows]
    ax2.plot(centers, [r["pearson"] for r in rows], "-o", color="#4C72B0", label="Pearson")
    ax2.plot(centers, [r["spearman"] for r in rows], "-s", color="#DD8452", label="Spearman")
    ax2.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax2.set(xlabel="true fitness bin (center)", ylabel="correlation within bin",
            title="Correlation by fitness range (collapses at the top)", ylim=(-0.5, 1.0))
    # annotate the sparse top bins with their n
    for c, r in zip(centers, rows):
        if not np.isnan(r["pearson"]):
            ax2.annotate(f"n={r['n']}", (c, r["pearson"]), fontsize=7,
                         textcoords="offset points", xytext=(0, 6), ha="center")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return save_figure(fig, "predictor_parity.png", out_dir), rows, overall_p, overall_s


def main():
    p = argparse.ArgumentParser(description="Predictor test-set parity + top-end saturation.")
    p.add_argument("--scheme", default="b", choices=["a", "b", "c"])
    p.add_argument("--limit", type=int, default=None, help="subsample test rows for speed")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    print(f"device={device} | scheme={args.scheme} | predictor={PREDICTOR_CKPT.name}")
    t0 = time.time()
    preds, trues = predict_test(args.scheme, args.limit, device)
    print(f"scored {len(preds):,} test sequences ({time.time() - t0:.1f}s)")

    path, rows, op, os_ = plot(preds, trues, FIGURES / "predictor")
    print(f"\noverall: Pearson={op:.3f}  Spearman={os_:.3f}")
    print("correlation within true-fitness bins:")
    for r in rows:
        print(f"  [{r['lo']:+.0f},{r['hi']:+.0f})  n={r['n']:<6}  "
              f"pearson={r['pearson']:+.3f}  spearman={r['spearman']:+.3f}")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
