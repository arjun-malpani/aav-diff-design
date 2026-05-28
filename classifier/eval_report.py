"""Load the best scheme-B checkpoint, run a full eval on predictor_test, and
break the test set down by partition so we can see where the model wins and
where it doesn't. Prints a small report and (optionally) saves a scatter plot.

Usage:
    uv run python classifier/eval_report.py \
        --scheme b --weights classifier/weights/esm35m_b.pt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from model import EVAL_SPLIT, MODEL_NAME, build_model, get_dataloader
from train import get_device, pearson, spearman

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "bryant" / "allseqs_20191230.csv"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="b", choices=["a", "b", "c"])
    p.add_argument("--weights", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--scatter", default=None,
                   help="optional path to save scatter plot (e.g. eval_scatter.png)")
    return p.parse_args()


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    use_amp = device.type == "cuda"
    preds, targets = [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(**batch)
        preds.append(out.logits.float().squeeze(-1).cpu())
        targets.append(batch["labels"].cpu())
    return torch.cat(preds), torch.cat(targets)


def summarize(preds, targets, label):
    mse = ((preds - targets) ** 2).mean().item()
    return {
        "split": label,
        "n": int(preds.numel()),
        "mse": mse,
        "rmse": mse ** 0.5,
        "pearson": pearson(preds, targets),
        "spearman": spearman(preds, targets),
    }


def main():
    args = parse()
    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    eval_split = EVAL_SPLIT[args.scheme.lower()]
    loader = get_dataloader(args.scheme, eval_split, tokenizer, args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = build_model(freeze_backbone=False).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state)

    preds, targets = predict_all(model, loader, device)

    rows = [summarize(preds, targets, "overall")]

    # Per-partition breakdown. We need to rejoin partition labels — load them
    # by mapping the sequences in the test split back to the raw CSV.
    seqs = torch.load(ROOT / "data" / "processed" / "bryant" / f"scheme_{args.scheme.lower()}"
                      / f"{eval_split}_seq.pt", weights_only=False)
    raw = pd.read_csv(RAW, usecols=["sequence", "partition"]).drop_duplicates("sequence")
    raw["sequence"] = raw["sequence"].str.upper()
    seq_to_part = dict(zip(raw["sequence"], raw["partition"]))
    partitions = np.array([seq_to_part.get(s, "unknown") for s in seqs])

    for part in sorted(set(partitions)):
        mask = partitions == part
        if mask.sum() < 50:
            continue
        rows.append(summarize(preds[mask], targets[mask], part))

    df = pd.DataFrame(rows)
    df["pearson"] = df["pearson"].map(lambda v: f"{v:.4f}")
    df["spearman"] = df["spearman"].map(lambda v: f"{v:.4f}")
    df["rmse"] = df["rmse"].map(lambda v: f"{v:.4f}")
    df["mse"] = df["mse"].map(lambda v: f"{v:.4f}")
    print(df.to_string(index=False))

    if args.scatter:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(targets.numpy(), preds.numpy(), s=4, alpha=0.3)
        lo, hi = float(min(targets.min(), preds.min())), float(max(targets.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set(xlabel="true viral_selection", ylabel="predicted",
               title=f"scheme {args.scheme.upper()} test — pearson={pearson(preds, targets):.3f}")
        fig.tight_layout()
        fig.savefig(args.scatter, dpi=150)
        print(f"scatter -> {args.scatter}")


if __name__ == "__main__":
    main()
