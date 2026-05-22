#!/usr/bin/env python
"""Preprocess the Bryant AAV dataset into PyTorch tensors.

Reads Data/raw/bryant/allseqs_20191230.csv, applies common cleaning and
tokenization, then writes three train/eval schemes to Data/processed/bryant/.
Run from anywhere: paths are resolved relative to the repo root.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Data" / "raw" / "bryant" / "allseqs_20191230.csv"
OUT = ROOT / "Data" / "processed" / "bryant"

SEED = 42
DROP_PARTITIONS = ["stop", "wild_type", "previous_chip_viable", "previous_chip_nonviable"]

# The 20 canonical amino acids. Bryant writes mutated positions in lowercase,
# so we case-fold to upper before tokenizing; the mutation info lives separately
# in the mutation_sequence / num_mutations columns.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<pad>"
PAD_ID = 0


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df = df[~df["partition"].isin(DROP_PARTITIONS)].copy()
    df["viral_selection"] = pd.to_numeric(df["viral_selection"], errors="coerce")
    df = df[~np.isneginf(df["viral_selection"])]
    df = df.drop_duplicates("sequence")
    return df.reset_index(drop=True)


def build_tokenizer() -> dict:
    stoi = {PAD_TOKEN: PAD_ID}
    for i, aa in enumerate(AMINO_ACIDS, start=1):
        stoi[aa] = i
    return stoi


def encode(sequences, stoi: dict, max_len: int) -> torch.Tensor:
    arr = np.full((len(sequences), max_len), PAD_ID, dtype=np.int64)
    for i, seq in enumerate(sequences):
        for j, ch in enumerate(seq.upper()):
            arr[i, j] = stoi[ch]
    return torch.from_numpy(arr)


def split_indices(n: int, fractions, seed: int = SEED):
    """Reproducible random split. Last fraction absorbs the rounding remainder."""
    perm = np.random.default_rng(seed).permutation(n)
    chunks, start = [], 0
    for frac in fractions[:-1]:
        size = int(round(frac * n))
        chunks.append(perm[start:start + size])
        start += size
    chunks.append(perm[start:])
    return chunks


def save_split(out_dir: Path, sub: pd.DataFrame, prefix: str, stoi: dict,
               max_len: int, suffix: str = "") -> dict:
    seq = encode(sub["sequence"].tolist(), stoi, max_len)
    score = torch.tensor(sub["viral_selection"].to_numpy(dtype=np.float32))
    viable = torch.tensor(sub["is_viable"].to_numpy(dtype=np.float32))
    torch.save(seq, out_dir / f"{prefix}_seq{suffix}.pt")
    torch.save(score, out_dir / f"{prefix}_score{suffix}.pt")
    torch.save(viable, out_dir / f"{prefix}_viable{suffix}.pt")
    return {"rows": int(len(sub)), "viable_frac": round(float(sub["is_viable"].mean()), 4)}


def write_stats(out_dir: Path, scheme: str, splits: dict) -> None:
    with open(out_dir / "stats.json", "w") as f:
        json.dump({"scheme": scheme, "splits": splits}, f, indent=2)


def main() -> None:
    df = load_and_clean()
    n = len(df)
    max_len = int(df["sequence"].str.len().max())
    stoi = build_tokenizer()

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tokenizer.json", "w") as f:
        json.dump(
            {"stoi": stoi, "pad_token": PAD_TOKEN, "pad_id": PAD_ID,
             "max_len": max_len, "vocab_size": len(stoi)},
            f, indent=2,
        )

    print(f"Cleaned dataset: {n} sequences, max_len={max_len}, vocab={len(stoi)}")

    # Scheme A: 80 / 15 / 5 random split across all data.
    a_dir = OUT / "scheme_a"
    a_dir.mkdir(exist_ok=True)
    diff_idx, ptr_idx, pval_idx = split_indices(n, [0.80, 0.15, 0.05])
    stats_a = {
        "diffusion": save_split(a_dir, df.iloc[diff_idx], "diffusion", stoi, max_len),
        "predictor_train": save_split(a_dir, df.iloc[ptr_idx], "predictor_train", stoi, max_len),
        "predictor_val": save_split(a_dir, df.iloc[pval_idx], "predictor_val", stoi, max_len),
    }
    write_stats(a_dir, "A", stats_a)

    # Scheme B: diffusion sees everything; predictor gets a 75/25 random split.
    b_dir = OUT / "scheme_b"
    b_dir.mkdir(exist_ok=True)
    btr_idx, bte_idx = split_indices(n, [0.75, 0.25])
    stats_b = {
        "diffusion_full": save_split(b_dir, df, "diffusion", stoi, max_len, suffix="_full"),
        "predictor_train": save_split(b_dir, df.iloc[btr_idx], "predictor_train", stoi, max_len),
        "predictor_test": save_split(b_dir, df.iloc[bte_idx], "predictor_test", stoi, max_len),
    }
    write_stats(b_dir, "B", stats_b)

    # Scheme C: predictor split by partition (non-walked train, walked test).
    # Diffusion reuses Scheme B's full set, so no diffusion files here.
    c_dir = OUT / "scheme_c"
    c_dir.mkdir(exist_ok=True)
    is_walked = df["partition"].str.endswith("_walked")
    stats_c = {
        "predictor_train": save_split(c_dir, df[~is_walked], "predictor_train", stoi, max_len),
        "predictor_test": save_split(c_dir, df[is_walked], "predictor_test", stoi, max_len),
    }
    write_stats(c_dir, "C", stats_c)

    for scheme, stats in [("A", stats_a), ("B", stats_b), ("C", stats_c)]:
        print(f"\nScheme {scheme}:")
        for name, s in stats.items():
            print(f"  {name:<16} rows={s['rows']:>7,}  viable_frac={s['viable_frac']}")


if __name__ == "__main__":
    main()
