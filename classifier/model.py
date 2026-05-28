"""Data loading and model construction for AAV fitness regression.

- `get_dataloader(scheme, split, tokenizer, ...)` reads a processed split
  (raw sequence strings + viral_selection scores) and tokenizes on the fly.
- `build_model(freeze_backbone=...)` returns ESM-2 35M with its built-in
  regression head; freezing the backbone trains the head only.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForSequenceClassification

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"  # ESM-2, 35M params

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "bryant"

# The held-out split is named "test" in every scheme.
EVAL_SPLIT = {"a": "predictor_test", "b": "predictor_test", "c": "predictor_test"}


class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = list(seqs)
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def get_dataloader(scheme, split, tokenizer, batch_size=32, shuffle=False,
                   limit=None, seed=42, num_workers=0):
    """Build a DataLoader for one scheme/split. `limit` subsamples for quick runs."""
    d = PROCESSED / f"scheme_{scheme.lower()}"
    seqs = torch.load(d / f"{split}_seq.pt", weights_only=False)
    labels = torch.load(d / f"{split}_score.pt", weights_only=False).float()

    if limit is not None and limit < len(seqs):
        idx = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(seed))[:limit]
        seqs = [seqs[i] for i in idx.tolist()]
        labels = labels[idx]

    def collate(batch):
        s, y = zip(*batch)
        enc = tokenizer(list(s), padding=True, return_tensors="pt")
        enc["labels"] = torch.stack(y)
        return enc

    return DataLoader(SeqDataset(seqs, labels), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers, collate_fn=collate)


def build_model(freeze_backbone=True, model_name=MODEL_NAME):
    """ESM-2 with its built-in regression head (predicts viral_selection).

    freeze_backbone=True trains only the head; False fine-tunes everything.
    """
    model = EsmForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression")
    if freeze_backbone:
        for p in model.esm.parameters():
            p.requires_grad = False
    return model
