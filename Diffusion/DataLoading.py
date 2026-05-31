"""PyTorch dataset and dataloader for the AAV diffusion model.

Loads the pre-tokenized canvas tensors written by preprocess.py (raw sequences
-> fixed L=56 integer canvas) together with the continuous fitness score used
for conditioning. Pre-tokenizing once keeps training fast and side-steps the
case-loss in the classifier's processed *_seq.pt tensors.

Fitness is STANDARDIZED here (z = (score - mean) / std, clamped to +/- clamp std)
using the frozen stats from preprocess.py, so the network's Fourier conditioning
sees a well-scaled ~N(0, 1) scalar. The same standardize_fitness() is used at
sampling time to turn a requested target into the conditioning value.

Run preprocess.py first to create the tensors this module loads.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DIFFUSION_DATA = ROOT / "data" / "processed" / "bryant" / "diffusion"


def load_fitness_stats(data_dir=DIFFUSION_DATA):
    """Frozen {mean, std, clamp} for fitness standardization (from preprocess.py)."""
    return json.loads((Path(data_dir) / "fitness_stats.json").read_text())


def standardize_fitness(score, stats):
    """z = clamp((score - mean) / std, -clamp, +clamp). Accepts a float or tensor."""
    z = (score - stats["mean"]) / stats["std"]
    return z.clamp(-stats["clamp"], stats["clamp"]) if torch.is_tensor(z) \
        else max(-stats["clamp"], min(stats["clamp"], z))


class CanvasDataset(Dataset):
    """Pre-tokenized canvases + standardized fitness scores for diffusion training."""

    def __init__(self, data_dir=DIFFUSION_DATA, standardize=True):
        data_dir = Path(data_dir)
        self.canvas = torch.load(data_dir / "canvas.pt", weights_only=False)
        score = torch.load(data_dir / "score.pt", weights_only=False).float()
        self.score = standardize_fitness(score, load_fitness_stats(data_dir)) if standardize else score

    def __len__(self):
        return self.canvas.shape[0]

    def __getitem__(self, index):
        return self.canvas[index], self.score[index]


def get_dataloader(batch_size=128, shuffle=True, data_dir=DIFFUSION_DATA, num_workers=0):
    """Build a DataLoader over the pre-tokenized diffusion canvases.

    Each batch is (canvas, score):
        canvas: LongTensor  (batch_size, canvas_len)  token ids on the canvas
        score:  FloatTensor (batch_size,)             standardized fitness (~N(0, 1))
    """
    dataset = CanvasDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
