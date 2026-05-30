"""PyTorch dataset and dataloader for the AAV diffusion model.

Loads the pre-tokenized canvas tensors written by preprocess.py (raw sequences
-> fixed L=56 integer canvas) together with the continuous fitness score used
for conditioning. Pre-tokenizing once keeps training fast and side-steps the
case-loss in the classifier's processed *_seq.pt tensors.

Run preprocess.py first to create the tensors this module loads.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DIFFUSION_DATA = ROOT / "data" / "processed" / "bryant" / "diffusion"


class CanvasDataset(Dataset):
    """Pre-tokenized canvases + fitness scores for diffusion training."""

    def __init__(self, data_dir=DIFFUSION_DATA):
        data_dir = Path(data_dir)
        self.canvas = torch.load(data_dir / "canvas.pt", weights_only=False)
        self.score = torch.load(data_dir / "score.pt", weights_only=False).float()

    def __len__(self):
        return self.canvas.shape[0]

    def __getitem__(self, index):
        return self.canvas[index], self.score[index]


def get_dataloader(batch_size=128, shuffle=True, data_dir=DIFFUSION_DATA, num_workers=0):
    """Build a DataLoader over the pre-tokenized diffusion canvases.

    Each batch is (canvas, score):
        canvas: LongTensor  (batch_size, canvas_len)  token ids on the canvas
        score:  FloatTensor (batch_size,)             viral_selection fitness
    """
    dataset = CanvasDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
