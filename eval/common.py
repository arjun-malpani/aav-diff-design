"""Shared helpers for the diffusion eval suite.

Loads the trained models (diffusion generator + ESM-2 fitness judge), derives the
wild-type VR-VIII reference, and provides the distance / distribution / plotting
utilities used across the individual eval scripts.

The ESM-2 + MLP predictor is used ONLY as a held-out judge here -- never as a
guidance signal and never fine-tuned -- so the evaluation stays uncircular.

Distances are true Levenshtein edit distance (rapidfuzz, C-speed). Distribution
metrics (per-anchor entropy, per-anchor amino-acid JSD, t-SNE) operate on the 28
anchor slots of the canvas, which are already aligned to the wild-type scaffold,
so no MSA step is needed.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process as rf_process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "diffusion"))
sys.path.insert(0, str(ROOT / "classifier"))

from config import Config, TokenizerConfig      # noqa: E402  (diffusion/)
from diffusion import DiffusionModel            # noqa: E402  (diffusion/)
from tokenizer import AAVTokenizer              # noqa: E402  (diffusion/)
import model as classifier_model                # noqa: E402  (Classifier/model.py)

DIFFUSION_CKPT = ROOT / "diffusion" / "weights" / "diffusion.pt"
RAW_CSV = ROOT / "data" / "raw" / "bryant" / "allseqs_20191230.csv"
DIFFUSION_DATA = ROOT / "data" / "processed" / "bryant" / "diffusion"
FIGURES = Path(__file__).resolve().parent / "figures"

# Held-out judge: prefer the fully-trained scheme-B predictor (esm35m_b.pt); fall
# back to the scheme-A run (ghost_a.pt) if scheme B isn't present locally. Both are
# gitignored, so which one resolves depends on what's on the machine.
_CLASSIFIER_WEIGHTS = ROOT / "classifier" / "weights"
_JUDGE_PREFERENCE = ["esm35m_b.pt", "ghost_a.pt"]
CLASSIFIER_CKPT = next(
    (_CLASSIFIER_WEIGHTS / name for name in _JUDGE_PREFERENCE if (_CLASSIFIER_WEIGHTS / name).exists()),
    _CLASSIFIER_WEIGHTS / _JUDGE_PREFERENCE[0],  # default path even if absent, for a clear error
)

# Anchor slots hold a clean token: one of 20 amino acids OR [gap] (a deletion at
# that WT position). [mask] never survives sampling, so the anchor alphabet is 21.
# Real Bryant sequences have AAs at every anchor; the generator CAN emit a gap, and
# that's a meaningful event the composition/entropy statistics should count.
N_ANCHOR_TOKENS = 21


def get_device(prefer=None):
    """cuda > mps > cpu, matching the training scripts."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def tokenizer():
    return AAVTokenizer(TokenizerConfig())


# --- references --------------------------------------------------------------
@lru_cache(maxsize=1)
def viable_threshold():
    """Bryant's viability boundary in viral_selection units.

    is_viable isn't a clean cut on viral_selection (the classes overlap), so we
    recover the single threshold that best reproduces Bryant's label. It lands
    near -2.7 (~98% agreement, ~53% viable), NOT 0.0; the score distribution is
    centered well below 0 (mean ~ -2.3), so 0.0 would wrongly mark almost
    everything non-viable.
    """
    import pandas as pd
    df = pd.read_csv(RAW_CSV, usecols=["viral_selection", "is_viable"], low_memory=False)
    df["viral_selection"] = pd.to_numeric(df["viral_selection"], errors="coerce")
    df = df[~np.isneginf(df["viral_selection"])].dropna(subset=["viral_selection"])
    y = df["is_viable"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}).to_numpy()
    s = df["viral_selection"].to_numpy()
    candidates = np.quantile(s, np.linspace(0.01, 0.99, 197))
    accuracies = [((s > t) == y).mean() for t in candidates]
    return float(candidates[int(np.argmax(accuracies))])


@lru_cache(maxsize=1)
def wild_type_sequence():
    """The 28-residue AAV2 VR-VIII wild type, recovered from the raw CSV."""
    import pandas as pd
    df = pd.read_csv(RAW_CSV, usecols=["sequence", "mutation_sequence", "num_mutations"])
    wt = df[df["num_mutations"] == 0]
    if len(wt):
        return str(wt.iloc[0]["sequence"]).upper()
    mask_all_wt = df["mutation_sequence"].astype(str).str.fullmatch("_+")
    return str(df[mask_all_wt].iloc[0]["sequence"]).upper()


def load_training(viable_only=False, n=None, seed=0):
    """Load training canvases (and decoded strings) the diffusion model trained on.

    Returns (canvas_ids (N, L) int array, sequences list[str]). viable_only keeps
    only Bryant-viable sequences -- the right reference for the distribution-match
    metrics (the generator is asked to produce viable sequences). novelty uses the
    full set (memorization is about copying ANY training sequence).
    """
    canvas = torch.load(DIFFUSION_DATA / "canvas.pt", weights_only=False).numpy()
    if viable_only:
        viable = torch.load(DIFFUSION_DATA / "viable.pt", weights_only=False).numpy().astype(bool)
        canvas = canvas[viable]
    if n is not None and n < len(canvas):
        idx = np.random.default_rng(seed).choice(len(canvas), n, replace=False)
        canvas = canvas[idx]
    tok = tokenizer()
    seqs = [tok.decode(row) for row in canvas]
    return canvas, seqs


# --- edit distance -----------------------------------------------------------
def edit_distance(a, b):
    """True Levenshtein distance (insert/delete/substitute). Length-robust."""
    return Levenshtein.distance(a, b)


def mutation_counts(seqs, wt=None):
    """Edit distance from each sequence to wild type (per-sequence mutation count)."""
    wt = wt or wild_type_sequence()
    return np.array([Levenshtein.distance(s, wt) for s in seqs])


def novelty_to_training(seqs, train_seqs, max_train=10000, seed=0):
    """Per-sequence min edit distance to the training set (0 = exact memorization).

    Uses rapidfuzz.cdist (C, multithreaded) over a training sample for speed.
    """
    if len(train_seqs) > max_train:
        idx = np.random.default_rng(seed).choice(len(train_seqs), max_train, replace=False)
        train_seqs = [train_seqs[i] for i in idx]
    dist = rf_process.cdist(seqs, train_seqs, scorer=Levenshtein.distance, workers=-1)
    return dist.min(axis=1)


def mean_pairwise_edit_distance(seqs, max_n=600, seed=0):
    """Mean pairwise edit distance within a set (internal diversity).

    Subsamples to max_n sequences, then averages the full pairwise matrix's
    off-diagonal (rapidfuzz.cdist makes the n^2 matrix cheap at this size).
    """
    if len(seqs) < 2:
        return 0.0
    if len(seqs) > max_n:
        idx = np.random.default_rng(seed).choice(len(seqs), max_n, replace=False)
        seqs = [seqs[i] for i in idx]
    dist = rf_process.cdist(seqs, seqs, scorer=Levenshtein.distance, workers=-1)
    n = len(seqs)
    return float(dist.sum() / (n * (n - 1)))  # exclude the zero diagonal


# --- distribution metrics (on the 28 aligned anchor slots) -------------------
def anchor_ids(canvas):
    """Extract the 28 anchor-slot token ids from a (N, L) canvas array -> (N, 28)."""
    return canvas[:, tokenizer().anchor_slots]


def _anchor_frequencies(canvas):
    """Per-anchor token frequency table: (28, 21), rows sum to 1.

    Columns are the 20 amino acids plus [gap] (a deletion at that anchor).
    """
    anchors = anchor_ids(canvas)
    freqs = np.zeros((anchors.shape[1], N_ANCHOR_TOKENS))
    for pos in range(anchors.shape[1]):
        counts = np.bincount(anchors[:, pos], minlength=N_ANCHOR_TOKENS)[:N_ANCHOR_TOKENS]
        freqs[pos] = counts / max(1, counts.sum())
    return freqs


def per_anchor_entropy(canvas):
    """Shannon entropy (nats) at each of the 28 anchor positions -> (28,).

    Low = conserved position, high = variable. Captures which positions the model
    keeps fixed vs. lets vary -- the structural grammar of viable sequences.
    """
    freqs = _anchor_frequencies(canvas)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(freqs > 0, freqs * np.log(freqs), 0.0)
    return -terms.sum(axis=1)


def per_anchor_jsd(canvas_a, canvas_b):
    """Jensen-Shannon divergence (nats) between two sets' per-anchor AA frequencies.

    Returns (28,): 0 = identical composition at that position, ln2 = disjoint.
    """
    from scipy.spatial.distance import jensenshannon
    fa, fb = _anchor_frequencies(canvas_a), _anchor_frequencies(canvas_b)
    return np.array([jensenshannon(fa[p], fb[p], base=np.e) ** 2 for p in range(fa.shape[0])])


def tsne_2d(canvas_groups, seed=0, perplexity=30):
    """t-SNE of anchor one-hot encodings. canvas_groups: list of (N_i, L) arrays.

    Returns (coords (sum N_i, 2), labels (sum N_i,)) where labels index the group.
    One-hot of the 28 anchors (28*20=560 dims) feeds t-SNE; purely a visualization
    of distribution overlap -- inter-cluster distances are not meaningful.
    """
    from sklearn.manifold import TSNE
    onehots, labels = [], []
    for group_index, canvas in enumerate(canvas_groups):
        anchors = anchor_ids(canvas)
        oh = np.zeros((len(anchors), anchors.shape[1], N_ANCHOR_TOKENS), dtype=np.float32)
        for i in range(len(anchors)):
            oh[i, np.arange(anchors.shape[1]), anchors[i]] = 1.0
        onehots.append(oh.reshape(len(anchors), -1))
        labels.append(np.full(len(anchors), group_index))
    X = np.concatenate(onehots)
    labels = np.concatenate(labels)
    perplexity = min(perplexity, max(2, (len(X) - 1) // 3))  # t-SNE needs perplexity < n/3
    coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=seed).fit_transform(X)
    return coords, labels


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# --- model loading -----------------------------------------------------------
def load_generator(device, config=None):
    """Trained diffusion generator (DiffusionModel) in eval mode."""
    config = config or Config()
    gen = DiffusionModel(config).to(device)
    gen.load_state_dict(torch.load(DIFFUSION_CKPT, map_location=device, weights_only=False))
    gen.eval()
    return gen


def load_judge(device):
    """Trained ESM-2 + MLP fitness predictor (held-out judge) + its tokenizer."""
    from transformers import AutoTokenizer
    judge = classifier_model.build_model(freeze_backbone=False).to(device)
    judge.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False))
    judge.eval()
    tok = AutoTokenizer.from_pretrained(classifier_model.MODEL_NAME)
    return judge, tok


@torch.no_grad()
def score_fitness(seqs, judge, judge_tokenizer, device, batch_size=128):
    """Predicted viral_selection fitness for a list of sequences (numpy array)."""
    preds = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        enc = judge_tokenizer(list(batch), padding=True, return_tensors="pt").to(device)
        out = judge(**enc)
        preds.append(out.logits.float().squeeze(-1).cpu())
    return torch.cat(preds).numpy() if preds else np.array([])


# --- plotting ----------------------------------------------------------------
def setup_style():
    """Light, presentable matplotlib defaults (kept simple on purpose)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    })


def save_figure(fig, name, out_dir=FIGURES):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    return path
