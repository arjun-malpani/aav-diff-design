"""Configuration for the AAV discrete-diffusion generator.

A single place for every tunable flag, grouped as dataclasses with the project
defaults. Code reads settings from here instead of hardcoding them; an argparse
overlay (as in Classifier/train.py) can be added per stage for CLI runs.

Currently populated:
  - TokenizerConfig  (consumed by diffusion/tokenizer.py)
  - ModelConfig      (consumed by the embedding stem / transformer)
  - DiffusionConfig  (corruption kernel + noise schedule)
  - TrainConfig      (consumed by diffusion/train.py)
  - SamplerConfig    (consumed by diffusion/denoising.py)
"""
from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Vocabulary and fixed-canvas geometry.

    The canvas interleaves one substitution slot per WT position with
    `insertions_per_gap` insertion slot(s) after it, giving
    `L = n_anchor * (1 + insertions_per_gap)` slots total.

    DECISION POINT (canvas size): default n_anchor=28, insertions_per_gap=1
    (-> L=56) matches the Bryant VR-VIII design space (verified: every Bryant
    sequence has exactly 28 anchor residues and never >1 insertion per gap).
    Designing multi-residue insertions (e.g. AAV2.7m8's 10-mer) requires raising
    `insertions_per_gap`; this is a deliberate ablation knob, not a fixed value.
    """
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical AAs (matches scripts/preprocess_data.py)
    n_anchor: int = 28          # WT scaffold positions (AAV2 VP1 561-588, VR-VIII)
    insertions_per_gap: int = 1  # insertion slots after each anchor; 1 -> L=56
    gap_token: str = "[gap]"     # clean, semantic empty-slot / deletion token
    mask_token: str = "[mask]"   # corruption-only absorbing state (never a prediction target)

    @property
    def canvas_len(self) -> int:
        return self.n_anchor * (1 + self.insertions_per_gap)


@dataclass
class ModelConfig:
    """Bidirectional Transformer encoder + embedding stem (x0-parameterization).

    These dims target ~24M parameters (printed at model init). That is well under
    AAVDiff's ~65M (12 layers / 512 hidden / 4096 ffn) but larger than a minimal
    from-scratch baseline -- so the from-scratch regularization (weight decay,
    dropout, early stopping, optional augmentation) matters at this size: watch the
    train/val gap for memorization. All dims are knobs; scale down if it overfits.
    """
    hidden: int = 384
    depth: int = 9
    heads: int = 12          # head dim = hidden / heads = 32
    ffn: int = 1536          # 4 x hidden
    dropout: float = 0.1
    blosum_init: bool = True  # seed AA token-embedding geometry from BLOSUM62 (Henikoff 1992)

    # --- conditioning (timestep + fitness, injected via AdaLN-Zero) ---
    fourier_num_frequencies: int = 128  # sinusoids per scalar; conditioning sees 2*128 features
    cfg_dropout_prob: float = 0.15      # CFG: fraction of training steps with fitness -> null embedding


@dataclass
class DiffusionConfig:
    """Forward corruption process: noise schedule + corruption kernel.

    DECISION POINT (corruption kernel): `absorbing` (the default) corrupts tokens
    to a dedicated [mask] state; `uniform` would instead corrupt to a random
    amino acid. Absorbing consistently wins among discrete methods on text and
    proteins (Sahoo 2024; Lou 2024; Austin 2021; Alamdari 2023). A contrary claim
    that uniform can be competitive at small vocab sizes is treated as an
    UNRESOLVED ABLATION, not a settled fact. Only absorbing is wired up so far.
    """
    schedule: str = "linear"     # see diffusion/schedule.py (only 'linear' implemented)
    kernel: str = "absorbing"    # 'absorbing' (default) | 'uniform' (not yet implemented)


@dataclass
class TrainConfig:
    """Training hyperparameters.

    The optimizer recipe (lr, warmup fraction, weight decay, linear LR decay)
    follows AAVDiff (Liu 2024, Supp. p.14). batch_size, grad_clip, val_frac, and
    the early-stopping settings are ours -- AAVDiff did not report them. `epochs`
    is a high cap; early stopping on held-out validation loss decides the real
    stopping point (from-scratch regularization, per the project spec).

    Weight decay is applied only to matmul weights, never to biases, the (affine-
    free) LayerNorms, or the embeddings -- see build_optimizer in train.py.
    """
    lr: float = 1e-4
    weight_decay: float = 0.04
    warmup_frac: float = 0.1       # fraction of total steps spent ramping LR up
    batch_size: int = 256
    grad_clip: float = 1.0         # max global grad norm (0 disables)
    epochs: int = 100              # cap; early stopping usually ends it sooner
    val_frac: float = 0.05         # held-out fraction for val-loss early stopping
    early_stop_patience: int = 5   # stop after N epochs without val-loss improvement (0 disables)
    seed: int = 42


@dataclass
class SamplerConfig:
    """Reverse-diffusion sampling dials (see diffusion/denoising.py).

    num_steps is the number of reverse steps N along continuous time [1 -> 0];
    because the model is continuous-time, N is chosen here at sampling time, not
    baked in at training. More steps -> higher quality, slower. For "confidence"
    decoding, N up to canvas_len (56) is efficient; beyond that some steps commit
    nothing.

    guidance_scale w applies classifier-free guidance in logit space
    (l = l_uncond + w*(l_cond - l_uncond)); w=1 disables guidance, higher w pushes
    harder toward the fitness target at some cost to diversity. temperature tau
    sharpens (<1) or flattens (>1) the per-position categorical before sampling.
    """
    num_steps: int = 128
    guidance_scale: float = 2.0      # w; 1.0 = no guidance
    temperature: float = 1.0         # tau
    decoding: str = "random"         # "random" (schedule posterior) | "confidence" (MaskGIT top-k)
    commit: str = "sample"           # "sample" (default, preserves diversity) | "argmax"


@dataclass
class Config:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
