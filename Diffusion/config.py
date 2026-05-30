"""Configuration for the AAV discrete-diffusion generator.

A single place for every tunable flag, grouped as dataclasses with the project
defaults. Code reads settings from here instead of hardcoding them; an argparse
overlay (as in Classifier/train.py) can be added per stage for CLI runs.

Currently populated:
  - TokenizerConfig  (consumed by Diffusion/tokenizer.py)
  - ModelConfig      (consumed by the embedding stem / transformer, next step)
  - DiffusionConfig  (corruption kernel + noise schedule)

Groups added in later stages: SamplerConfig (decoding rule / temperature /
guidance), TrainConfig.
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

    Kept modest on purpose: trained from scratch on a small, low-complexity
    corpus, so over-capacity memorizes (Bryant's own nets were 55K-129K params).
    Only `hidden` / `dropout` / `blosum_init` are consumed by the embedding stem;
    depth / heads / ffn are defined now for the transformer next step.
    """
    hidden: int = 256
    depth: int = 8
    heads: int = 8
    ffn: int = 1024
    dropout: float = 0.1
    blosum_init: bool = True  # seed AA token-embedding geometry from BLOSUM62 (Henikoff 1992)


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
    schedule: str = "linear"     # see Diffusion/schedule.py (only 'linear' implemented)
    kernel: str = "absorbing"    # 'absorbing' (default) | 'uniform' (not yet implemented)


@dataclass
class Config:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
