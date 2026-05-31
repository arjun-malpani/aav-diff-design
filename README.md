# aav-diff-design

Diffusion-based design of AAV capsid variants, using the Bryant et al. AAV2
dataset for training a generator and a viability/fitness predictor.

## Data

### Quick start

```bash
bash scripts/download_data.sh     # fetch raw CSVs into data/raw/bryant/
python scripts/preprocess_data.py # write tensors into data/processed/bryant/
```

Run preprocessing inside the project's Python environment (the `aav` conda env),
which has `pandas`, `numpy`, and `torch` installed.

### Layout

```
data/
├── raw/
│   └── bryant/
│       ├── allseqs_20191230.csv          # main sequence dataset
│       └── ValidationChipwithModelScores.csv
└── processed/
    └── bryant/
        ├── tokenizer.json                # amino-acid → integer ID mapping
        ├── scheme_a/  scheme_b/  scheme_c/
        └── (each scheme: *.pt tensors + stats.json)
```

Each `.pt` file holds a single tensor and loads directly:

```python
import torch
seq    = torch.load("data/processed/bryant/scheme_a/diffusion_seq.pt")    # [N, L] int64, 0 = padding
score  = torch.load("data/processed/bryant/scheme_a/diffusion_score.pt")  # [N]   float32 (viral_selection)
viable = torch.load("data/processed/bryant/scheme_a/diffusion_viable.pt") # [N]   float32 (0.0 / 1.0)
```

### Common preprocessing

Applied identically before every scheme is built:

- Drop non-experimental / control partitions: `stop`, `wild_type`,
  `previous_chip_viable`, `previous_chip_nonviable`.
- Drop rows whose `viral_selection` is `-inf` (sequences never observed in the
  virus pool — undefined enrichment).
- Drop duplicate sequences (the raw file repeats sequences across partitions).
- Tokenize: sequences use the 20 canonical amino acids. Bryant writes *mutated*
  positions in lowercase, so sequences are upper-cased before tokenizing
  (the mutation detail is preserved separately in the source columns). Tokens
  are integer IDs with `0` reserved for padding (21 tokens total), right-padded
  to the longest sequence in the data.
- `random_state=42` is used for every split, so runs are reproducible.

This leaves ~293k unique sequences, roughly balanced (~52% viable).

### The Bryant partitions

The Bryant dataset engineers a short variable region of the AAV2 capsid
protein. Each row is a candidate sequence and a label for whether it produced a
**viable** (assembly-competent, packaging) virus, measured by an enrichment
score (`viral_selection`). Sequences come from several *sources*, recorded in
the `partition` column:

| Partition | What it is |
|-----------|------------|
| `designed` | Sequences proposed by ML models as likely-viable candidates. |
| `rand` | Sequences with random mutations from wild type — a baseline. |
| `random_doubles` | Sequences carrying two random mutations. |
| `single` / `singles` | Single-mutant sequences (one change from wild type). |
| `*_seed` | Starting points for in-silico optimization "walks". |
| `*_walked` | Sequences produced by iteratively *walking* away from a seed to maximize a model's predicted viability — these probe sequence space far from wild type. |

The `cnn_` / `rnn_` / `lr_` prefixes on the walked/seed partitions name the model
that drove the walk (CNN, RNN, logistic regression); the middle of the name
(`standard`, `designed_plus_rand_train`, `rand_doubles_plus_single`) names the
data that model was trained on.

**Why partition matters for evaluation.** Different partitions are drawn from
different distributions. Random/single mutants sit close to wild type, while
`*_walked` sequences are deliberately pushed far away. A predictor that scores
well when train and test are mixed (i.i.d.) may fail when asked to judge
sequences from a source it never trained on. Splitting *by partition* therefore
tests a harder, more realistic kind of generalization.

### The three schemes

Two models are trained in this project: a **diffusion** generator and a
**predictor** that scores viability/fitness. The schemes differ in how the
cleaned data is divided between them.

| Scheme | Diffusion gets | Predictor split | Tests |
|--------|----------------|-----------------|-------|
| **A** | 80% (disjoint from predictor) | 15% train / 5% test | In-distribution predictor performance with **no leakage** into the generator's training data. |
| **B** | 100% of the data | 75% train / 25% test (random) | Maximum generative coverage; in-distribution predictor performance, accepting overlap between generator and predictor data. |
| **C** | reuses Scheme B's full set | train = non-`*_walked`, test = `*_walked` | **Out-of-distribution** generalization: can a predictor trained on near-wild-type sequences judge model-explored sequences far from it? |

Scheme A and B both measure how well the predictor does on held-out data drawn
from the same mixture it trained on. Scheme C is the stress test: train and test
come from different sources, so the gap between Scheme B and Scheme C accuracy
reveals how much the predictor relies on the test distribution looking like the
training one. (In the processed data this shift is visible even in the labels:
the non-walked train split is ~41% viable, the walked test split ~58%.)

Each scheme directory also contains a `stats.json` with per-split row counts and
viability fractions.

## Generating sequences (Diffusion)

An **absorbing-state discrete diffusion** model (MDLM-style, continuous time) that
generates VR-VIII sequences conditioned on a target fitness via **classifier-free
guidance**. Variants are encoded on a fixed **L=56 canvas** (28 substitution slots
interleaved with 28 insertion slots) so insertions stay aligned to the wild-type
scaffold. The network is a ~25M-param bidirectional Transformer with AdaLN-Zero
conditioning on the diffusion timestep and the fitness scalar. It is trained from
scratch on the full cleaned distribution (viable + non-viable, so the
classifier-free-guidance unconditional path sees the whole landscape); the ESM-2
predictor is used only as a held-out judge, never as a guidance signal.

All commands run from the `Diffusion/` directory, inside the `aav` conda env. Every
flag lives in `Diffusion/config.py`.

**Step 1 — Pre-tokenize the dataset** (raw CSV → canvas tensors). Run once. It reads
the *case-preserving* raw CSV (insertions are lowercase there, so this cannot use
the upper-cased `scheme_*` tensors), encodes every variant onto the canvas, and
writes `data/processed/bryant/diffusion/`:

```bash
cd Diffusion
python preprocess.py
```

This produces `canvas.pt` (`[N, 56]` int64 token ids), `score.pt`, `viable.pt`,
`tokenizer.json`, and `fitness_stats.json` — the frozen mean/std used to
standardize the conditioning scalar identically at train and sample time. Rows the
canvas cannot represent (≈0.1%, a leading insertion before the first anchor) are
skipped and logged.

**Step 2 — Train the generator.** Early-stops on held-out validation loss (masked
cross-entropy) and saves the best checkpoint. TensorBoard logging is opt-in via
`--log-dir`:

```bash
# quick local sanity check (subsample + a few epochs)
python train.py --train-limit 4096 --epochs 5

# full run with TensorBoard logging
python train.py --epochs 100 --log-dir runs/ --out weights/diffusion.pt
```

Common overrides (defaults in `config.py:TrainConfig`, optimizer recipe from
AAVDiff): `--lr`, `--batch-size`, `--weight-decay`, `--warmup-frac`, `--grad-clip`,
`--val-frac`, `--early-stop-patience`.

**Step 3 — Generate sequences** from a trained checkpoint:

```bash
# 20 sequences targeting high fitness, with guidance
python denoising.py -n 20 --fitness 5 --guidance 2 --decoding confidence

# unconditional baseline (omit --fitness)
python denoising.py -n 20
```

`--fitness` is a target in raw `viral_selection` units (standardized internally).
Sampler dials (defaults in `config.py:SamplerConfig`): `--steps` (reverse steps),
`--guidance` (CFG scale *w*; 1.0 = off), `--temperature` (*tau*), `--decoding`
(`random` = more diverse | `confidence` = more precise, MaskGIT-style).

**Component → config mapping** (`Diffusion/config.py`):

| Component | File | Config group |
|-----------|------|--------------|
| Vocab + L=56 canvas | `tokenizer.py` | `TokenizerConfig` |
| Transformer + conditioning | `network_modules.py` | `ModelConfig` |
| Noise schedule + corruption kernel | `schedule.py`, `diffusion.py` | `DiffusionConfig` |
| Training loop | `train.py` | `TrainConfig` |
| Reverse sampler | `denoising.py` | `SamplerConfig` |

**Unresolved decision points** (left as config knobs, not hardcoded):

1. **Corruption kernel** (`DiffusionConfig.kernel`) — defaults to `absorbing`
   (consistently strong on text/proteins). The contrary claim that `uniform` can be
   competitive at small vocab sizes is treated as an open ablation; only absorbing
   is wired up so far.
2. **Canvas size** (`TokenizerConfig.n_anchor`, `insertions_per_gap`) — defaults to
   28 / 1 (→ L=56), which covers 100% of the Bryant data (never >1 insertion per
   gap). Multi-residue insertions (e.g. AAV2.7m8's 10-mer) require raising
   `insertions_per_gap`.

## Model weights

Training (`classifier/train.py`) writes checkpoints to `classifier/weights/`.
These files are large (the ESM-2 35M checkpoint is ~130 MB, over GitHub's 100 MB
limit) and are **gitignored** — they are not stored in the repo. Regenerate them
by running training, or transfer them out-of-band.

The diffusion generator (`Diffusion/train.py`) writes its best checkpoint to
`Diffusion/weights/diffusion.pt` (~100 MB at the default ~25M params), also
gitignored. Regenerate by running training (Step 2 above).
