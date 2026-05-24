# aav-diff-design

Diffusion-based design of AAV capsid variants, using the Bryant et al. AAV2
dataset for training a generator and a viability/fitness predictor.

## Data

### Quick start

```bash
bash scripts/download_data.sh     # fetch raw CSVs into Data/raw/bryant/
python scripts/preprocess_data.py # write tensors into Data/processed/bryant/
```

Run preprocessing inside the project's Python environment (the `aav` conda env),
which has `pandas`, `numpy`, and `torch` installed.

### Layout

```
Data/
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
seq    = torch.load("Data/processed/bryant/scheme_a/diffusion_seq.pt")    # [N, L] int64, 0 = padding
score  = torch.load("Data/processed/bryant/scheme_a/diffusion_score.pt")  # [N]   float32 (viral_selection)
viable = torch.load("Data/processed/bryant/scheme_a/diffusion_viable.pt") # [N]   float32 (0.0 / 1.0)
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

## Model weights

Training (`Classifier/train.py`) writes checkpoints to `Classifier/weights/`.
These files are large (the ESM-2 35M checkpoint is ~130 MB, over GitHub's 100 MB
limit) and are **gitignored** — they are not stored in the repo. Regenerate them
by running training, or transfer them out-of-band.
