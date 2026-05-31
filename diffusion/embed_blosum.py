"""Model components for the AAV discrete-diffusion generator.

This step implements only the EMBEDDING STEM. The full model (planned next) is a
bidirectional Transformer encoder with x0-parameterization: it reads the embedded
(possibly [mask]-corrupted) canvas and predicts per-position categorical logits
over the 21 clean tokens, with timestep + continuous-fitness conditioning injected
via AdaLN-Zero. None of that is built yet.

SequenceEmbedding turns a canvas of token ids -> vectors:
    - learned token embedding over all 22 tokens (MUST include [mask], since
      corrupted inputs carry it),
    - learned ABSOLUTE positional embedding over the L canvas slots (each slot has
      a fixed structural identity: anchor i, or the insertion after anchor i),
    - optional BLOSUM62 geometric init of the 20 amino-acid rows (Henikoff 1992)
      so biochemically similar residues (e.g. I/L) start near each other; [gap]
      and [mask] rows have no biochemistry and stay randomly initialized.
"""
import numpy as np
import torch
import torch.nn as nn

from config import ModelConfig, TokenizerConfig

EMB_INIT_STD = 0.02  # shared init scale for token + positional embeddings

# BLOSUM62 substitution matrix (Henikoff & Henikoff 1992), standard 20-AA order.
# Higher score = the pair substitutes more often in real alignments = more similar.
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62 = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]


def blosum62_init(amino_acids: str, dim: int) -> np.ndarray:
    """Geometric embedding of the amino acids from BLOSUM62, shape (n_aa, dim).

    The matrix is pairwise similarity scores, not vectors. We factorize it so
    that dot products approximate those scores: eigendecompose S = V diag(L) V^T,
    keep the positive eigenvalues, and set vectors = V * sqrt(L). Biochemically
    similar residues then start with high dot product / cosine similarity.

    BLOSUM62 is not positive semi-definite (it has negative eigenvalues, which
    cannot be reproduced by real dot products), so only the positive spectrum is
    kept -- a standard PSD approximation. Output is normalized to unit std (so
    SequenceEmbedding can rescale it to match the other rows); only the first
    min(#positive_eigenvalues, dim) columns are populated, the rest are zero.
    """
    order_index = [_BLOSUM62_ORDER.index(aa) for aa in amino_acids]
    full = np.array(_BLOSUM62, dtype=np.float64)
    scores = full[np.ix_(order_index, order_index)]  # reorder to our AA ordering

    eigenvalues, eigenvectors = np.linalg.eigh(scores)  # ascending, symmetric input
    descending = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending]
    eigenvectors = eigenvectors[:, descending]

    n_positive = int((eigenvalues > 1e-8).sum())
    keep = min(n_positive, dim)
    vectors = eigenvectors[:, :keep] * np.sqrt(eigenvalues[:keep])  # (n_aa, keep)
    vectors = vectors / vectors.std()                                # unit std; cosine preserved

    out = np.zeros((len(amino_acids), dim), dtype=np.float64)
    out[:, :keep] = vectors
    return out


class SequenceEmbedding(nn.Module):
    """Token + absolute positional embedding for the fixed canvas.

    forward(token_ids) maps a (batch, canvas_len) LongTensor of ids to a
    (batch, canvas_len, hidden) FloatTensor.
    """

    def __init__(self, model_config: ModelConfig, tokenizer_config: TokenizerConfig):
        super().__init__()
        amino_acids = tokenizer_config.amino_acids
        vocab_size = len(amino_acids) + 2          # 20 AAs + [gap] + [mask]
        hidden = model_config.hidden

        self.token_embedding = nn.Embedding(vocab_size, hidden)
        self.position_embedding = nn.Parameter(
            torch.randn(1, tokenizer_config.canvas_len, hidden) * EMB_INIT_STD)
        self.dropout = nn.Dropout(model_config.dropout)

        nn.init.normal_(self.token_embedding.weight, std=EMB_INIT_STD)
        if model_config.blosum_init:
            # seed the 20 amino-acid rows; [gap]/[mask] have no biochemistry, leave them random
            vectors = blosum62_init(amino_acids, hidden) * EMB_INIT_STD
            with torch.no_grad():
                self.token_embedding.weight[:len(amino_acids)] = torch.as_tensor(
                    vectors, dtype=self.token_embedding.weight.dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids) + self.position_embedding
        return self.dropout(x)


if __name__ == "__main__":
    from config import Config

    config = Config()
    config.model.blosum_init = True  # exercise the BLOSUM path explicitly
    embedding = SequenceEmbedding(config.model, config.tokenizer)

    batch, canvas_len, vocab_size = 4, config.tokenizer.canvas_len, 22
    token_ids = torch.randint(0, vocab_size, (batch, canvas_len))
    out = embedding(token_ids)
    print("input ids shape: ", tuple(token_ids.shape))
    print("output shape:    ", tuple(out.shape), "(expected (4, 56, %d))" % config.model.hidden)
    assert out.shape == (batch, canvas_len, config.model.hidden)

    # BLOSUM sanity: a conservative pair (I/L) should start more similar than a
    # drastic pair (I/D) in cosine similarity of their token embeddings.
    weight = embedding.token_embedding.weight.detach()
    aa = config.tokenizer.amino_acids
    cos = nn.functional.cosine_similarity
    il = cos(weight[aa.index("I")], weight[aa.index("L")], dim=0).item()
    idd = cos(weight[aa.index("I")], weight[aa.index("D")], dim=0).item()
    print(f"cosine(I, L)={il:+.3f}  cosine(I, D)={idd:+.3f}  -> I/L closer: {il > idd}")
    assert il > idd

    print("embedding smoke test passed")
