"""The denoiser network (the "tower") for the AAV discrete-diffusion model.

A bidirectional Transformer encoder with x0-parameterization: it reads the
embedded, [mask]-corrupted canvas and predicts, for every position, a categorical
distribution over the 21 CLEAN tokens (20 amino acids + [gap]; never [mask]).

Conditioning is the whole point of this model -- it is a classifier-free-guided
conditional generator. Two scalars steer it: the diffusion TIMESTEP t (how noisy
the input is) and a target FITNESS y. Both are injected into every block via
AdaLN-Zero (Peebles & Xie 2023): the conditioning sets each LayerNorm's scale and
shift and gates each sublayer, with the producing MLP zero-initialized so every
block starts as the identity and a deep stack trains stably from scratch.

Pipeline:  x_t -> SequenceEmbedding -> [AdaLNZeroBlock] x depth -> FinalLayer -> logits
            (t, y) -> ConditioningEmbedding -> c  ----feeds AdaLN in every block----^

------------------------------------------------------------------------------
A NOTE ON FOURIER FEATURES (read this if the conditioning math is unfamiliar)
------------------------------------------------------------------------------
A raw scalar fed straight into an MLP is a weak signal: the first layer can only
form a LINEAR function of it, and neural nets are bad at learning sharp/high-
frequency dependence on a single number ("spectral bias"). So before the MLP we
expand each scalar x into many sines and cosines at a range of frequencies:

    fourier(x) = [sin(2*pi*f_1*x), cos(2*pi*f_1*x), sin(2*pi*f_2*x), cos(2*pi*f_2*x), ...]

  - LOW frequencies change slowly across the input range  -> encode coarse magnitude
  - HIGH frequencies change quickly                        -> separate nearby values
    (so the MLP can tell t=0.50 from t=0.51, or fitness 1.0 from 1.1)

Because the useful frequency band depends on the input's RANGE, timestep and
fitness use different bands (see the constants below). This is the standard
DDPM/DiT timestep embedding, generalized to also encode the fitness scalar.
"""
import math

import torch
import torch.nn as nn

from config import Config
from embed_blosum import SequenceEmbedding
from tokenizer import AAVTokenizer

# --- Fourier frequency bands, in cycles per unit of input ---
# Timestep t lives in [0, 1]: from 1 cycle across the whole schedule (coarse "how
# far along are we") up to 1000 cycles (separates very close timesteps at sampling).
TIME_MIN_FREQ, TIME_MAX_FREQ = 1.0, 1000.0
# Standardized fitness lives in ~[-4, 4]: from a fraction of a cycle across that
# range (coarse magnitude) up to a few cycles per unit (separates nearby targets).
FITNESS_MIN_FREQ, FITNESS_MAX_FREQ = 0.05, 5.0


def modulate(x, scale, shift):
    """Apply AdaLN scale/shift. x: (B, L, H); scale/shift: (B, H) broadcast over L.

    Uses the (1 + scale) convention so that a zero-initialized scale leaves the
    normalized activations unchanged (identity at the start of training).
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FourierFeatures(nn.Module):
    """Lift one scalar into 2*num_frequencies sinusoidal features (see module docstring).

    Frequencies are geometrically spaced in [min_freq, max_freq] and are FIXED
    (a buffer, not learned). forward maps (B,) -> (B, 2*num_frequencies).
    """

    def __init__(self, num_frequencies: int, min_freq: float, max_freq: float):
        super().__init__()
        freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), num_frequencies)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        angles = 2 * math.pi * x[:, None] * self.freqs[None, :]   # (B, num_frequencies)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ConditioningEmbedding(nn.Module):
    """Builds the conditioning vector c from the timestep t and the fitness y.

    Each scalar -> Fourier features -> a 2-layer MLP -> a hidden-dim vector; the two
    vectors are SUMMED into c (DiT convention). c drives AdaLN in every block.

    Classifier-free guidance: with probability cfg_dropout_prob during training (or
    when drop_fitness=True at sampling), the fitness vector is replaced by a single
    LEARNED "null" embedding, so one network also learns an unconditional denoiser.
    Encoding "no fitness" as a learned vector (not a scalar value) keeps it distinct
    from every real fitness, including 0. Timestep is NEVER dropped -- the model
    always needs to know the noise level.
    """

    def __init__(self, hidden: int, num_frequencies: int, cfg_dropout_prob: float):
        super().__init__()
        self.cfg_dropout_prob = cfg_dropout_prob
        fourier_dim = 2 * num_frequencies

        self.time_fourier = FourierFeatures(num_frequencies, TIME_MIN_FREQ, TIME_MAX_FREQ)
        self.fitness_fourier = FourierFeatures(num_frequencies, FITNESS_MIN_FREQ, FITNESS_MAX_FREQ)
        self.time_mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.fitness_mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.null_fitness = nn.Parameter(torch.randn(hidden) * 0.02)  # learned "no fitness" vector

    def _dropout_mask(self, batch, device, drop_fitness):
        """Per-sample bool: True where the null fitness embedding should be used."""
        if drop_fitness is True:
            return torch.ones(batch, dtype=torch.bool, device=device)
        if drop_fitness is False:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        # drop_fitness is None: random dropout during training, nothing at eval
        if self.training and self.cfg_dropout_prob > 0:
            return torch.rand(batch, device=device) < self.cfg_dropout_prob
        return torch.zeros(batch, dtype=torch.bool, device=device)

    def forward(self, t, fitness, drop_fitness=None):
        time_emb = self.time_mlp(self.time_fourier(t))                  # (B, H)
        fitness_emb = self.fitness_mlp(self.fitness_fourier(fitness))   # (B, H)
        drop = self._dropout_mask(t.shape[0], t.device, drop_fitness)
        fitness_emb = torch.where(drop[:, None], self.null_fitness[None, :], fitness_emb)
        return time_emb + fitness_emb


class AdaLNZeroBlock(nn.Module):
    """One bidirectional Transformer block with AdaLN-Zero conditioning (DiT-style).

    A pre-norm block is  x = x + Attn(LN(x));  x = x + FFN(LN(x)).  Here each LN has
    no learned affine; instead the conditioning c supplies per-sublayer scale, shift,
    and a GATE:
        x = x + gate_attn * Attn( modulate(LN(x), scale_attn, shift_attn) )
        x = x + gate_ffn  * FFN ( modulate(LN(x), scale_ffn,  shift_ffn ) )
    The MLP producing (scale, shift, gate) is ZERO-INITIALIZED, so at init every gate
    is 0 -> the block is the identity -> the deep stack trains stably and blocks
    "switch on" as useful conditioning is learned.
    """

    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, hidden))
        # one MLP -> 6 modulation vectors (scale/shift/gate for attn and ffn)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, c):
        scale1, shift1, gate1, scale2, shift2, gate2 = self.modulation(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), scale1, shift1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)         # full bidirectional attention
        x = x + gate1.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), scale2, shift2)
        x = x + gate2.unsqueeze(1) * self.ffn(h)
        return x


class FinalLayer(nn.Module):
    """Final AdaLN modulation + linear projection to per-position logits.

    The head is zero-initialized so the model starts by emitting uniform logits
    (a calm, high-entropy starting point, as in DiT).
    """

    def __init__(self, hidden: int, out_vocab: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.head = nn.Linear(hidden, out_vocab)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, c):
        scale, shift = self.modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), scale, shift)
        return self.head(x)


class DiffusionTransformer(nn.Module):
    """x0-predictor: corrupted canvas + (timestep, fitness) -> clean-token logits.

    forward(x_t, t, fitness, drop_fitness=None) -> (B, L, 21) logits over the clean
    tokens (ids 0..20). [mask] (id 21) is a corruption state, never a target, so the
    head simply does not have an output for it.
    """

    def __init__(self, config: Config):
        super().__init__()
        model_config, tokenizer_config = config.model, config.tokenizer
        hidden = model_config.hidden
        tokenizer = AAVTokenizer(tokenizer_config)

        self.embedding = SequenceEmbedding(model_config, tokenizer_config)
        self.conditioning = ConditioningEmbedding(
            hidden, model_config.fourier_num_frequencies, model_config.cfg_dropout_prob)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(hidden, model_config.heads, model_config.ffn, model_config.dropout)
            for _ in range(model_config.depth)
        ])
        self.final = FinalLayer(hidden, tokenizer.clean_vocab_size)  # 21 clean tokens

    def forward(self, x_t, t, fitness, drop_fitness=None):
        c = self.conditioning(t, fitness, drop_fitness)   # (B, hidden)
        x = self.embedding(x_t)                            # (B, L, hidden)
        for block in self.blocks:
            x = block(x, c)
        return self.final(x, c)                            # (B, L, 21)


if __name__ == "__main__":
    config = Config()
    model = DiffusionTransformer(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M")

    batch, canvas_len = 8, config.tokenizer.canvas_len
    mask_id = AAVTokenizer(config.tokenizer).mask_id
    x_t = torch.randint(0, mask_id + 1, (batch, canvas_len))   # ids may include [mask]
    t = torch.rand(batch)                                       # timesteps in [0, 1]
    fitness = torch.randn(batch)                                # standardized fitness

    logits = model(x_t, t, fitness)
    print("logits shape:", tuple(logits.shape), "(expected (8, 56, 21))")
    assert logits.shape == (batch, canvas_len, 21)

    # AdaLN-Zero starts as the identity with a zero-init head, so at init every block
    # is bypassed and logits are uniform (all ~0) REGARDLESS of conditioning. That is
    # correct DiT behavior: conditioning only takes effect once the gates are trained.
    print(f"init logit spread (max - min): {(logits.max() - logits.min()).item():.2e} (expect ~0)")

    # Train a few steps on a dummy target so the gates switch on; then conditioning
    # (cond vs null fitness) must produce DIFFERENT predictions -- required for CFG.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randint(0, 21, (batch, canvas_len))
    for _ in range(20):
        optimizer.zero_grad()
        step_logits = model(x_t, t, fitness)
        loss = nn.functional.cross_entropy(step_logits.reshape(-1, 21), target.reshape(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    cond = model(x_t, t, fitness, drop_fitness=False)
    null = model(x_t, t, fitness, drop_fitness=True)
    print(f"after 20 steps, cond vs null mean|delta|: {(cond - null).abs().mean():.4f} (expect > 0)")
    assert not torch.allclose(cond, null)
    print("tower smoke test passed")
