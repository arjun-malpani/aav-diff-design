"""Absorbing-state discrete diffusion model for AAV canvas sequences.

Ties the pieces together: the forward (noising) process, the denoiser network
(model.DiffusionTransformer), and the training loss.

Forward corruption q(x_t | x_0): each token independently jumps to the [mask]
absorbing state with probability mask_prob(t) = t (linear schedule), otherwise it
stays clean. Because [mask] is absorbing and the tokens are independent, we sample
any time t in one shot instead of stepping. [gap] is a clean token and gets
corrupted like any amino acid.

Training: corrupt x0 -> let the network predict clean-token logits conditioned on
(t, fitness) -> cross-entropy on the corrupted positions only. The reverse
(unmasking) sampler is a later step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from network_modules import DiffusionTransformer
from schedule import NoiseSchedule
from tokenizer import AAVTokenizer


class DiffusionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.diffusion.kernel != "absorbing":
            raise NotImplementedError(
                f"kernel {config.diffusion.kernel!r} not implemented; only 'absorbing' is available")
        self.config = config
        self.schedule = NoiseSchedule(config.diffusion.schedule)
        self.mask_id = AAVTokenizer(config.tokenizer).mask_id
        self.network = DiffusionTransformer(config)

    def add_noise(self, x0, t=None):
        """Corrupt clean ids x0 (B, L) toward [mask]; returns (x_t, t, is_masked).

        t (B,) are the per-sequence times in [0, 1], drawn uniformly if not given.
        is_masked (B, L) is a bool flag of which positions were absorbed -- the
        training loss is scored on exactly these positions.
        """
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        mask_prob = self.schedule.mask_prob(t).unsqueeze(1)            # (B, 1)
        is_masked = torch.rand_like(x0, dtype=torch.float) < mask_prob  # (B, L)
        x_t = x0.masked_fill(is_masked, self.mask_id)
        return x_t, t, is_masked

    def forward(self, x0, fitness, t=None, drop_fitness=None):
        """Noise x0, then predict clean-token logits. Returns (logits, is_masked).

        fitness (B,) is the STANDARDIZED conditioning scalar (see DataLoading.py).
        The same sampled t drives both the corruption level and the conditioning.
        """
        x_t, t, is_masked = self.add_noise(x0, t)
        logits = self.network(x_t, t, fitness, drop_fitness)
        return logits, is_masked

    def loss(self, x0, fitness, t=None):
        """Masked cross-entropy: mean over the corrupted ([mask]ed) positions.

        This is the simple BERT/MaskGIT-style objective. The principled MDLM
        continuous-time NELBO additionally weights each position by ~1/t; that
        weighting is left as a documented future knob rather than silently chosen.
        """
        logits, is_masked = self.forward(x0, fitness, t)             # (B, L, 21), (B, L)
        per_position = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x0.reshape(-1), reduction="none"
        ).reshape(x0.shape)
        scored = is_masked.float()
        return (per_position * scored).sum() / scored.sum().clamp_min(1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    config = Config()
    model = DiffusionModel(config)

    batch, canvas_len = 256, config.tokenizer.canvas_len
    x0 = torch.randint(0, model.mask_id, (batch, canvas_len))  # clean ids only (no [mask])
    fitness = torch.randn(batch)                               # standardized fitness

    # --- forward noising invariants ---
    *_, none_masked = model.add_noise(x0, torch.zeros(batch))
    *_, all_masked = model.add_noise(x0, torch.ones(batch))
    _, _, is_masked = model.add_noise(x0)
    print(f"t=0 / t=1 / random masked fraction: {none_masked.float().mean():.3f} / "
          f"{all_masked.float().mean():.3f} / {is_masked.float().mean():.3f} (expect 0 / 1 / ~0.5)")

    # --- forward through the network + loss ---
    logits, masked = model(x0, fitness)
    print("logits shape:", tuple(logits.shape), "(expected (256, 56, 21))")
    loss = model.loss(x0, fitness)
    loss.backward()
    print(f"loss: {loss.item():.4f} (expect ~ln(21)={torch.log(torch.tensor(21.0)):.4f} at init)")
    print("diffusion model smoke test passed")
