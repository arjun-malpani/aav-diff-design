"""Reverse-diffusion sampler: generate AAV canvas sequences from the trained model.

Absorbing-state denoising. We start from an all-[mask] canvas and walk continuous
time from t=1 down to t=0 over num_steps reverse steps. At each step the network
predicts a clean-token distribution for every position; we then UNMASK a subset of
the still-masked positions and commit sampled tokens there. Committed tokens are
frozen for the rest of the run (absorbing diffusion is monotone -- no re-masking).

There is NO DDPM-style noise re-injection: the only stochasticity is (a) which
masked positions get unmasked this step and (b) which token is drawn from the
predicted categorical. See diffusion.py for the matching forward process.

Three sampler dials (SamplerConfig):
  - guidance_scale w : classifier-free guidance in LOGIT space,
        l = l_uncond + w * (l_cond - l_uncond);  w=1 disables guidance.
  - temperature tau  : divides logits before softmax (tau<1 sharpens, >1 flattens).
  - decoding rule    : "random"     -> unmask via the schedule posterior (diverse),
                       "confidence" -> commit the most-confident positions first
                                       (MaskGIT-style; precise but mode-seeking).
  - commit           : "sample" (default, preserves diversity) | "argmax".

Fitness conditioning: pass a target in RAW viral_selection units; it is
standardized with the SAME frozen stats used in training (DataLoading.py) so the
target means the same thing at train and sample time. fitness=None generates
unconditionally (the CFG null path) -- useful for the eval baselines.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config
from DataLoading import load_fitness_stats, standardize_fitness
from diffusion import DiffusionModel
from tokenizer import AAVTokenizer

DEFAULT_CKPT = Path(__file__).resolve().parent / "weights" / "diffusion.pt"


def _predict_logits(model, x, t, fitness, guidance_scale, unconditional):
    """Clean-token logits (B, L, 21), with classifier-free guidance applied.

    unconditional=True runs only the null (fitness-dropped) path. Otherwise, when
    guidance_scale != 1 we combine the conditional and unconditional logits; at
    guidance_scale == 1 a single conditional pass suffices.
    """
    if unconditional:
        return model.network(x, t, fitness, drop_fitness=True)
    cond = model.network(x, t, fitness, drop_fitness=False)
    if guidance_scale == 1.0:
        return cond
    uncond = model.network(x, t, fitness, drop_fitness=True)
    return uncond + guidance_scale * (cond - uncond)


def _positions_to_commit(is_masked, confidence, t_cur, t_next, canvas_len, decoding):
    """Boolean (B, L): which still-masked positions to unmask on this step.

    random:     each masked position unmasks with the absorbing reverse-posterior
                probability (alpha_next - alpha_cur)/(1 - alpha_cur). For the linear
                schedule (alpha_bar(t)=1-t) that simplifies to (t_cur - t_next)/t_cur.
    confidence: keep exactly round(canvas_len * t_next) positions masked per sequence
                and commit the rest, choosing the highest-confidence ones first.
    Both reach zero masked at t_next=0, so the canvas is always fully denoised.
    """
    if decoding == "random":
        unmask_prob = ((t_cur - t_next) / t_cur.clamp_min(1e-8))
        drawn = torch.rand_like(is_masked, dtype=torch.float) < unmask_prob
        return is_masked & drawn

    if decoding == "confidence":
        target_masked = int(round(canvas_len * float(t_next)))
        n_masked = is_masked.sum(dim=1)                          # (B,)
        n_commit = (n_masked - target_masked).clamp_min(0)       # (B,)
        # rank positions by confidence within each row (0 = most confident); masked-out
        # positions get -inf so they always rank last and are never selected.
        conf = confidence.masked_fill(~is_masked, float("-inf"))
        order = conf.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(canvas_len, device=conf.device).expand_as(order))
        return is_masked & (ranks < n_commit.unsqueeze(1))

    raise ValueError(f"unknown decoding rule {decoding!r} (use 'random' or 'confidence')")


@torch.no_grad()
def generate(model, n, fitness=None, config=None, device=None, seed=None, return_ids=False):
    """Generate n sequences. Returns decoded strings (or canvas id tensor if return_ids).

    fitness: target in raw viral_selection units (standardized internally), or None
             for unconditional generation.
    """
    config = config or model.config
    sampler = config.sampler
    device = device or next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()

    tokenizer = AAVTokenizer(config.tokenizer)
    canvas_len = config.tokenizer.canvas_len
    mask_id = model.mask_id

    unconditional = fitness is None
    if unconditional:
        fitness_vec = torch.zeros(n, device=device)  # ignored: null path replaces it
    else:
        z = standardize_fitness(float(fitness), load_fitness_stats())
        fitness_vec = torch.full((n,), z, device=device)

    # All-[mask] start; never seed from WT.
    x = torch.full((n, canvas_len), mask_id, dtype=torch.long, device=device)
    times = torch.linspace(1.0, 0.0, sampler.num_steps + 1, device=device)

    for step in range(sampler.num_steps):
        t_cur, t_next = times[step], times[step + 1]
        is_masked = x == mask_id
        if not is_masked.any():
            break

        t_batch = torch.full((n,), t_cur, device=device)
        logits = _predict_logits(model, x, t_batch, fitness_vec, sampler.guidance_scale, unconditional)
        probs = F.softmax(logits / sampler.temperature, dim=-1)         # (n, L, 21)

        if sampler.commit == "argmax":
            candidate = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
        else:
            candidate = torch.distributions.Categorical(probs=probs).sample()
            confidence = probs.gather(-1, candidate.unsqueeze(-1)).squeeze(-1)

        commit = _positions_to_commit(is_masked, confidence, t_cur, t_next, canvas_len, sampler.decoding)
        x = torch.where(commit, candidate, x)

    if (x == mask_id).any():  # safety: t grid ends at 0 so this should not trigger
        raise RuntimeError("sampling left [mask] tokens; check the time grid / decoding rule")

    if return_ids:
        return x
    return [tokenizer.decode(row) for row in x.cpu().numpy()]


def load_model(ckpt_path=DEFAULT_CKPT, config=None, device=None):
    """Rebuild the model and load a trained checkpoint for sampling."""
    config = config or Config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionModel(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    return model


def _parse_args():
    p = argparse.ArgumentParser(description="Sample sequences from the AAV diffusion model.")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("-n", "--num", type=int, default=10, help="how many sequences to generate")
    p.add_argument("--fitness", type=float, default=None,
                   help="target fitness in raw viral_selection units (omit = unconditional)")
    p.add_argument("--steps", type=int, default=None, help="override num reverse steps")
    p.add_argument("--guidance", type=float, default=None, help="override guidance scale w")
    p.add_argument("--temperature", type=float, default=None, help="override temperature tau")
    p.add_argument("--decoding", choices=["random", "confidence"], default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    config = Config()
    if args.steps is not None:
        config.sampler.num_steps = args.steps
    if args.guidance is not None:
        config.sampler.guidance_scale = args.guidance
    if args.temperature is not None:
        config.sampler.temperature = args.temperature
    if args.decoding is not None:
        config.sampler.decoding = args.decoding

    model = load_model(args.ckpt, config)
    seqs = generate(model, args.num, fitness=args.fitness, config=config, seed=args.seed)
    target = "unconditional" if args.fitness is None else f"fitness={args.fitness}"
    print(f"# {len(seqs)} sequences ({target}, w={config.sampler.guidance_scale}, "
          f"tau={config.sampler.temperature}, {config.sampler.decoding}, {config.sampler.num_steps} steps)")
    for s in seqs:
        print(s)


if __name__ == "__main__":
    # Smoke test on an UNTRAINED model: checks the sampling machinery (shapes, full
    # denoising, both decoding rules, CFG paths, determinism). Sequences are garbage
    # until trained, but must be structurally valid (clean tokens only, no [mask]).
    torch.manual_seed(0)
    config = Config()
    config.sampler.num_steps = 16  # small for a fast smoke test
    model = DiffusionModel(config)
    tokenizer = AAVTokenizer(config.tokenizer)

    for decoding in ("random", "confidence"):
        config.sampler.decoding = decoding
        ids = generate(model, 32, fitness=2.0, config=config, seed=0, return_ids=True)
        assert ids.shape == (32, config.tokenizer.canvas_len)
        assert (ids != model.mask_id).all(), "found [mask] after sampling"
        assert ids.min() >= 0 and ids.max() <= tokenizer.gap_id, "non-clean token id"
        print(f"{decoding:>10}: ids {tuple(ids.shape)}, no [mask], ids in clean range OK")

    # unconditional path runs and decodes
    seqs = generate(model, 4, fitness=None, config=config, seed=1)
    assert all(set(s) <= set(config.tokenizer.amino_acids) for s in seqs), "decoded non-AA char"
    print(f"unconditional: generated {len(seqs)} sequences, all valid amino-acid strings")

    # determinism: same seed -> identical output
    a = generate(model, 8, fitness=2.0, config=config, seed=7, return_ids=True)
    b = generate(model, 8, fitness=2.0, config=config, seed=7, return_ids=True)
    assert torch.equal(a, b), "same seed produced different output"

    # guidance combines two paths without error; different w -> different logits path
    config.sampler.guidance_scale = 3.0
    _ = generate(model, 4, fitness=2.0, config=config, seed=0)
    print("sampler smoke test passed")
