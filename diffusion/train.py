"""Training loop for the AAV discrete-diffusion generator.

Mirrors Classifier/train.py's instrumentation -- TensorBoard logging, best-
checkpoint saving, and early stopping -- but the objective is the diffusion
masked cross-entropy (DiffusionModel.loss), and early stopping monitors held-out
VALIDATION LOSS (lower is better) rather than a correlation.

    # quick local sanity run (subsample + few epochs)
    from train import train
    train(epochs=2, train_limit=2048)

    # full run on a GPU, with tensorboard
    python train.py --epochs 100 --log-dir runs/ --out weights/diffusion.pt

Reload the trained network for sampling with:

    from config import Config
    from diffusion import DiffusionModel
    m = DiffusionModel(Config())
    m.load_state_dict(torch.load("weights/diffusion.pt"))

The eval split is carved out of the single diffusion pool at load time with a
fixed seed. Validation noising is seeded too (same timesteps/masks every epoch)
so the val-loss curve is comparable across epochs rather than jittering with the
random corruption; the training RNG stream is saved and restored around it.
"""
import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from tqdm.auto import tqdm
# SummaryWriter is imported lazily inside train() so logging is opt-in: training
# runs without tensorboard installed unless you pass log_dir.

from config import Config
from DataLoading import CanvasDataset
from diffusion import DiffusionModel

DEFAULT_OUT = Path(__file__).resolve().parent / "weights" / "diffusion.pt"
VAL_NOISE_SEED = 1234  # fixed corruption for a comparable val-loss curve


def get_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr, weight_decay):
    """AdamW with weight decay on matmul weights only.

    Biases and the affine-free LayerNorms have ndim < 2; embeddings (token +
    positional) and the learned null-fitness vector are excluded by name. Decaying
    those tends to hurt, so they go in a weight_decay=0 group.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        excluded_by_name = any(k in name for k in ("embedding", "position", "null"))
        if param.ndim < 2 or excluded_by_name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def build_scheduler(optimizer, total_steps, warmup_frac):
    """Linear warmup for warmup_frac of training, then linear decay to 0."""
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_scale(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Mean masked-CE over the val set, with corruption fixed by VAL_NOISE_SEED.

    The RNG state is saved and restored so seeding the validation noise does not
    perturb the training RNG stream.
    """
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    torch.manual_seed(VAL_NOISE_SEED)

    total_loss, total_seqs = 0.0, 0
    for canvas, fitness in loader:
        canvas = canvas.to(device, non_blocking=True)
        fitness = fitness.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model.loss(canvas, fitness)
        total_loss += loss.item() * canvas.shape[0]
        total_seqs += canvas.shape[0]

    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return total_loss / total_seqs


def train(out_path=DEFAULT_OUT, config=None, epochs=None, lr=None, batch_size=None,
          weight_decay=None, warmup_frac=None, grad_clip=None, val_frac=None,
          early_stop_patience=None, seed=None, device=None, num_workers=0,
          train_limit=None, log_dir=None, run_name=None):
    """Train the diffusion generator; save the best (lowest val-loss) checkpoint.

    Most hyperparameters default to config.train (TrainConfig). `train_limit` caps
    training rows for fast local runs. Returns (model, history).
    """
    config = config or Config()
    tc = config.train
    epochs = epochs if epochs is not None else tc.epochs
    lr = lr if lr is not None else tc.lr
    batch_size = batch_size if batch_size is not None else tc.batch_size
    weight_decay = weight_decay if weight_decay is not None else tc.weight_decay
    warmup_frac = warmup_frac if warmup_frac is not None else tc.warmup_frac
    grad_clip = grad_clip if grad_clip is not None else tc.grad_clip
    val_frac = val_frac if val_frac is not None else tc.val_frac
    if early_stop_patience is None:
        early_stop_patience = tc.early_stop_patience
    seed = seed if seed is not None else tc.seed

    set_seed(seed)
    device = get_device(device)
    use_amp = device.type == "cuda"
    if use_amp:
        torch.set_float32_matmul_precision("high")

    # Single diffusion pool -> seeded train/val split for val-loss early stopping.
    dataset = CanvasDataset()
    n_val = int(round(val_frac * len(dataset)))
    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val], generator=split_gen)
    if train_limit is not None and train_limit < len(train_set):
        keep = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed))[:train_limit]
        train_set = Subset(train_set, keep.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    model = DiffusionModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} | params={n_params/1e6:.1f}M | "
          f"train seqs={len(train_set):,} val seqs={len(val_set):,}")
    print(f"effective capacity: {n_params/max(1,len(train_set)):.1f} params/train-seq")

    optimizer = build_optimizer(model, lr, weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, warmup_frac)

    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        run_name = run_name or f"diffusion_{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = os.path.join(log_dir, run_name)
        writer = SummaryWriter(run_dir)
        writer.add_hparams(
            {"lr": lr, "batch_size": batch_size, "epochs": epochs,
             "weight_decay": weight_decay, "warmup_frac": warmup_frac,
             "grad_clip": grad_clip, "cfg_dropout_prob": config.model.cfg_dropout_prob}, {})
        print(f"tensorboard logging -> {run_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    epochs_since_best = 0

    history = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for canvas, fitness in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            canvas = canvas.to(device, non_blocking=True)
            fitness = fitness.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model.loss(canvas, fitness)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())
            if writer is not None:
                writer.add_scalar("train/loss_step", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate(model, val_loader, device, use_amp)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)

        if val_loss < best_val:
            best_val, best_epoch, epochs_since_best = val_loss, epoch, 0
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_path)
            print(f"  ↳ new best val_loss={val_loss:.4f}, saved -> {out_path}")
        else:
            epochs_since_best += 1
            if early_stop_patience and epochs_since_best >= early_stop_patience:
                print(f"early stop: no improvement in {early_stop_patience} epochs "
                      f"(best val_loss={best_val:.4f} @ epoch {best_epoch})")
                break

    print(f"best val_loss={best_val:.4f} @ epoch {best_epoch} -> {out_path}")
    if writer is not None:
        writer.close()
    return model, history


def _parse_args():
    p = argparse.ArgumentParser(description="Train the AAV discrete-diffusion generator.")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="checkpoint output path")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-frac", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="stop after N epochs without val-loss improvement (0 disables)")
    p.add_argument("--train-limit", type=int, default=None, help="cap training rows (quick runs)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-dir", default=None, help="enable tensorboard logging here (e.g. runs/)")
    p.add_argument("--run-name", default=None, help="subdir under --log-dir (default: timestamp)")
    return p.parse_args()


def main():
    args = _parse_args()
    train(
        out_path=args.out, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        weight_decay=args.weight_decay, warmup_frac=args.warmup_frac, grad_clip=args.grad_clip,
        val_frac=args.val_frac, early_stop_patience=args.early_stop_patience,
        train_limit=args.train_limit, num_workers=args.num_workers, device=args.device,
        seed=args.seed, log_dir=args.log_dir, run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
