"""Training infrastructure for the ESM-2 AAV fitness regressor.

Call `train(...)` from a notebook to pick a data scheme, learning rate, and the
other hyperparameters, and to point it at a file where weights get saved. It is
built so you can do a quick local sanity run and then hand the same call to a
GPU machine:

    # fast local check on a Mac (subsample + few epochs, then plot)
    from train import train
    model, history = train(scheme="a", out_path="weights/esm35m_a.pt",
                           epochs=2, train_limit=256, eval_limit=256, plot=True)

    # full run on a friend's GPU (CLI)
    python train.py --scheme b --epochs 30 --lr 1e-4 --full-finetune \
        --out weights/esm35m_b.pt

`train` auto-selects the device (cuda > mps > cpu), returns the per-epoch
`history` for plotting, and saves the model weights to `out_path`. Reload with:

    from model import build_model
    m = build_model(); m.load_state_dict(torch.load("weights/esm35m_a.pt"))
"""
import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from model import EVAL_SPLIT, MODEL_NAME, build_model, get_dataloader


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


def _rank(t):
    order = torch.argsort(t)
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(t.numel(), dtype=torch.double)
    return ranks


def pearson(x, y):
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    denom = x.norm() * y.norm()
    return float(x @ y / denom) if denom > 0 else float("nan")


def spearman(x, y):
    return pearson(_rank(x), _rank(y))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    use_amp = device.type == "cuda"
    preds, targets = [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(**batch)
        preds.append(out.logits.float().squeeze(-1).cpu())
        targets.append(batch["labels"].cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse = ((preds - targets) ** 2).mean().item()
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "pearson": pearson(preds, targets),
        "spearman": spearman(preds, targets),
    }


def train(scheme, out_path, epochs=3, lr=1e-3, batch_size=32, freeze_backbone=True,
          weight_decay=0.0, train_limit=None, eval_limit=None,
          model_name=MODEL_NAME, device=None, num_workers=0, seed=42,
          eval_every=1, plot=False, log_dir=None, run_name=None,
          early_stop_patience=5, early_stop_metric="test_pearson"):
    """Fine-tune the ESM-2 regressor on one data scheme.

    Args:
        scheme: "a", "b", or "c" (which processed split to load).
        out_path: file to save the model weights (state_dict) to.
        epochs, lr, batch_size, weight_decay: training hyperparameters.
        freeze_backbone: True = train head only; False = full fine-tune.
        train_limit / eval_limit: cap rows for fast local runs (None = full data).
        device: force a device string; default auto-picks cuda > mps > cpu.
        eval_every: evaluate every N epochs.
        plot: if True, show loss/accuracy curves at the end.

    Returns:
        (model, history) where history is a list of per-epoch metric dicts.
    """
    set_seed(seed)
    device = get_device(device)
    use_amp = device.type == "cuda"
    if use_amp:
        torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = get_dataloader(scheme, "predictor_train", tokenizer, batch_size,
                                  shuffle=True, limit=train_limit, seed=seed,
                                  num_workers=num_workers)
    eval_loader = get_dataloader(scheme, EVAL_SPLIT[scheme.lower()], tokenizer, batch_size,
                                 shuffle=False, limit=eval_limit, seed=seed,
                                 num_workers=num_workers)

    model = build_model(freeze_backbone=freeze_backbone, model_name=model_name).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={device} | scheme={scheme} | train batches={len(train_loader)} "
          f"eval batches={len(eval_loader)}")
    print(f"freeze_backbone={freeze_backbone} | trainable params={n_train:,} / {n_total:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    writer = None
    if log_dir is not None:
        run_name = run_name or f"scheme_{scheme}_{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = os.path.join(log_dir, run_name)
        writer = SummaryWriter(run_dir)
        writer.add_hparams(
            {"scheme": scheme, "lr": lr, "batch_size": batch_size, "epochs": epochs,
             "freeze_backbone": freeze_backbone, "weight_decay": weight_decay},
            {},
        )
        print(f"tensorboard logging -> {run_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    metric_higher_is_better = early_stop_metric in {"test_pearson", "test_spearman"}
    best_metric = -float("inf") if metric_higher_is_better else float("inf")
    best_epoch = 0
    epochs_since_best = 0

    history = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_backbone:
            model.esm.eval()  # keep frozen backbone's dropout/stats fixed
        batch_losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(**batch)
            out.loss.backward()
            optimizer.step()
            loss = out.loss.item()
            batch_losses.append(loss)
            if writer is not None:
                writer.add_scalar("train/loss_step", loss, global_step)
            global_step += 1

        record = {"epoch": epoch, "train_loss": float(np.mean(batch_losses))}
        if epoch % eval_every == 0 or epoch == epochs:
            record.update({f"test_{k}": v for k, v in evaluate(model, eval_loader, device).items()})
        history.append(record)
        msg = f"epoch {epoch}: train_loss={record['train_loss']:.4f}"
        if "test_mse" in record:
            msg += (f" | test_rmse={record['test_rmse']:.4f}"
                    f" pearson={record['test_pearson']:.4f} spearman={record['test_spearman']:.4f}")
        print(msg)
        if writer is not None:
            writer.add_scalar("train/loss_epoch", record["train_loss"], epoch)
            for k in ("test_mse", "test_rmse", "test_pearson", "test_spearman"):
                if k in record:
                    writer.add_scalar(k.replace("_", "/", 1), record[k], epoch)

        cur = record.get(early_stop_metric)
        if cur is not None:
            improved = cur > best_metric if metric_higher_is_better else cur < best_metric
            if improved:
                best_metric, best_epoch, epochs_since_best = cur, epoch, 0
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_path)
                print(f"  ↳ new best {early_stop_metric}={cur:.4f}, saved -> {out_path}")
            else:
                epochs_since_best += 1
                if early_stop_patience and epochs_since_best >= early_stop_patience:
                    print(f"early stop: no improvement in {early_stop_patience} epochs "
                          f"(best {early_stop_metric}={best_metric:.4f} @ epoch {best_epoch})")
                    break

    print(f"best {early_stop_metric}={best_metric:.4f} @ epoch {best_epoch} -> {out_path}")

    if writer is not None:
        writer.close()

    if plot:
        plot_history(history)
    return model, history


def plot_history(history):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], "-o", label="train")
    test_pts = [(h["epoch"], h["test_mse"]) for h in history if "test_mse" in h]
    if test_pts:
        ax1.plot(*zip(*test_pts), "-o", label="test")
    ax1.set(xlabel="epoch", ylabel="MSE", title="loss")
    ax1.legend()

    pear = [(h["epoch"], h["test_pearson"]) for h in history if "test_pearson" in h]
    spear = [(h["epoch"], h["test_spearman"]) for h in history if "test_spearman" in h]
    if pear:
        ax2.plot(*zip(*pear), "-o", label="pearson")
    if spear:
        ax2.plot(*zip(*spear), "-o", label="spearman")
    ax2.set(xlabel="epoch", ylabel="correlation", title="test")
    ax2.legend()

    fig.tight_layout()
    plt.show()
    return fig


def _parse_args():
    p = argparse.ArgumentParser(description="Train the ESM-2 AAV fitness regressor.")
    p.add_argument("--scheme", required=True, choices=["a", "b", "c"])
    p.add_argument("--out", required=True, help="weights output path")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--full-finetune", action="store_true",
                   help="fine-tune the whole backbone (default: head only)")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", default=None,
                   help="enable tensorboard logging to this directory (e.g. runs/)")
    p.add_argument("--run-name", default=None,
                   help="subdirectory name under --log-dir (default: scheme + timestamp)")
    p.add_argument("--early-stop-patience", type=int, default=5,
                   help="stop if metric does not improve for N consecutive epochs (0 disables)")
    p.add_argument("--early-stop-metric", default="test_pearson",
                   choices=["test_pearson", "test_spearman", "test_mse", "test_rmse"])
    return p.parse_args()


def main():
    args = _parse_args()
    train(
        scheme=args.scheme, out_path=args.out, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, freeze_backbone=not args.full_finetune,
        weight_decay=args.weight_decay, train_limit=args.train_limit,
        eval_limit=args.eval_limit, num_workers=args.num_workers,
        device=args.device, seed=args.seed,
        log_dir=args.log_dir, run_name=args.run_name,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
    )


if __name__ == "__main__":
    main()
