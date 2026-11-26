"""Training loop of the thermal network (weighted BCE with soft targets)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..lift.neural import ThermalNet, save_checkpoint
from .dataset import BACKGROUND, THERMAL, ThermalDataset


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-5
    hidden: tuple[int, ...] = (64, 64, 32)
    val_fraction: float = 0.15  # fraction of FLIGHTS held out for validation
    patience: int = 8  # early stopping (epochs without validation improvement)
    seed: int = 0


def validation_flights(flights, val_fraction: float, seed: int) -> np.ndarray:
    """Flight indices held out for validation, among the sorted indices `flights` of the usable flights."""
    rng = np.random.default_rng(seed)
    flights = np.unique(np.asarray(flights))
    return rng.choice(flights, max(1, int(round(val_fraction * len(flights)))), replace=False)


def thermal_auc(score: np.ndarray, kind: np.ndarray) -> float:
    """Probability that a thermal sample scores above a background sample (0.5 = chance, 1 = always): does the
    network tell where the thermals are from the ground the pilot did not climb over? nan without both kinds."""
    from scipy.stats import rankdata

    pos, neg = kind == THERMAL, kind == BACKGROUND
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(score[pos | neg])[pos[pos | neg]]  # Mann-Whitney U, ties count half
    return float((ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def split_by_flight(ds: ThermalDataset, val_fraction: float, seed: int):
    """Split samples so that a flight is entirely in train or in validation."""
    val = np.isin(ds.group, validation_flights(ds.group, val_fraction, seed))
    return ~val, val


def train(
    ds: ThermalDataset, cfg: TrainConfig | None = None, out_path: str | None = None, extra_meta: dict | None = None
):
    """Train a ThermalNet. Returns (model, history dict)."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    tr, va = split_by_flight(ds, cfg.val_fraction, cfg.seed)
    X, y, w = (torch.from_numpy(a) for a in (ds.X, ds.y, ds.w))

    model = ThermalNet(ds.X.shape[1], cfg.hidden)
    model.mean.copy_(X[tr].mean(dim=0))
    model.std.copy_(X[tr].std(dim=0).clamp_min(1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    def loss_fn(idx, logits=None):
        logits = model(X[idx]) if logits is None else logits
        per_sample = nn.functional.binary_cross_entropy_with_logits(logits, y[idx], reduction="none")
        return (per_sample * w[idx]).sum() / w[idx].sum()

    va_kind = ds.kind[va] if ds.kind is not None else np.full(int(va.sum()), -1)

    tr_idx, va_idx = torch.from_numpy(np.flatnonzero(tr)), torch.from_numpy(np.flatnonzero(va))
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best, best_state, stale = np.inf, None, 0
    print(
        f"[train] {len(tr_idx)} train / {len(va_idx)} validation samples, "
        f"{sum(p.numel() for p in model.parameters())} parameters"
    )
    for epoch in range(cfg.epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        total = 0.0
        for batch in perm.split(cfg.batch_size):
            opt.zero_grad()
            loss = loss_fn(batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(X[va_idx])
            val_loss = loss_fn(va_idx, logits).item()
        history["train_loss"].append(total / len(tr_idx))
        history["val_loss"].append(val_loss)
        history["val_auc"].append(thermal_auc(logits.numpy(), va_kind))
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(
                f"[train] epoch {epoch + 1:3d}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}  "
                f"thermal AUC {history['val_auc'][-1]:.3f}"
            )
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"[train] early stopping at epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred = torch.sigmoid(model(X[va_idx])).numpy()
    y_val, w_val = ds.y[va], ds.w[va]
    history.update(
        val_pred=pred,
        val_target=y_val,
        val_weight=w_val,
        best_val_loss=float(best),
        val_pearson=float(np.corrcoef(pred, y_val)[0, 1]),
        val_mae=float(np.average(np.abs(pred - y_val), weights=w_val)),
        val_thermal_auc=thermal_auc(pred, va_kind),
    )
    print(
        f"[train] best val loss {best:.4f} | Pearson r {history['val_pearson']:.3f} | "
        f"weighted MAE {history['val_mae']:.3f} | thermal AUC {history['val_thermal_auc']:.3f}"
    )
    if out_path:
        meta = {
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "best_val_loss": float(best),
            "val_pearson": history["val_pearson"],
            "val_thermal_auc": history["val_thermal_auc"],
            **(extra_meta or {}),
        }
        save_checkpoint(out_path, model, cfg.hidden, meta)
        print(f"[train] checkpoint saved to {out_path}")
    return model, history


def plot_report(history: dict, path: str) -> None:
    """Save a training report: loss curves and calibration of the validation predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    a0.plot(epochs, history["train_loss"], label="train")
    a0.plot(epochs, history["val_loss"], label="validation (held-out flights)")
    a0.set(xlabel="epoch", ylabel="weighted BCE", title="Loss")
    a0.legend()
    a0.grid(alpha=0.3)

    pred, target, weight = history["val_pred"], history["val_target"], history["val_weight"]
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(pred, bins) - 1, 0, 9)
    centres, means = [], []
    for b in range(10):
        m = which == b
        if weight[m].sum() > 0:
            centres.append(pred[m].mean())
            means.append(np.average(target[m], weights=weight[m]))
    a1.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    a1.plot(centres, means, "o-", label="validation")
    a1.set(
        xlabel="predicted lift index",
        ylabel="mean target lift index",
        title=f"Calibration (Pearson r = {history['val_pearson']:.2f})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    a1.legend()
    a1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[train] report saved to {path}")
