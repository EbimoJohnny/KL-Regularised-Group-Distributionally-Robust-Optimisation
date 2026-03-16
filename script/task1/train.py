
import os, json, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import f1_score
from transformers import get_cosine_schedule_with_warmup
import timm

from dataset import get_dataloaders   # your existing dataset module

CHECKPOINT_DIR = "../checkpoints"
RESULTS_PATH   = os.path.join(CHECKPOINT_DIR, "results.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BASE_CFG = {
    "backbone":      "mobilevit_xxs",
    "pretrained":    True,
    "num_slices":    64,
    "embed_dim":     320,
    "num_classes":   2,
    "dropout":       0.3,
    "epochs":        50,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "warmup_steps":  5,
    "patience":      10,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "class_weights": [1223 / 659, 1223 / 564],
}



class SliceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg["backbone"], pretrained=cfg["pretrained"], num_classes=0
        )
        # adapt first conv to 1-channel input
        w = self.backbone.stem.conv.weight.data          # (out, 3, k, k)
        self.backbone.stem.conv.weight = nn.Parameter(w.mean(dim=1, keepdim=True))
        self.backbone.stem.conv.in_channels = 1

    def forward(self, x):
        return self.backbone(x)   # (B, embed_dim)


class SliceTransformer(nn.Module):
    """Transformer-based slice aggregator."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        return self.encoder(x).mean(dim=1)   # (B, embed_dim)


class SliceAggregator(nn.Module):
    """Attention-weighted pooling aggregator."""
    def __init__(self, embed_dim, num_slices):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)   # (B, S, 1)
        return (x * w).sum(dim=1)                 # (B, embed_dim)


class CovidViT(nn.Module):
    def __init__(self, cfg, use_transformer=True):
        super().__init__()
        embed_dim = cfg["embed_dim"]
        self.encoder    = SliceEncoder(cfg)
        self.aggregator = (
            SliceTransformer(embed_dim)
            if use_transformer
            else SliceAggregator(embed_dim, cfg["num_slices"])
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(embed_dim, cfg["num_classes"]),
        )

    def forward(self, x):
        B, C, S, H, W = x.shape
        x     = x.permute(0, 2, 1, 3, 4).reshape(B * S, C, H, W)
        feats = self.encoder(x).view(B, S, -1)
        return self.head(self.aggregator(feats))


class DomainBalancedLoss(nn.Module):
    def __init__(self, class_weights, device):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device),
            reduction="mean",
        )

    def forward(self, logits, labels, centres):
        losses = [
            self.ce(logits[centres == c], labels[centres == c])
            for c in centres.unique()
        ]
        return torch.stack(losses).mean()


class GroupDROLoss(nn.Module):
    """
    Group DRO with KL regularisation toward uniform group weights.
    kl_alpha=0.0 collapses to plain per-group CE (no DRO dynamics).
    """
    def __init__(self, num_groups, dro_lr, kl_alpha, class_weights, device):
        super().__init__()
        self.dro_lr   = dro_lr
        self.kl_alpha = kl_alpha
        cw = torch.tensor(class_weights, dtype=torch.float).to(device)
        self.ce = nn.CrossEntropyLoss(weight=cw, reduction="mean")
        self.register_buffer("weights", torch.ones(num_groups, device=device) / num_groups)
        self.register_buffer("uniform", torch.ones(num_groups, device=device) / num_groups)

    def forward(self, logits, labels, centres):
        group_losses = torch.zeros(len(self.weights), device=logits.device)
        for c in centres.unique():
            mask = centres == c
            if mask.sum() > 0:
                group_losses[int(c)] = self.ce(logits[mask], labels[mask])

        # exponentiated gradient update
        self.weights = self.weights * torch.exp(self.dro_lr * group_losses.detach())
        self.weights = self.weights / self.weights.sum()

        kl = (self.weights * (self.weights / self.uniform).log()).sum()
        return (self.weights * group_losses).sum() + self.kl_alpha * kl


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None, device="cpu"):
        super().__init__()
        self.gamma = gamma
        self.weight = (
            torch.tensor(class_weights, dtype=torch.float).to(device)
            if class_weights else None
        )

    def forward(self, logits, labels, centres=None):
        ce    = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt    = torch.exp(-ce)
        loss  = ((1 - pt) ** self.gamma) * ce
        return loss.mean()



def challenge_f1(labels, preds, centres, verbose=False):
    """1/4 * Σ_i (F1_covid + F1_noncovid) / 2  for each centre i."""
    labels, preds, centres = map(np.array, (labels, preds, centres))
    centre_scores = []
    for c in range(4):
        mask = centres == c
        if not mask.any():
            continue
        f1c = f1_score(labels[mask], preds[mask], pos_label=1, zero_division=0)
        f1n = f1_score(labels[mask], preds[mask], pos_label=0, zero_division=0)
        avg = (f1c + f1n) / 2
        centre_scores.append(avg)
        if verbose:
            print(f"    Centre {c} | covid: {f1c:.4f}  non-covid: {f1n:.4f}  avg: {avg:.4f}")
    return float(np.mean(centre_scores)) if centre_scores else 0.0

def run_epoch(model, loader, criterion, optimizer, scheduler, device, verbose=False):
    training = optimizer is not None
    model.train() if training else model.eval()

    all_preds, all_labels, all_centres, total_loss = [], [], [], 0.0

    with torch.set_grad_enabled(training):
        for volumes, labels, centres in loader:
            volumes, labels, centres = (
                volumes.to(device), labels.to(device), centres.to(device)
            )
            logits = model(volumes)
            loss   = criterion(logits, labels, centres)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss    += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_centres.extend(centres.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = challenge_f1(all_labels, all_preds, all_centres, verbose=verbose)
    return avg_loss, f1



def run_experiment(exp_name, cfg, criterion_fn, use_transformer=True,
                   train_loader=None, val_loader=None):
    """
    Train one model configuration and return a result dict.

    Parameters
    ----------
    exp_name       : str   – human-readable experiment name
    cfg            : dict  – full config
    criterion_fn   : callable(device) → nn.Module
    use_transformer: bool  – SliceTransformer vs SliceAggregator
    """
    device    = cfg["device"]
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp_name.replace(' ', '_')}.pt")

    print(f"\n{'═'*70}")
    print(f"  Experiment : {exp_name}")
    print(f"  Aggregator : {'SliceTransformer' if use_transformer else 'SliceAggregator'}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'═'*70}")

    model     = CovidViT(cfg, use_transformer=use_transformer).to(device)
    criterion = criterion_fn(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=cfg["epochs"] * len(train_loader),
    )

    best_f1, no_improve = 0.0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        va_loss, va_f1 = run_epoch(model, val_loader,   criterion, None,      None,      device,
                                   verbose=(epoch % 10 == 0))

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6), "train_f1": round(tr_f1, 6),
            "val_loss":   round(va_loss, 6), "val_f1":   round(va_f1, 6),
        })

        improved = va_f1 > best_f1
        marker   = " ✓" if improved else ""
        print(f"  [{epoch:02d}/{cfg['epochs']}] "
              f"TR loss={tr_loss:.4f} f1={tr_f1:.4f} | "
              f"VA loss={va_loss:.4f} f1={va_f1:.4f}{marker}")

        if improved:
            best_f1, no_improve = va_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg['patience']} epochs)")
                break

    elapsed = round(time.time() - t0, 1)
    print(f"\n  ➜ Best Val F1 = {best_f1:.4f}  |  Time = {elapsed}s")

    return {
        "experiment":    exp_name,
        "aggregator":    "SliceTransformer" if use_transformer else "SliceAggregator",
        "best_val_f1":   round(best_f1, 6),
        "checkpoint":    ckpt_path,
        "elapsed_sec":   elapsed,
        "history":       history,
    }



def make_dro_criterion(kl_alpha, dro_lr=0.01):
    def factory(device):
        return GroupDROLoss(
            num_groups=4, dro_lr=dro_lr, kl_alpha=kl_alpha,
            class_weights=BASE_CFG["class_weights"], device=device,
        )
    return factory


def make_focal_criterion(gamma=2.0):
    def factory(device):
        return FocalLoss(gamma=gamma, class_weights=BASE_CFG["class_weights"], device=device)
    return factory


EXPERIMENTS_AB = [

    ("DRO kl_alpha=0.5", make_dro_criterion(0.5), True),
]


def main():
    train_loader, val_loader = get_dataloaders()

    all_results  = []
    loader_kwargs = dict(train_loader=train_loader, val_loader=val_loader)



    for exp_name, criterion_fn, use_tf in EXPERIMENTS_AB:
        result = run_experiment(
            exp_name, BASE_CFG, criterion_fn,
            use_transformer=use_tf, **loader_kwargs,
        )
        all_results.append(result)
        _save_results(all_results)   # checkpoint after every run




def _save_results(results):
    """Persist results list to JSON (called after each experiment)."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()