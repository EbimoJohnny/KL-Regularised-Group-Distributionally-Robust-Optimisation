import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

from dataset2 import get_dataloaders, challenge_f1, NUM_CLASSES
from model2 import CTViT

BASE_CFG = {
    "epochs":       100,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "patience":     15,
    "grad_clip":    1.0,
    "lr_backbone":  1e-5,
    "lr_head":      1e-4,
    "weight_decay": 1e-2,
    "dro_lr":       0.01,
}


KL_ALPHAS     = [0.5]

RESULTS_PATH  = "../checkpoints/dro_sweep_results.json"
os.makedirs("../checkpoints", exist_ok=True)


# ── Group DRO + KL Loss ───────────────────────────────────────────────────────
class GroupDROLoss(nn.Module):
    """
    Group DRO with KL regularisation.
    Groups = gender x class (8 total):
      male-A=0  male-G=1  male-covid=2  male-normal=3
      female-A=4  female-G=5  female-covid=6  female-normal=7
    """
    def __init__(self, num_classes, dro_lr, kl_alpha, device):
        super().__init__()
        self.num_classes = num_classes
        self.dro_lr      = dro_lr
        self.kl_alpha    = kl_alpha
        num_groups       = 2 * num_classes

        self.register_buffer(
            "weights", torch.ones(num_groups, device=device) / num_groups)
        self.register_buffer(
            "uniform", torch.ones(num_groups, device=device) / num_groups)

    def forward(self, logits, labels, genders):
        group_ids    = genders * self.num_classes + labels
        num_groups   = len(self.weights)
        group_losses = torch.zeros(num_groups, device=logits.device)

        for g in group_ids.unique():
            mask = group_ids == g
            if mask.sum() > 0:
                group_losses[g] = F.cross_entropy(
                    logits[mask], labels[mask], reduction="mean"
                )

        new_weights  = self.weights * torch.exp(self.dro_lr * group_losses.detach())
        self.weights = new_weights / new_weights.sum()

        kl = (self.weights * (self.weights / self.uniform).log()).sum()
        return (self.weights * group_losses).sum() + self.kl_alpha * kl


# ── Train / Eval ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, device, optimizer=None, verbose=False):
    training = optimizer is not None
    model.train() if training else model.eval()

    all_preds, all_labels, all_genders, total_loss = [], [], [], 0.0

    with torch.set_grad_enabled(training):
        for volumes, labels, genders in loader:
            volumes = volumes.to(device)
            labels  = labels.to(device)
            genders = genders.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(volumes)
            loss   = criterion(logits, labels, genders)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), BASE_CFG["grad_clip"])
                optimizer.step()

            total_loss    += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_genders.extend(genders.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = challenge_f1(all_labels, all_preds, all_genders, verbose=verbose)
    return avg_loss, f1


def get_detailed_f1(all_labels, all_preds, all_genders):
    """Return per-gender per-class F1 as a dict for storing in results."""
    labels  = np.array(all_labels)
    preds   = np.array(all_preds)
    genders = np.array(all_genders)

    class_names  = ["A", "G", "covid", "normal"]
    gender_names = {0: "male", 1: "female"}
    out = {}

    for g in [0, 1]:
        mask = genders == g
        if mask.sum() == 0:
            continue
        per = f1_score(labels[mask], preds[mask],
                       average=None, zero_division=0,
                       labels=list(range(NUM_CLASSES)))
        macro = f1_score(labels[mask], preds[mask],
                         average="macro", zero_division=0,
                         labels=list(range(NUM_CLASSES)))
        gname = gender_names[g]
        out[gname] = {cls: float(per[i]) for i, cls in enumerate(class_names)}
        out[gname]["macro"] = float(macro)

    out["overall"] = float(np.mean([out[g]["macro"] for g in out if g != "overall"]))
    return out


def print_group_weights(criterion):
    names = [f"{g}-{c}" for g in ["Male","Female"]
             for c in ["A","G","covid","normal"]]
    weights = criterion.weights.cpu().numpy()
    print("  Group weights:")
    for name, w in zip(names, weights):
        bar = "█" * int(w * 200)
        print(f"    {name:18s}: {w:.4f} {bar}")


# ── Single run ────────────────────────────────────────────────────────────────
def train_one(kl_alpha, train_loader, val_loader):
    device = BASE_CFG["device"]
    print(f"\n{'='*65}")
    print(f"  kl_alpha = {kl_alpha}")
    print(f"{'='*65}")

    model     = CTViT().to(device)
    criterion = GroupDROLoss(
        num_classes=NUM_CLASSES,
        dro_lr=BASE_CFG["dro_lr"],
        kl_alpha=kl_alpha,
        device=device,
    )

    optimizer = AdamW(model.parameters(), lr=BASE_CFG["lr_head"], weight_decay=BASE_CFG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # freeze backbone for first 5 epochs
    print("  Phase 1: backbone frozen (epochs 1-5)")
    for param in model.encoder.parameters():
        param.requires_grad = False

    best_f1       = 0.0
    best_detail   = {}
    best_epoch    = 0
    no_improve    = 0
    save_path     = f"../checkpoints/best_model_task2_dro_kl{kl_alpha}.pt"
    epoch_log     = []

    for epoch in range(1, BASE_CFG["epochs"] + 1):
        if epoch == 6:
            print("  Phase 2: backbone unfrozen")
            for param in model.encoder.parameters():
                param.requires_grad = True

        train_loss, train_f1 = run_epoch(model, train_loader, criterion, device, optimizer)

        # collect val predictions for detailed breakdown
        model.eval()
        all_preds, all_labels, all_genders = [], [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for volumes, labels, genders in val_loader:
                volumes = volumes.to(device)
                labels  = labels.to(device)
                genders = genders.to(device)
                logits  = model(volumes)
                val_loss_total += criterion(logits, labels, genders).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_genders.extend(genders.cpu().numpy())

        val_loss = val_loss_total / len(val_loader)
        val_f1   = challenge_f1(all_labels, all_preds, all_genders, verbose=True)
        detail   = get_detailed_f1(all_labels, all_preds, all_genders)
        scheduler.step(val_f1)

        if epoch % 5 == 0:
            print_group_weights(criterion)

        print(f"Epoch {epoch:02d}/{BASE_CFG['epochs']} | "
              f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}  F1: {val_f1:.4f}")

        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_f1":   round(train_f1,   4),
            "val_loss":   round(val_loss,    4),
            "val_f1":     round(val_f1,      4),
        })

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            best_detail  = detail
            best_epoch   = epoch
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (F1: {best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= BASE_CFG["patience"]:
                print(f"  Early stopping (no improvement for {BASE_CFG['patience']} epochs)")
                break

    print(f"\n  Best Val F1: {best_f1:.4f} at epoch {best_epoch}")
    return {
        "kl_alpha":    kl_alpha,
        "best_f1":     round(best_f1, 4),
        "best_epoch":  best_epoch,
        "best_detail": best_detail,
        "../checkpoints/":  save_path,
        "epoch_log":   epoch_log,
    }


def main():
    train_loader, val_loader = get_dataloaders()

    all_results = []

    for kl_alpha in KL_ALPHAS:
        result = train_one(kl_alpha, train_loader, val_loader)
        all_results.append(result)

        # save after every run so progress isn't lost if something crashes
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()