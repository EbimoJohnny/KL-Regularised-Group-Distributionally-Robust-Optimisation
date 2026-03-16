
import os
import sys
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data"
CKPT_DIR = "../checkpoints"
OUT_DIR  = "../predictions"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SLICES = 64
IMG_SIZE   = 224
BATCH_SIZE = 4

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for sub in ["task1", "task2"]:
    p = os.path.join(SCRIPTS_DIR, sub)
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Test dataset — loads CT volume from a folder of slice images ──────────────
class CTTestDataset(Dataset):
    def __init__(self, test_dir, num_slices=NUM_SLICES, img_size=IMG_SIZE):
        self.test_dir  = test_dir
        self.num_slices = num_slices
        self.transform  = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        # each subdirectory is one CT volume
        self.scan_ids = sorted([
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        ])

    def __len__(self):
        return len(self.scan_ids)

    def _load_volume(self, scan_path):
        """Load all slices from a scan directory, sample/pad to num_slices."""
        EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dcm')

        def numeric_key(f):
            try:
                return (0, int(os.path.splitext(f)[0]))
            except ValueError:
                return (1, f)

        all_files = sorted(os.listdir(scan_path), key=numeric_key)
        slices    = [f for f in all_files if f.lower().endswith(EXTS)]

        if not slices:
            slices = all_files

        if not slices:
            # truly empty directory — return a blank volume
            blank = torch.zeros(1, self.num_slices,
                                IMG_SIZE, IMG_SIZE)
            return blank

        # sample or pad to exactly num_slices
        if len(slices) >= self.num_slices:
            idx    = np.linspace(0, len(slices) - 1,
                                 self.num_slices, dtype=int)
            slices = [slices[i] for i in idx]
        else:
            # tile slices to reach num_slices
            times  = (self.num_slices // len(slices)) + 1
            slices = (slices * times)[:self.num_slices]

        frames = []
        for fname in slices:
            fpath = os.path.join(scan_path, fname)
            try:
                if fname.lower().endswith('.dcm'):
                    import pydicom
                    ds  = pydicom.dcmread(fpath)
                    arr = ds.pixel_array.astype(np.float32)
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                    arr = (arr * 255).astype(np.uint8)
                    img = Image.fromarray(arr).convert("L")
                else:
                    img = Image.open(fpath).convert("L")
            except Exception:
                img = Image.fromarray(
                    np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))
            frames.append(self.transform(img))

        vol = torch.stack(frames, dim=1)  # (1, S, H, W)
        return vol

    def __getitem__(self, idx):
        scan_id   = self.scan_ids[idx]
        scan_path = os.path.join(self.test_dir, scan_id)
        volume    = self._load_volume(scan_path)
        return volume, scan_id


# ── Task 1 model ──────────────────────────────────────────────────────────────
def load_task1_model():
    from script.task1.model import CovidViT
    model = CovidViT().to(DEVICE)
    ckpt  = os.path.join(CKPT_DIR, "DRO_kl_alpha=0.5.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"  Task 1 model loaded from {ckpt}")
    return model


# ── Task 2 model ──────────────────────────────────────────────────────────────
def load_task2_model():
    from script.task2.model2 import CTViT
    model = CTViT().to(DEVICE)
    ckpt  = os.path.join(CKPT_DIR, "best_model_task2_dro_kl0.5.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"  Task 2 model loaded from {ckpt}")
    return model


# ── Predict loop ──────────────────────────────────────────────────────────────
def predict(model, loader):
    all_ids   = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for volumes, scan_ids in loader:
            volumes = volumes.to(DEVICE)
            logits  = model(volumes)
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_ids.extend(scan_ids)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return all_ids, all_preds, all_probs


# ── Save predictions ──────────────────────────────────────────────────────────
def save_task1_predictions(scan_ids, preds, probs, out_path):
    """
    Task 1: binary COVID / non-COVID
    Label map: 0 = non-COVID, 1 = COVID
    """
    label_map = {0: "non-covid", 1: "covid"}
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scan_id", "prediction", "label",
                         "prob_non_covid", "prob_covid"])
        for sid, pred, prob in zip(scan_ids, preds, probs):
            writer.writerow([
                sid,
                int(pred),
                label_map[int(pred)],
                round(prob[0], 4),
                round(prob[1], 4),
            ])
    print(f"  Saved {len(scan_ids)} predictions → {out_path}")


def save_task2_predictions(scan_ids, preds, probs, out_path):
    """
    Task 2: 4-class lung pathology
    Label map: 0=A, 1=G, 2=covid, 3=normal
    """
    label_map = {0: "A", 1: "G", 2: "covid", 3: "normal"}
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scan_id", "prediction", "label",
                         "prob_A", "prob_G", "prob_covid", "prob_normal"])
        for sid, pred, prob in zip(scan_ids, preds, probs):
            writer.writerow([
                sid,
                int(pred),
                label_map[int(pred)],
                round(prob[0], 4),
                round(prob[1], 4),
                round(prob[2], 4),
                round(prob[3], 4),
            ])
    print(f"  Saved {len(scan_ids)} predictions → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    # ── Task 1 ───────────────────────────────────────────────────────────────
    task1_test_dir = os.path.join(DATA_DIR, "task1", "test")
    if os.path.exists(task1_test_dir):
        print("═" * 50)
        print("  TASK 1 — COVID-19 Detection")
        print("═" * 50)

        dataset = CTTestDataset(task1_test_dir)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)
        print(f"  Found {len(dataset)} test volumes")

        model  = load_task1_model()
        ids, preds, probs = predict(model, loader)

        out = os.path.join(OUT_DIR, "task1_predictions.csv")
        save_task1_predictions(ids, preds, probs, out)

    else:
        print(f"  [SKIP] Task 1 test dir not found: {task1_test_dir}")

    print()

    # ── Task 2 ───────────────────────────────────────────────────────────────
    task2_test_dir = os.path.join(DATA_DIR, "task2", "test")
    if os.path.exists(task2_test_dir):
        print("═" * 50)
        print("  TASK 2 — Lung Pathology")
        print("═" * 50)

        dataset = CTTestDataset(task2_test_dir)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)
        print(f"  Found {len(dataset)} test volumes")

        model  = load_task2_model()
        ids, preds, probs = predict(model, loader)

        out = os.path.join(OUT_DIR, "task2_predictions.csv")
        save_task2_predictions(ids, preds, probs, out)
5
    else:
        print(f"  [SKIP] Task 2 test dir not found: {task2_test_dir}")

    print(f"\nDone. Predictions saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()