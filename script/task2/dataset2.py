import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import f1_score

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(os.path.dirname(__file__), "../../data/task2/")
SLICE_SIZE = (224, 224)
NUM_SLICES = 64
BATCH_SIZE = 4

# Gender mapping
GENDER_MAP = {"male": 0, "female": 1}

# Fixed label map — covers all folder name variants across train, train1, and val
# A = Adenocarcinoma, G = Squamous/Glandular, covid = Covid-19, normal = Healthy
CLASS_LABEL_MAP = {
    "A":      0,   # Adenocarcinoma (train / train1)
    "G":      1,   # Squamous       (train / train1)
    "covid":  2,   # Covid-19       (val)
    "normal": 3,   # Healthy        (val)
}


NUM_CLASSES = len(CLASS_LABEL_MAP)


# ── Build master dataframe from folder structure ──────────────────────────────
# Structure: task2/{split_folder}/{class}/{gender}/{scan_name}/
# train and train1 are both treated as "train" split.
def build_master_df():
    import pandas as pd

    records = []

    # Map folder name → split label
    # train1 is treated identically to train — just more training data
    split_folders = {
        "train1":  "train",
        "train2": "train",   # ← additional training data, same structure
        "val":    "val",
        # "test":   "test",
    }

    for folder_name, split in split_folders.items():
        split_dir = os.path.join(DATA_ROOT, folder_name)
        if not os.path.isdir(split_dir):
            print(f"  [INFO] Folder not found (skipping): {split_dir}")
            continue

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            if class_name not in CLASS_LABEL_MAP:
                print(f"  [WARN] Unknown class folder '{class_name}' in {folder_name} — skipping")
                continue

            label = CLASS_LABEL_MAP[class_name]

            for gender_name in sorted(os.listdir(class_dir)):
                gender_dir = os.path.join(class_dir, gender_name)
                if not os.path.isdir(gender_dir) or gender_name not in GENDER_MAP:
                    continue

                gender = GENDER_MAP[gender_name]

                for scan_name in sorted(os.listdir(gender_dir)):
                    scan_path = os.path.join(gender_dir, scan_name)
                    if not os.path.isdir(scan_path):
                        continue

                    has_jpg = any(
                        f.endswith(".jpg") and not f.startswith("._")
                        for f in os.listdir(scan_path)
                    )
                    if not has_jpg:
                        continue

                    records.append({
                        "scan_path":  scan_path,
                        "class_name": class_name,
                        "label":      label,
                        "gender":     gender,
                        "split":      split,
                    })

    df = pd.DataFrame(records)
    return df


# ── Gender x class balanced sampler ─────────────────────────────────────────
def make_weighted_sampler(df):
    """
    Weight each sample by 1 / (count of its gender+label group).
    Ensures every gender x class combination is seen equally during training,
    directly optimising for the per-gender macro F1 metric.
    """
    df = df.copy().reset_index(drop=True)
    group_counts = df.groupby(["gender", "label"])["label"].transform("count")
    weights = 1.0 / group_counts.values
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(df),
        replacement=True,
    )


class CTDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)

        base = [
            transforms.Resize(SLICE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        aug = [
            transforms.Resize(SLICE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            # transforms.RandomAdjustSharpness(2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        self.transform = transforms.Compose(aug if augment else base)

    def __len__(self):
        return len(self.df)

    def _load_volume(self, scan_path):
        files = sorted(
            [os.path.join(r, f)
             for r, _, fs in os.walk(scan_path)
             for f in fs if f.endswith(".jpg") and not f.startswith("._")],
            key=lambda x: int(os.path.basename(x).replace(".jpg", ""))
        )
        if len(files) == 0:
            raise RuntimeError(f"No jpg slices found in {scan_path}")

        if len(files) >= NUM_SLICES:
            files = [files[i] for i in np.linspace(0, len(files)-1, NUM_SLICES, dtype=int)]
        else:
            files = files + [files[-1]] * (NUM_SLICES - len(files))

        slices = [self.transform(Image.open(f).convert("L")) for f in files]
        return torch.stack(slices, dim=1)  # (1, NUM_SLICES, H, W)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            self._load_volume(row["scan_path"]),
            torch.tensor(row["label"],  dtype=torch.long),
            torch.tensor(row["gender"], dtype=torch.long),
        )


def challenge_f1(labels, preds, genders, verbose=False):
    """
    P = (macro_F1_male + macro_F1_female) / 2
    macro F1 = unweighted average F1 across all NUM_CLASSES for that gender.
    """
    labels  = np.array(labels)
    preds   = np.array(preds)
    genders = np.array(genders)

    gender_scores = []
    gender_names  = {0: "Male", 1: "Female"}
    class_names   = {v: k for k, v in CLASS_LABEL_MAP.items()}

    for g in [0, 1]:
        mask = genders == g
        if mask.sum() == 0:
            if verbose:
                print(f"  {gender_names[g]}: no samples")
            continue

        score = f1_score(
            labels[mask], preds[mask],
            average="macro",
            zero_division=0,
            labels=list(range(NUM_CLASSES)),
        )
        gender_scores.append(score)

        if verbose:
            per_class = f1_score(
                labels[mask], preds[mask],
                average=None,
                zero_division=0,
                labels=list(range(NUM_CLASSES)),
            )
            details = "  ".join(
                f"F1_{class_names[i]}: {per_class[i]:.4f}"
                for i in range(NUM_CLASSES)
            )
            print(f"  {gender_names[g]} | {details} | macro: {score:.4f}")

    return np.mean(gender_scores) if gender_scores else 0.0


def get_dataloaders():
    master = build_master_df()

    print(f"Class label map: {CLASS_LABEL_MAP}")
    print(f"Total scans — train: {(master.split=='train').sum()}, val: {(master.split=='val').sum()}")
    print(f"\nPer-gender class distribution:")
    print(master.groupby(["split", "gender", "class_name"]).size().to_string())
    print()

    train_df = master[master.split == "train"].reset_index(drop=True)
    val_df   = master[master.split == "val"].reset_index(drop=True)

    train_ds = CTDataset(train_df, augment=True)
    val_ds   = CTDataset(val_df,   augment=False)

    sampler = make_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader


# # ── Sanity check ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     train_loader, val_loader = get_dataloaders()
#     volume, label, gender = next(iter(train_loader))
#     print(f"Volume : {volume.shape}")
#     print(f"Labels : {label}")
#     print(f"Genders: {gender}")
    