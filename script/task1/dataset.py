import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "../../data/task1")
SLICE_SIZE  = (224, 224)
NUM_SLICES  = 64        # fixed slices per volume (sample or pad)
BATCH_SIZE  = 8

# ── Build master dataframe ───────────────────────────────────────────────────
def build_master_csv():
    records = []

 

    sources = [
        ("train_covid.csv",          ["train/covid1", "train/covid2"],                    1, "train"),
        ("train_non_covid.csv",      ["train/non-covid1","train/non-covid2","train/non-covid3"], 0, "train"),
        ("validation_covid.csv",     ["val/covid"],                           1, "val"),
        ("validation_non_covid.csv", ["val/non-covid"],                       0, "val"),
    ]
    for csv_file, scan_dirs, label, split in sources:
        csv_path = os.path.join(DATA_ROOT, csv_file)
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            scan_name   = row["ct_scan_name"]
            data_centre = row["data_centre"]

            scan_path = None
            for d in scan_dirs:
                candidate = os.path.join(DATA_ROOT, d, scan_name)
                if os.path.isdir(candidate):
                    scan_path = candidate
                    break

            if scan_path and any(f.endswith(".jpg") for f in os.listdir(scan_path)):
                records.append({
                    "scan_path":   scan_path,
                    "label":       label,
                    "data_centre": data_centre,
                    "split":       split,
                })

    return pd.DataFrame(records)


class CovidCTDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize(SLICE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(SLICE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def _load_volume(self, scan_path):
        # load slices in correct numerical order (handles flat and nested subdirs)
        files = sorted(
            [os.path.join(r, f) for r, _, fs in os.walk(scan_path) for f in fs if f.endswith(".jpg") and not f.startswith("._")],
            key=lambda x: int(os.path.basename(x).replace(".jpg", ""))
        )

        if len(files) == 0:
            raise RuntimeError(f"No jpg slices found in {scan_path}")

        if len(files) >= NUM_SLICES:
            idx = np.linspace(0, len(files)-1, NUM_SLICES, dtype=int)
            files = [files[i] for i in idx]
        else:
            # print(f"Warning: {scan_path} has only {len(files)} slices, padding to {NUM_SLICES}, and the center is {self.df[self.df.scan_path == scan_path]['data_centre'].values[0]}")
            files = files + [files[-1]] * (NUM_SLICES - len(files))  # pad with last slice

        slices = []
        for f in files:
            img = Image.open(f).convert("L")  # grayscale
            slices.append(self.transform(img))

        return torch.stack(slices, dim=1)  # (1, NUM_SLICES, H, W)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        volume = self._load_volume(row["scan_path"])
        label  = torch.tensor(row["label"], dtype=torch.long)
    
        centre = torch.tensor(row["data_centre"], dtype=torch.long)
        # if centre == 2:
        #     print(f"Scan {row['scan_path']} from centre {centre} has only {volume.shape[1]} slices, padding to {NUM_SLICES}")
        return volume, label, centre





# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_dataloaders():
    master = build_master_csv()
    print(f"Total scans — train: {(master.split=='train').sum()}, val: {(master.split=='val').sum()}")
    print(f"Label distribution:\n{master.groupby(['split','label']).size()}\n")

    train_ds = CovidCTDataset(master[master.split == "train"], augment=True)
    val_ds   = CovidCTDataset(master[master.split == "val"],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


# # ── Quick sanity check ────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     train_loader, val_loader = get_dataloaders()
#     volume, label, centre = next(iter(train_loader))
#     print(f"Volume shape : {volume.shape}")   # (B, 1, NUM_SLICES, H, W)
#     print(f"Labels       : {label}")
#     print(f"Data centres : {centre}")