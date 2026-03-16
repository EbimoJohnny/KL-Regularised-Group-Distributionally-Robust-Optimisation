# submission

This folder contains scripts and helpers for running predictions for the CVPR submission.

Contents
--------

- `predict.py` — main prediction runner. Loads pre-trained checkpoints for Task 1 and Task 2, scans a folder of CT slice images per volume, runs inference, and writes CSV prediction files to `../predictions`.
- `data/` — expected dataset directory (not included here). The code expects `../data/task1/test` and `../data/task2/test` folders containing one subfolder per CT volume (each subfolder holds 2D image slices or DICOMs).
- `script/` — training and model code used by the predictor:
  - `script/task1/`
    - `dataset.py` — dataset utilities for task 1
    - `model.py` — model definition (CovidViT)
    - `train.py` — training script for task 1
  - `script/task2/`
    - `dataset2.py` — dataset utilities for task 2
    - `model2.py` — model definition (CTViT)
    - `train.py` — training script for task 2

Quick summary of `predict.py`
-----------------------------

- Device: uses CUDA if available.
- Expected directories (relative to `submission/`):
  - `../data` — test data (see above)
  - `../checkpoints` — model checkpoint files (the script uses the following filenames by default):
    - Task 1: `DRO_kl_alpha=0.5.pt`
    - Task 2: `best_model_task2_dro_kl0.5.pt`
  - `../predictions` — output CSV files are written here (`task1_predictions.csv`, `task2_predictions.csv`). The directory is created if missing.

Predicted CSV formats
---------------------
- Task 1 CSV columns: `scan_id`, `prediction`, `label`, `prob_non_covid`, `prob_covid` (binary task: 0=non-covid, 1=covid).
- Task 2 CSV columns: `scan_id`, `prediction`, `label`, `prob_A`, `prob_G`, `prob_covid`, `prob_normal` (4-class lung pathology: 0=A, 1=G, 2=covid, 3=normal).

Dependencies
------------

The runner requires (minimum):

- Python 3.8+
- torch
- torchvision
- pillow (PIL)
- numpy
- pydicom (optional, only needed if test data include `.dcm` slices which was not included in this challange)

Install with pip (example):

```powershell
python -m pip install torch torchvision
```

How to run
----------

Run the training and prediction scripts from the `submission/` directory (recommended). The code uses relative paths like `../data` and `../checkpoints` so running from `submission/` makes those point to the repository root.

Train (recommended)
-------------------

Run:

```powershell
cd submission
# Task 1: training (Covid detection)
python script/task1/train.py

# Task 2: training (lung pathology)
python script/task2/train.py
```

Alternatively, you can run each training script from its own folder:

```powershell
cd submission/script/task1
python train.py

cd ../task2
python train.py
```

Inference / Predictions
-----------------------

After training (or if you already have checkpoints), run predictions from `submission/` so the script finds `../data` and `../checkpoints` correctly:

```powershell
cd submission
python predict.py
```

If you must run the scripts from a different working directory, either adjust the relative paths at the top of the scripts (`DATA_DIR`, `CKPT_DIR`, `OUT_DIR`, `CHECKPOINT_DIR`) or run the scripts through a short wrapper that sets the working directory to `submission` before executing the script.

Notes and suggestions
---------------------

- Ensure checkpoint files listed above exist in `../checkpoints` relative to where you run `predict.py`. If you have differently-named checkpoints, update `CKPT_DIR` and the filenames inside `predict.py` or replace the files with the expected names.
- The `data` layout should be:
  - `data/task1/test/<scan_id>/*.png` (or .jpg, .dcm, etc.)
  - `data/task2/test/<scan_id>/*.png`
- `predict.py` uses simple sampling/padding to make volumes have exactly 64 slices and resizes images to 224×224; adjust `NUM_SLICES` and `IMG_SIZE` at the top of the file if needed.
- For large test sets, consider increasing `BATCH_SIZE` or `num_workers` in the DataLoader to improve throughput.


