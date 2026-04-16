#!/bin/bash
# Download benchmark datasets (Shopee-IET, optionally Butterfly) to EC2 NVMe.
#
# Shopee-IET: AutoGluon public S3 mirror (41.9 MB, identical to AutoML-Agent paper).
# Butterfly:  needs Kaggle credentials — see instructions at bottom.
#
# Usage: run on EC2, not local:
#   ssh ec2  "bash /home/ubuntu/alchemist/baselines/prepare_datasets.sh"

set -e
ROOT=/home/ubuntu/data
mkdir -p "$ROOT"
LOG=/tmp/dataset_prep.log
echo "[$(date -Is)] prepare_datasets START" > "$LOG"

# ----------------------------------------------------------------------
# 1. Shopee-IET (public S3, no auth needed)
# ----------------------------------------------------------------------
if [ -d "$ROOT/shopee-iet/train" ]; then
    echo "shopee-iet already prepared, skipping" | tee -a "$LOG"
else
    echo "=== Shopee-IET download ===" | tee -a "$LOG"
    cd "$ROOT"
    curl -L -o shopee-iet.zip \
      https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip 2>&1 | tail -2
    unzip -q -o shopee-iet.zip -d shopee-iet_raw
    rm shopee-iet.zip
    # Normalize to train/val/test ImageFolder layout. AutoGluon's zip
    # typically contains a nested folder; flatten if needed.
    if [ -d "$ROOT/shopee-iet_raw/shopee-iet" ]; then
        mv "$ROOT/shopee-iet_raw/shopee-iet" "$ROOT/shopee-iet"
        rm -rf "$ROOT/shopee-iet_raw"
    else
        mv "$ROOT/shopee-iet_raw" "$ROOT/shopee-iet"
    fi
    echo "  shopee-iet structure:" | tee -a "$LOG"
    find "$ROOT/shopee-iet" -maxdepth 3 -type d | head -20 | tee -a "$LOG"
    echo "  image counts:" | tee -a "$LOG"
    for split in train val valid test; do
        if [ -d "$ROOT/shopee-iet/$split" ]; then
            cnt=$(find "$ROOT/shopee-iet/$split" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) | wc -l)
            echo "    $split: $cnt" | tee -a "$LOG"
        fi
    done
fi

# ----------------------------------------------------------------------
# 2. Butterfly Image Classification (needs Kaggle auth)
# ----------------------------------------------------------------------
if [ -d "$ROOT/butterfly/train" ]; then
    echo "butterfly already prepared, skipping" | tee -a "$LOG"
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "=== Butterfly download via Kaggle API ===" | tee -a "$LOG"
    chmod 600 "$HOME/.kaggle/kaggle.json"
    pip install -q kaggle 2>&1 | tail -1
    cd "$ROOT"
    kaggle datasets download -d phucthaiv02/butterfly-image-classification
    unzip -q -o butterfly-image-classification.zip -d butterfly_raw
    rm butterfly-image-classification.zip
    # Reshape to ImageFolder layout if CSV-indexed
    python3 <<'PY' 2>&1 | tee -a "$LOG"
import os, shutil
from pathlib import Path
root = Path("/home/ubuntu/data/butterfly_raw")
out  = Path("/home/ubuntu/data/butterfly")
# Kaggle format: Training_set.csv with columns "filename,label", images in "train/" folder
import csv
def fold(csv_name, img_dir, dest_split):
    csv_path = root / csv_name
    if not csv_path.exists():
        return 0
    (out / dest_split).mkdir(parents=True, exist_ok=True)
    n = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label") or row.get("class") or row.get("Class") or "unknown"
            fn = row.get("filename") or row.get("image_path") or row.get("image")
            src = root / img_dir / fn
            if not src.exists():
                continue
            (out / dest_split / label).mkdir(parents=True, exist_ok=True)
            shutil.copy(src, out / dest_split / label / fn)
            n += 1
    return n

n_train = fold("Training_set.csv", "train", "train")
n_val   = fold("Testing_set.csv",  "test",  "val")
print(f"  butterfly: train={n_train} val={n_val}")
PY
    rm -rf "$ROOT/butterfly_raw"
else
    echo "⚠️  butterfly skipped — provide Kaggle credentials:" | tee -a "$LOG"
    echo "    1. visit https://kaggle.com/settings → API → Create New Token" | tee -a "$LOG"
    echo "    2. copy kaggle.json to $HOME/.kaggle/kaggle.json" | tee -a "$LOG"
    echo "    3. re-run this script" | tee -a "$LOG"
fi

echo "[$(date -Is)] prepare_datasets DONE" | tee -a "$LOG"
df -h /home/ubuntu | tail -1 | tee -a "$LOG"
