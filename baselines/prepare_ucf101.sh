#!/bin/bash
# Download and prepare UCF-101 dataset for video classification.
# Run on EC2: bash prepare_ucf101.sh
#
# Structure after setup:
#   /home/ubuntu/data/ucf101/
#   ├── train/<class>/*.avi
#   └── val/<class>/*.avi

set -eu
DATA_DIR="/home/ubuntu/data/ucf101"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading UCF-101 ==="
if [ ! -f UCF101.rar ]; then
    wget -q --show-progress "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar" -O UCF101.rar
fi

echo "=== Downloading train/test splits ==="
if [ ! -f UCF101TrainTestSplits-RecognitionTask.zip ]; then
    wget -q --show-progress "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip" -O UCF101TrainTestSplits-RecognitionTask.zip
fi

echo "=== Extracting ==="
if [ ! -d UCF-101 ]; then
    unrar x -y UCF101.rar
fi
unzip -qo UCF101TrainTestSplits-RecognitionTask.zip

echo "=== Organizing into train/val splits ==="
python3 << 'PY'
import os, shutil
from pathlib import Path

data_dir = Path("/home/ubuntu/data/ucf101")
video_dir = data_dir / "UCF-101"
split_dir = data_dir / "ucfTrainTestlist"

# Read split 1
train_files = set()
with open(split_dir / "trainlist01.txt") as f:
    for line in f:
        parts = line.strip().split()
        train_files.add(parts[0])  # e.g., "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"

test_files = set()
with open(split_dir / "testlist01.txt") as f:
    for line in f:
        test_files.add(line.strip())

# Create directories
for split in ["train", "val"]:
    (data_dir / split).mkdir(exist_ok=True)

# Move files
moved = {"train": 0, "val": 0}
for cls_dir in sorted(video_dir.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls_name = cls_dir.name
    (data_dir / "train" / cls_name).mkdir(exist_ok=True)
    (data_dir / "val" / cls_name).mkdir(exist_ok=True)

    for video in cls_dir.iterdir():
        if not video.is_file():
            continue
        rel_path = f"{cls_name}/{video.name}"
        if rel_path in train_files:
            dst = data_dir / "train" / cls_name / video.name
        elif rel_path in test_files:
            dst = data_dir / "val" / cls_name / video.name
        else:
            dst = data_dir / "train" / cls_name / video.name  # default to train

        if not dst.exists():
            shutil.copy2(str(video), str(dst))
            moved["train" if "train" in str(dst) else "val"] += 1

print(f"Done: train={moved['train']}, val={moved['val']}")
# Count
train_count = sum(1 for _ in (data_dir / "train").rglob("*.avi"))
val_count = sum(1 for _ in (data_dir / "val").rglob("*.avi"))
classes = len(list((data_dir / "train").iterdir()))
print(f"Final: {classes} classes, {train_count} train videos, {val_count} val videos")
PY

echo "=== Installing decord ==="
pip install -q decord

echo "=== Done! Dataset at $DATA_DIR ==="
ls -la "$DATA_DIR/train/" | head -5
echo "..."
ls -la "$DATA_DIR/train/" | wc -l
echo "classes total"
