#!/bin/bash
# COCO Object Detection: AutoML-Agent vs Alchemist comparison
#
# Prerequisites:
#   1. COCO prepared: bash baselines/prepare_coco.sh (on EC2)
#   2. ultralytics installed: pip install ultralytics
#
# Both agents get:
#   - Same GPU (NVIDIA A10G 24GB or L40S 48GB)
#   - Same time budget (4 hours)
#   - Same COCO 2017 dataset
#   - Same evaluation metric (mAP50-95 on val2017)
#
# Usage:
#   bash baselines/run_coco_comparison.sh

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EC2_HOST="${EC2_HOST:-ubuntu@<EC2_IP>}"
EC2_KEY="${EC2_KEY:-$HOME/.ssh/alchemist-gpu-key-use1.pem}"

echo "================================================"
echo "  COCO Object Detection: AutoML-Agent vs Alchemist"
echo "================================================"

# --- Alchemist Agent ---
echo ""
echo "=== [1/2] Alchemist Agent ==="
mkdir -p logs

python main.py research \
  --base-model yolov8m \
  --task-name coco_detection \
  --task-desc "COCO 2017 object detection. 80 classes. Train: 118K images. Val: 5K images. Metric: mAP50-95. Budget: 4 hours." \
  --data-path /home/ubuntu/data/coco/coco.yaml \
  --num-classes 80 \
  --eval-metric mAP50-95 \
  --budget 4.0 \
  --max-trials 6 \
  --max-rounds 2 \
  --llm claude \
  --llm-model sonnet \
  --executor aws \
  --aws-host "$EC2_HOST" \
  --aws-key "$EC2_KEY" \
  --aws-work-dir /home/ubuntu/alchemist \
  --aws-python python3 \
  2>&1 | tee "logs/alchemist_coco_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=== [2/2] AutoML-Agent ==="
echo "  AutoML-Agent runs separately. See baselines/run_amla_coco.py"
echo ""
echo "=== Comparison complete. Check logs/ for results ==="
