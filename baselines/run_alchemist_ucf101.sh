#!/bin/bash
# Alchemist on UCF-101 Video Classification with V-JEPA.
#
# Prerequisites:
#   1. UCF-101 prepared: bash baselines/prepare_ucf101.sh (on EC2)
#   2. decord installed on EC2: pip install decord
#
# Usage:
#   bash baselines/run_alchemist_ucf101.sh

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EC2_HOST="${EC2_HOST:-ubuntu@18.234.183.81}"
EC2_KEY="${EC2_KEY:-$HOME/.ssh/alchemist-gpu-key-use1.pem}"

TASK_DESC=$(cat <<'DESC'
Video action recognition on UCF-101. 101 action classes from YouTube videos. Training set: ~9,537 videos. Validation set: ~3,783 videos. Evaluation metric: Top-1 accuracy on validation set. Hardware: NVIDIA A10G (24GB). Time budget: 4 hours. ImageNet-1K AND ImageNet-21K pretrained models allowed. V-JEPA self-supervised pretrained models allowed.
DESC
)

mkdir -p logs
LOG=logs/alchemist_ucf101_$(date +%Y%m%d_%H%M%S).log

python main.py research \
  --base-model vjepa_vit_huge \
  --task-name ucf101 \
  --task-desc "$TASK_DESC" \
  --data-path /home/ubuntu/data/ucf101 \
  --num-classes 101 \
  --eval-metric top1_accuracy \
  --budget 4.0 \
  --max-trials 8 \
  --max-rounds 2 \
  --llm claude \
  --llm-model sonnet \
  --executor aws \
  --aws-host "$EC2_HOST" \
  --aws-key "$EC2_KEY" \
  --aws-work-dir /home/ubuntu/alchemist \
  --aws-python /opt/pytorch/bin/python \
  2>&1 | tee "$LOG"

echo
echo "=== Alchemist UCF-101 done. Log: $LOG ==="
