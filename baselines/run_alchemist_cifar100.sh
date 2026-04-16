#!/bin/bash
# Alchemist on CIFAR-100 — fair comparison twin of run_amla_cifar100.py.
#
# Same constraints as AutoML-Agent baseline:
#   - Same GPU: A10G × 1 (via AWSExecutor SSH to EC2)
#   - Same LLM: Claude CLI (ClaudeCLIClient in alchemist/core/llm.py)
#   - Same data: /home/ubuntu/data/cifar100/{train,val}
#   - Same time budget: 8 GPU-hours
#   - Same rules: ImageNet-1K pretrained OK; no external data; val-only eval
#   - No hints about specific architectures/techniques
#
# Usage:
#   bash baselines/run_alchemist_cifar100.sh
#
# Logs:  logs/alchemist_cifar100_*.jsonl   (run), experiments/*

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EC2_HOST="${EC2_HOST:-ubuntu@3.90.166.150}"
EC2_KEY="${EC2_KEY:-$HOME/.ssh/alchemist-gpu-key-use1.pem}"
BUDGET="${BUDGET:-8.0}"

TASK_DESC=$(cat <<'DESC'
Build a model to classify images from the CIFAR-100 dataset (100 fine-grained classes, 32x32 RGB). Training set: 50,000 images at /home/ubuntu/data/cifar100/train/<class>/*.jpg. Validation set: 10,000 images at /home/ubuntu/data/cifar100/val/<class>/*.jpg. Evaluation metric: Top-1 accuracy on the validation set (single number, 2 decimal places). Hardware: one NVIDIA A10G (24GB VRAM). Time budget: up to 8 hours of wall-clock training/eval. External data is NOT allowed; ImageNet-1K and ImageNet-21K pretrained models are allowed (e.g., from timm or HuggingFace). External large-scale corpora (JFT, LAION, LVD-142M, or other proprietary datasets) are NOT allowed. Do NOT use a separate test split for model selection; use only train + val. Submit a single final model; ensembling and TTA are allowed but count against the compute budget. Save the best checkpoint under the experiment directory.
DESC
)

mkdir -p logs
LOG=logs/alchemist_cifar100_$(date +%Y%m%d_%H%M%S).log

python main.py run \
  --task-name cifar100 \
  --task-desc "$TASK_DESC" \
  --data-path /home/ubuntu/data/cifar100 \
  --num-classes 100 \
  --eval-metric top1_accuracy \
  --budget "$BUDGET" \
  --max-trials 12 \
  --max-rounds 3 \
  --llm claude \
  --llm-model sonnet \
  --executor aws \
  --aws-host "$EC2_HOST" \
  --aws-key "$EC2_KEY" \
  --aws-work-dir /home/ubuntu/alchemist \
  --aws-python /opt/pytorch/bin/python \
  2>&1 | tee "$LOG"

echo
echo "=== Alchemist CIFAR-100 done. Log: $LOG ==="
