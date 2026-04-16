#!/bin/bash
# Alchemist on Butterfly Image Classification — fair comparison twin of run_amla_butterfly.py.
#
# Same constraints as AutoML-Agent baseline:
#   - Same GPU: A10G × 1 (via AWSExecutor SSH to EC2)
#   - Same LLM: Claude CLI
#   - Same data: /home/ubuntu/data/butterfly/{train,val,test}
#   - Same time budget: 4 GPU-hours
#   - Same rules: ImageNet-1K pretrained OK; no external data; val-only eval
#   - No hints about specific architectures/techniques
#
# Usage:
#   bash baselines/run_alchemist_butterfly.sh

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EC2_HOST="${EC2_HOST:-ubuntu@3.90.166.150}"
EC2_KEY="${EC2_KEY:-$HOME/.ssh/alchemist-gpu-key-use1.pem}"
BUDGET="${BUDGET:-4.0}"

TASK_DESC=$(cat <<'DESC'
Build a model to classify butterfly species from images. Dataset: Butterfly Image Classification (75 fine-grained butterfly species; color images). Training set: 9,285 labeled images at /home/ubuntu/data/butterfly/train/<class>/*.jpg. Validation set: 375 images at /home/ubuntu/data/butterfly/val/<class>/*.jpg. Test set (NOT to be used for model selection): 375 images at /home/ubuntu/data/butterfly/test/. Evaluation metric: Top-1 accuracy on the validation set (single number, 2 decimal places). Hardware: one NVIDIA A10G (24GB VRAM). Time budget: up to 4 hours of wall-clock training/eval. External data is NOT allowed; ImageNet-1K and ImageNet-21K pretrained models are allowed (e.g., from timm or HuggingFace). External large-scale corpora (JFT, LAION, LVD-142M, or other proprietary datasets) are NOT allowed. Do NOT use the test split for selection or tuning; use only train + val. Submit a single final model; ensembling and TTA are allowed but count against the compute budget.
DESC
)

mkdir -p logs
LOG=logs/alchemist_butterfly_$(date +%Y%m%d_%H%M%S).log

python main.py run \
  --task-name butterfly \
  --task-desc "$TASK_DESC" \
  --data-path /home/ubuntu/data/butterfly \
  --num-classes 75 \
  --eval-metric top1_accuracy \
  --budget "$BUDGET" \
  --max-trials 10 \
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
echo "=== Alchemist Butterfly done. Log: $LOG ==="
