#!/bin/bash
# Alchemist on Shopee-IET — fair comparison twin of run_amla_shopee.py.
#
# Same constraints as AutoML-Agent baseline:
#   - Same GPU: A10G × 1 (via AWSExecutor SSH to EC2)
#   - Same LLM: Claude CLI
#   - Same data: /home/ubuntu/data/shopee-iet/{train,val,test}
#   - Same time budget: 2 GPU-hours
#   - Same rules: ImageNet-1K pretrained OK; no external data; val-only eval
#   - No hints about specific architectures/techniques
#
# Usage:
#   bash baselines/run_alchemist_shopee.sh

set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EC2_HOST="${EC2_HOST:-ubuntu@3.90.166.150}"
EC2_KEY="${EC2_KEY:-$HOME/.ssh/alchemist-gpu-key-use1.pem}"
BUDGET="${BUDGET:-2.0}"

TASK_DESC=$(cat <<'DESC'
Build a model to classify Shopee product images into their category. Dataset: Shopee-IET (4 classes: BabyPants, BabyShirt, womencasualshoes, womenchiffontop). Training set: 640 images (160 per class) at /home/ubuntu/data/shopee-iet/train/<class>/*.jpg. Validation set: 160 images (40 per class) at /home/ubuntu/data/shopee-iet/val/<class>/*.jpg. Additional held-out test set (NOT to be used for selection): 80 images at /home/ubuntu/data/shopee-iet/test/. Image sizes vary (not pre-resized). Evaluation metric: Top-1 accuracy on the validation set (single number, 2 decimal places). Hardware: one NVIDIA A10G (24GB VRAM). Time budget: up to 2 hours of wall-clock training/eval. External data is NOT allowed; ImageNet-1K and ImageNet-21K pretrained models are allowed (e.g., from timm or HuggingFace). External large-scale corpora (JFT, LAION, LVD-142M, or other proprietary datasets) are NOT allowed. Do NOT use the test split for selection or tuning; use only train + val. Submit a single final model; ensembling and TTA are allowed but count against the compute budget.
DESC
)

mkdir -p logs
LOG=logs/alchemist_shopee_$(date +%Y%m%d_%H%M%S).log

python main.py run \
  --task-name shopee_iet \
  --task-desc "$TASK_DESC" \
  --data-path /home/ubuntu/data/shopee-iet \
  --num-classes 4 \
  --eval-metric top1_accuracy \
  --budget "$BUDGET" \
  --max-trials 8 \
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
echo "=== Alchemist Shopee-IET done. Log: $LOG ==="
