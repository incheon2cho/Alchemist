#!/bin/bash
# ViT / Swin / Mamba — Fine-tuning + Linear Probe (v2, no timeout)
# Epochs adjusted for 30min per model on A10G

source /opt/pytorch/bin/activate
cd /home/ubuntu/alchemist
mkdir -p jobs results
rm -f results/vit_summary.txt

TASK='{"name":"cifar100","data_path":"/home/ubuntu/data/cifar100","num_classes":100,"eval_metric":"top1_accuracy"}'

echo "=== ViT/Swin/Mamba Batch v2 Start: $(date) ===" > vit_log.txt

# name:timm_id:bs_ft:lr:epochs_ft:bs_lp
MODELS=(
  "vit_tiny:vit_tiny_patch16_224.augreg_in21k_ft_in1k:128:0.0005:15:256"
  "vit_small:vit_small_patch16_224.augreg_in21k_ft_in1k:64:0.0005:15:256"
  "vit_base:vit_base_patch16_224.augreg2_in21k_ft_in1k:32:0.0003:10:128"
  "deit_small:deit_small_patch16_224.fb_in1k:64:0.0005:15:256"
  "deit_base:deit_base_patch16_224.fb_in1k:32:0.0003:10:128"
  "swin_tiny:swin_tiny_patch4_window7_224.ms_in1k:64:0.0005:15:256"
  "swin_small:swin_small_patch4_window7_224.ms_in1k:32:0.0003:12:128"
  "swin_base:swin_base_patch4_window7_224.ms_in22k_ft_in1k:32:0.0002:10:64"
  "mambaout_tiny:mambaout_tiny:128:0.0005:20:256"
  "mambaout_small:mambaout_small:64:0.0003:15:256"
)

TID=0
for entry in "${MODELS[@]}"; do
  IFS=':' read -r name model_id bs_ft lr epochs_ft bs_lp <<< "$entry"
  TID=$((TID + 1))

  echo "" >> vit_log.txt
  echo "=== Model $TID: $name ===" >> vit_log.txt

  # --- Fine-tuning ---
  cat > jobs/vit_ft_${TID}.json << JOBEOF
{
  "trial_id": ${TID},
  "task": $TASK,
  "arch": {
    "backbone": "${model_id}",
    "head_type": "mlp",
    "head_hidden": 512,
    "head_dropout": 0.3,
    "drop_path_rate": 0.1
  },
  "config": {
    "lr": ${lr}, "epochs": ${epochs_ft}, "batch_size": ${bs_ft},
    "weight_decay": 0.05, "label_smoothing": 0.1,
    "lr_schedule": "onecycle", "backbone_lr_scale": 0.1,
    "mixup": true, "mixup_alpha": 0.2,
    "cutmix": true, "cutmix_alpha": 1.0,
    "randaugment": true, "random_erasing": true
  }
}
JOBEOF

  echo "--- FT $name start: $(date) ---" >> vit_log.txt
  python3 nas_worker.py --job jobs/vit_ft_${TID}.json --output results/vit_ft_${TID}.json >> vit_log.txt 2>&1
  FT_SCORE=$(python3 -c "import json; d=json.load(open('results/vit_ft_${TID}.json')); print(d.get('score','ERR'))" 2>/dev/null)
  echo ">>> FT $name: ${FT_SCORE}% - $(date)" >> vit_log.txt

  # --- Linear Probe ---
  cat > jobs/vit_lp_${TID}.json << JOBEOF
{
  "trial_id": $((TID + 100)),
  "task": $TASK,
  "arch": {
    "backbone": "${model_id}",
    "head_type": "linear",
    "head_dropout": 0.0
  },
  "config": {
    "lr": 0.01, "epochs": 50, "batch_size": ${bs_lp},
    "weight_decay": 0.0, "label_smoothing": 0.0,
    "lr_schedule": "cosine", "backbone_lr_scale": 0.0,
    "mixup": false, "cutmix": false,
    "randaugment": false, "random_erasing": false
  }
}
JOBEOF

  echo "--- LP $name start: $(date) ---" >> vit_log.txt
  python3 nas_worker.py --job jobs/vit_lp_${TID}.json --output results/vit_lp_${TID}.json >> vit_log.txt 2>&1
  LP_SCORE=$(python3 -c "import json; d=json.load(open('results/vit_lp_${TID}.json')); print(d.get('score','ERR'))" 2>/dev/null)
  echo ">>> LP $name: ${LP_SCORE}% - $(date)" >> vit_log.txt

  echo "${name}|FT|${FT_SCORE}|LP|${LP_SCORE}" >> results/vit_summary.txt
done

# --- Vision Mamba (scratch, no pretrained) ---
echo "" >> vit_log.txt
echo "=== Vision Mamba (scratch) ===" >> vit_log.txt
TID=11

cat > jobs/vit_ft_${TID}.json << JOBEOF
{
  "trial_id": ${TID},
  "task": $TASK,
  "arch": {
    "backbone": "vim_small",
    "head_type": "linear",
    "head_dropout": 0.0
  },
  "config": {
    "lr": 0.001, "epochs": 30, "batch_size": 128,
    "weight_decay": 0.05, "label_smoothing": 0.1,
    "lr_schedule": "onecycle", "backbone_lr_scale": 1.0,
    "mixup": true, "mixup_alpha": 0.2,
    "cutmix": true, "cutmix_alpha": 1.0,
    "randaugment": true, "random_erasing": true
  }
}
JOBEOF

echo "--- FT vim_small (scratch) start: $(date) ---" >> vit_log.txt
python3 nas_worker.py --job jobs/vit_ft_${TID}.json --output results/vit_ft_${TID}.json >> vit_log.txt 2>&1
VIM_SCORE=$(python3 -c "import json; d=json.load(open('results/vit_ft_${TID}.json')); print(d.get('score','ERR'))" 2>/dev/null)
echo ">>> FT vim_small (scratch): ${VIM_SCORE}% - $(date)" >> vit_log.txt
echo "vim_small|FT|${VIM_SCORE}|LP|N/A" >> results/vit_summary.txt

# --- Summary ---
echo "" >> vit_log.txt
echo "=== FINAL SUMMARY ===" >> vit_log.txt
python3 << 'PYEOF' >> vit_log.txt
import json, glob
ft_results, lp_results = [], []
for f in sorted(glob.glob("results/vit_ft_*.json")):
    try:
        d = json.load(open(f))
        if d.get("status") == "ok":
            bb = d["arch"]["backbone"].split(".")[0]
            ft_results.append((d["score"], bb, d.get("elapsed_s",0)))
    except: pass
for f in sorted(glob.glob("results/vit_lp_*.json")):
    try:
        d = json.load(open(f))
        if d.get("status") == "ok":
            bb = d["arch"]["backbone"].split(".")[0]
            lp_results.append((d["score"], bb, d.get("elapsed_s",0)))
    except: pass
print("\n=== FINE-TUNING ===")
for s, bb, t in sorted(ft_results, reverse=True):
    print(f"  {s:.2f}%  {bb:<35} {t/60:.1f}m")
print("\n=== LINEAR PROBE ===")
for s, bb, t in sorted(lp_results, reverse=True):
    print(f"  {s:.2f}%  {bb:<35} {t/60:.1f}m")
if ft_results: print(f"\nBEST FT: {max(ft_results)[0]:.2f}%")
if lp_results: print(f"BEST LP: {max(lp_results)[0]:.2f}%")
PYEOF

echo "=== Batch Complete: $(date) ===" >> vit_log.txt
echo "ALL_DONE" >> results/vit_summary.txt
