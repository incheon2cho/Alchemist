#!/bin/bash
# Linear Probe Batch — frozen backbone + linear head only
# Tests representation quality across all backbones

source /opt/pytorch/bin/activate
cd /home/ubuntu/alchemist
mkdir -p jobs results

TASK='{"name":"cifar100","data_path":"/home/ubuntu/data/cifar100","num_classes":100,"eval_metric":"top1_accuracy"}'

echo "=== Linear Probe Batch Start: $(date) ===" > lp_log.txt

# All backbones to evaluate with linear probe
BACKBONES=(
  "1:resnet50:resnet50.a1_in1k"
  "2:resnet101:resnet101.a1_in1k"
  "3:wide_resnet50:wide_resnet50_2.tv2_in1k"
  "4:resnext50:resnext50_32x4d.a1h_in1k"
  "5:efficientnet_b2:efficientnet_b2.ra_in1k"
  "6:convnext_tiny:convnext_tiny.fb_in1k"
  "7:regnety_032:regnety_032.tv2_in1k"
  "8:convnext_small:convnext_small.fb_in1k"
  "9:densenet121:densenet121.tv_in1k"
  "10:mobilenetv3:mobilenetv3_large_100.ra_in1k"
)

# Phase 1: Linear probe (frozen backbone, linear head, 50ep)
echo "=== PHASE 1: Linear Probe (linear head) ===" >> lp_log.txt

for entry in "${BACKBONES[@]}"; do
  IFS=':' read -r tid name model_id <<< "$entry"

  cat > jobs/lp_t${tid}.json << JOBEOF
{
  "trial_id": ${tid},
  "task": $TASK,
  "arch": {
    "backbone": "${model_id}",
    "head_type": "linear",
    "head_dropout": 0.0
  },
  "config": {
    "lr": 0.01, "epochs": 50, "batch_size": 256,
    "weight_decay": 0.0, "label_smoothing": 0.0,
    "lr_schedule": "cosine",
    "backbone_lr_scale": 0.0,
    "mixup": false, "cutmix": false,
    "randaugment": false, "random_erasing": false
  }
}
JOBEOF

  echo "--- LP Trial $tid ($name) start: $(date) ---" >> lp_log.txt
  python3 nas_worker.py --job jobs/lp_t${tid}.json --output results/lp_t${tid}.json >> lp_log.txt 2>&1
  SCORE=$(python3 -c "import json; d=json.load(open('results/lp_t${tid}.json')); print(d.get('score','ERR'))" 2>/dev/null)
  echo ">>> LP Trial $tid ($name): ${SCORE}% - $(date)" >> lp_log.txt
  echo "LP_T${tid}|${name}|linear|${SCORE}" >> results/lp_summary.txt
done

echo "" >> lp_log.txt
echo "=== PHASE 1 COMPLETE ===" >> lp_log.txt

# Phase 2: MLP probe on top 5 (frozen backbone, MLP head, 50ep)
echo "=== PHASE 2: MLP Probe (top 5 backbones) ===" >> lp_log.txt

TOP5=$(python3 << 'PYEOF'
import json, glob
results = []
for f in sorted(glob.glob("results/lp_t*.json")):
    d = json.load(open(f))
    if d.get("status") == "ok":
        bb = d["arch"]["backbone"]
        results.append((d["score"], bb))
results.sort(reverse=True)
for s, bb in results[:5]:
    print(f"{bb}:{s}")
PYEOF
)

MLP_TID=20
for line in $TOP5; do
  IFS=':' read -r model_id base_score <<< "$line"
  MLP_TID=$((MLP_TID + 1))

  # Extract short name
  SHORT=$(python3 -c "print('${model_id}'.split('.')[0])")

  cat > jobs/lp_mlp_t${MLP_TID}.json << JOBEOF
{
  "trial_id": ${MLP_TID},
  "task": $TASK,
  "arch": {
    "backbone": "${model_id}",
    "head_type": "mlp",
    "head_hidden": 512,
    "head_dropout": 0.3
  },
  "config": {
    "lr": 0.001, "epochs": 50, "batch_size": 256,
    "weight_decay": 0.01, "label_smoothing": 0.1,
    "lr_schedule": "cosine",
    "backbone_lr_scale": 0.0,
    "mixup": false, "cutmix": false,
    "randaugment": false, "random_erasing": false
  }
}
JOBEOF

  echo "--- MLP Probe $MLP_TID ($SHORT) start: $(date) ---" >> lp_log.txt
  python3 nas_worker.py --job jobs/lp_mlp_t${MLP_TID}.json --output results/lp_mlp_t${MLP_TID}.json >> lp_log.txt 2>&1
  SCORE=$(python3 -c "import json; d=json.load(open('results/lp_mlp_t${MLP_TID}.json')); print(d.get('score','ERR'))" 2>/dev/null)
  echo ">>> MLP Probe $MLP_TID ($SHORT): ${SCORE}% - $(date)" >> lp_log.txt
  echo "MLP_T${MLP_TID}|${SHORT}|mlp|${SCORE}" >> results/lp_summary.txt
done

echo "" >> lp_log.txt
echo "=== PHASE 2 COMPLETE ===" >> lp_log.txt

# Summary
echo "" >> lp_log.txt
echo "=== FINAL SUMMARY ===" >> lp_log.txt
python3 << 'PYEOF' >> lp_log.txt
import json, glob
results = []
for f in sorted(glob.glob("results/lp_*.json")):
    try:
        d = json.load(open(f))
        if d.get("status") == "ok":
            bb = d["arch"]["backbone"].split(".")[0]
            head = d["arch"].get("head_type","linear")
            results.append((d["score"], bb, head, d.get("params_m",0), d.get("elapsed_s",0), f))
    except: pass
results.sort(reverse=True)
print(f"\nTotal trials: {len(results)}")
print(f"{'Rank':<5} {'Score':<8} {'Backbone':<22} {'Head':<8} {'Params':<8} {'Time':<8}")
print("-" * 70)
for i, (s, bb, h, p, t, f) in enumerate(results[:15], 1):
    print(f"{i:<5} {s:<8.2f} {bb:<22} {h:<8} {p:<8.1f}M {t/60:<8.1f}m")
if results:
    print(f"\nBEST LINEAR PROBE: {results[0][0]:.2f}% ({results[0][1]}, {results[0][2]})")
PYEOF

echo "=== LP Batch Complete: $(date) ===" >> lp_log.txt
echo "ALL_DONE" >> results/lp_summary.txt
