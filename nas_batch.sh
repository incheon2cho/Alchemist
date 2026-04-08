#!/bin/bash
# NAS Batch Runner — 8-hour CIFAR-100 Architecture Search
# Phase 1: Architecture exploration (short training, many backbones)
# Phase 2: Top backbone HP optimization (medium training)
# Phase 3: Final long training with best config

source /opt/pytorch/bin/activate
cd /home/ubuntu/alchemist
mkdir -p jobs results

TASK='{"name":"cifar100","data_path":"/home/ubuntu/data/cifar100","num_classes":100,"eval_metric":"top1_accuracy"}'
START_TIME=$(date +%s)
MAX_SECONDS=28800  # 8 hours

echo "=== NAS Batch Start: $(date) ===" > nas_log.txt
echo "Budget: 8 hours" >> nas_log.txt

# ============================================================
# Phase 1: Architecture Exploration (20ep each, ~15min each)
# ============================================================
echo "" >> nas_log.txt
echo "=== PHASE 1: Architecture Exploration ===" >> nas_log.txt

PHASE1_CONFIGS=(
  # trial:backbone:head:batch_size:lr
  "1:resnet50:mlp:128:0.001"
  "2:resnet101:mlp:64:0.0008"
  "3:wide_resnet50:mlp:64:0.0008"
  "4:resnext50:mlp:128:0.001"
  "5:efficientnet_b2:mlp:128:0.001"
  "6:convnext_tiny:mlp:128:0.0005"
  "7:regnety_032:mlp:128:0.001"
)

for entry in "${PHASE1_CONFIGS[@]}"; do
  IFS=':' read -r tid backbone head bs lr <<< "$entry"

  ELAPSED=$(( $(date +%s) - START_TIME ))
  if [ $ELAPSED -gt $MAX_SECONDS ]; then
    echo ">>> TIME LIMIT REACHED at Phase 1 trial $tid" >> nas_log.txt
    break
  fi

  cat > jobs/p1_t${tid}.json << JOBEOF
{
  "trial_id": ${tid},
  "task": $TASK,
  "arch": {
    "backbone": "${backbone}",
    "head_type": "${head}",
    "head_hidden": 512,
    "head_dropout": 0.3,
    "drop_path_rate": 0.0
  },
  "config": {
    "lr": ${lr}, "epochs": 20, "batch_size": ${bs},
    "weight_decay": 0.05, "label_smoothing": 0.1,
    "lr_schedule": "onecycle", "backbone_lr_scale": 0.1,
    "mixup": true, "mixup_alpha": 0.2,
    "cutmix": true, "cutmix_alpha": 1.0,
    "randaugment": true, "random_erasing": true
  }
}
JOBEOF

  echo "--- P1 Trial $tid ($backbone) start: $(date) ---" >> nas_log.txt
  python3 nas_worker.py --job jobs/p1_t${tid}.json --output results/p1_t${tid}.json >> nas_log.txt 2>&1
  SCORE=$(python3 -c "import json; d=json.load(open('results/p1_t${tid}.json')); print(d.get('score','ERR'))" 2>/dev/null)
  echo ">>> P1 Trial $tid ($backbone): ${SCORE}% - $(date)" >> nas_log.txt
  echo "P1_T${tid}|${backbone}|${SCORE}" >> results/nas_summary.txt
done

echo "" >> nas_log.txt
echo "=== PHASE 1 COMPLETE ===" >> nas_log.txt

# Find top 3 backbones from Phase 1
echo "Selecting top 3 backbones..." >> nas_log.txt
TOP3=$(python3 << 'PYEOF'
import json, glob
results = []
for f in sorted(glob.glob("results/p1_t*.json")):
    d = json.load(open(f))
    if d.get("status") == "ok":
        results.append((d["score"], d["arch"]["backbone"], f))
results.sort(reverse=True)
for score, bb, f in results[:3]:
    print(f"{bb}:{score}")
PYEOF
)
echo "Top 3: $TOP3" >> nas_log.txt

# ============================================================
# Phase 2: HP Optimization on Top Backbones (40ep, varied configs)
# ============================================================
echo "" >> nas_log.txt
echo "=== PHASE 2: HP Optimization ===" >> nas_log.txt

P2_TID=10
for line in $TOP3; do
  IFS=':' read -r backbone base_score <<< "$line"

  # Config variations for each top backbone
  P2_VARIATIONS=(
    # lr:epochs:batch:head:se:cbam:head_hidden:dropout:lr_schedule
    "0.001:40:128:mlp:false:false:512:0.3:onecycle"
    "0.0005:40:64:mlp_deep:false:false:1024:0.4:onecycle"
    "0.001:40:128:mlp:true:false:512:0.3:onecycle"
    "0.0008:40:128:mlp:false:true:512:0.3:cosine"
  )

  for variation in "${P2_VARIATIONS[@]}"; do
    IFS=':' read -r vlr vep vbs vhead vse vcbam vhidden vdrop vsched <<< "$variation"
    P2_TID=$((P2_TID + 1))

    ELAPSED=$(( $(date +%s) - START_TIME ))
    REMAINING=$((MAX_SECONDS - ELAPSED))
    if [ $REMAINING -lt 1800 ]; then  # need at least 30min for Phase 3
      echo ">>> TIME: Skipping remaining P2 trials, saving time for Phase 3" >> nas_log.txt
      break 2
    fi

    cat > jobs/p2_t${P2_TID}.json << JOBEOF
{
  "trial_id": ${P2_TID},
  "task": $TASK,
  "arch": {
    "backbone": "${backbone}",
    "head_type": "${vhead}",
    "head_hidden": ${vhidden},
    "head_dropout": ${vdrop},
    "add_se": ${vse},
    "add_cbam": ${vcbam},
    "drop_path_rate": 0.1
  },
  "config": {
    "lr": ${vlr}, "epochs": ${vep}, "batch_size": ${vbs},
    "weight_decay": 0.05, "label_smoothing": 0.1,
    "lr_schedule": "${vsched}", "backbone_lr_scale": 0.1,
    "warmup_epochs": 5,
    "mixup": true, "mixup_alpha": 0.3,
    "cutmix": true, "cutmix_alpha": 1.0,
    "randaugment": true, "random_erasing": true
  }
}
JOBEOF

    echo "--- P2 Trial $P2_TID ($backbone, lr=$vlr, head=$vhead, SE=$vse, CBAM=$vcbam) start: $(date) ---" >> nas_log.txt
    python3 nas_worker.py --job jobs/p2_t${P2_TID}.json --output results/p2_t${P2_TID}.json >> nas_log.txt 2>&1
    SCORE=$(python3 -c "import json; d=json.load(open('results/p2_t${P2_TID}.json')); print(d.get('score','ERR'))" 2>/dev/null)
    echo ">>> P2 Trial $P2_TID ($backbone, $vhead, SE=$vse): ${SCORE}% - $(date)" >> nas_log.txt
    echo "P2_T${P2_TID}|${backbone}|${vhead}|SE=${vse}|CBAM=${vcbam}|${SCORE}" >> results/nas_summary.txt
  done
done

echo "" >> nas_log.txt
echo "=== PHASE 2 COMPLETE ===" >> nas_log.txt

# ============================================================
# Phase 3: Final Long Training with Best Config
# ============================================================
echo "" >> nas_log.txt
echo "=== PHASE 3: Final Long Training ===" >> nas_log.txt

# Find overall best config
python3 << 'PYEOF' > /tmp/best_config.json
import json, glob
best_score = 0
best_file = None
for f in sorted(glob.glob("results/p*.json")):
    try:
        d = json.load(open(f))
        if d.get("status") == "ok" and d.get("score", 0) > best_score:
            best_score = d["score"]
            best_file = f
    except: pass

if best_file:
    d = json.load(open(best_file))
    # Create extended config for long training
    final = {
        "trial_id": 100,
        "task": d["task"] if "task" in d else {"name":"cifar100","data_path":"/home/ubuntu/data/cifar100","num_classes":100},
        "arch": d["arch"],
        "config": d["config"]
    }
    final["config"]["epochs"] = 100
    final["config"]["ema"] = True
    final["config"]["ema_decay"] = 0.9995
    final["trial_id"] = 100
    print(json.dumps(final, indent=2))
    import sys
    print(f"Best so far: {best_score}% from {best_file}", file=sys.stderr)
else:
    print("{}", file=sys.stderr)
PYEOF

ELAPSED=$(( $(date +%s) - START_TIME ))
REMAINING=$((MAX_SECONDS - ELAPSED))

if [ $REMAINING -gt 600 ] && [ -s /tmp/best_config.json ]; then
  cp /tmp/best_config.json jobs/p3_final.json
  echo "--- P3 Final training start: $(date), remaining: ${REMAINING}s ---" >> nas_log.txt

  # Adjust epochs based on remaining time (rough: ~90s per epoch for ResNet-50)
  AVAIL_EPOCHS=$((REMAINING / 100))
  if [ $AVAIL_EPOCHS -gt 100 ]; then AVAIL_EPOCHS=100; fi
  if [ $AVAIL_EPOCHS -lt 30 ]; then AVAIL_EPOCHS=30; fi

  python3 -c "
import json
d = json.load(open('jobs/p3_final.json'))
d['config']['epochs'] = $AVAIL_EPOCHS
d['trial_id'] = 100
json.dump(d, open('jobs/p3_final.json','w'), indent=2)
print(f'Final: {d[\"arch\"][\"backbone\"]}, {$AVAIL_EPOCHS} epochs')
" >> nas_log.txt

  python3 nas_worker.py --job jobs/p3_final.json --output results/p3_final.json >> nas_log.txt 2>&1
  SCORE=$(python3 -c "import json; d=json.load(open('results/p3_final.json')); print(d.get('score','ERR'))" 2>/dev/null)
  echo ">>> P3 FINAL: ${SCORE}% - $(date)" >> nas_log.txt
  echo "P3_FINAL|${SCORE}" >> results/nas_summary.txt
fi

# ============================================================
# Summary
# ============================================================
echo "" >> nas_log.txt
echo "=== FINAL SUMMARY ===" >> nas_log.txt
python3 << 'PYEOF' >> nas_log.txt
import json, glob
results = []
for f in sorted(glob.glob("results/p*.json")):
    try:
        d = json.load(open(f))
        if d.get("status") == "ok":
            bb = d.get("arch",{}).get("backbone","?")
            head = d.get("arch",{}).get("head_type","?")
            results.append((d["score"], bb, head, d.get("elapsed_s",0), f))
    except: pass

results.sort(reverse=True)
print(f"\nTotal trials: {len(results)}")
print(f"{'Rank':<5} {'Score':<8} {'Backbone':<20} {'Head':<10} {'Time':<8} {'File'}")
print("-" * 80)
for i, (score, bb, head, t, f) in enumerate(results[:10], 1):
    print(f"{i:<5} {score:<8.2f} {bb:<20} {head:<10} {t/60:<8.1f}m {f}")
if results:
    print(f"\nBEST: {results[0][0]:.2f}% ({results[0][1]}, {results[0][2]})")
PYEOF

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
echo "Total time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m" >> nas_log.txt
echo "=== NAS Batch Complete: $(date) ===" >> nas_log.txt
echo "ALL_DONE" >> results/nas_summary.txt
