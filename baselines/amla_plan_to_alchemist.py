#!/usr/bin/env python3
"""Hybrid: AutoML-Agent planning + Alchemist execution.

Pipeline:
  1. Extract the most recent AutoML-Agent plan from /tmp/amla_<dataset>_run.log
     (model, optimizer, augmentation, HPs — agent already did a great job).
  2. Translate it to Alchemist's `nas_worker.py` job JSON schema.
  3. Execute via Alchemist's AWSExecutor on EC2 (proven, stable).
  4. Pull results back and report.

Why this design:
  - AutoML-Agent's PLAN quality is solid (ConvNeXt-Base + LLRD + Mixup/CutMix ...)
  - But its LLM-generated CODE is unstable (CUDA assert fails, path bugs)
  - Alchemist's pre-validated `nas_worker.py` has all the same techniques
    baked in (SAM, EMA, Mixup, CutMix, RandAugment, LLRD, cosine schedule)
  - Decoupling planning from code-gen gives a fair + reliable comparison
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


# ----------------------------------------------------------------------
# Plan extractor (AutoML-Agent log → dict)
# ----------------------------------------------------------------------
def extract_plan(log_path: str) -> dict:
    """Scrape the agent's chosen plan from its run log.

    Looks for the final selected-solution and salient HPs. Returns a dict
    with defaults for anything not found.
    """
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""

    p: dict = {
        # sensible defaults (Alchemist baseline)
        "backbone": "convnext_base.fb_in1k",
        "head_type": "linear",
        "batch_size": 128,
        "epochs": 50,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 2,
        "layer_decay": 0.7,
        "label_smoothing": 0.1,
        "mixup": True,
        "mixup_alpha": 0.8,
        "cutmix": True,
        "cutmix_alpha": 1.0,
        "randaugment": True,
        "randaugment_n": 2,
        "randaugment_m": 9,
        "random_erasing": True,
        "random_erasing_prob": 0.25,
        "ema": True,
        "ema_decay": 0.9999,
        "drop_path_rate": 0.1,
        "img_size": 224,
    }

    def _first_match(pattern, cast=str, default=None):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return default
        try:
            return cast(m.group(1))
        except Exception:
            return default

    # Extract actual plan fields
    backbone = _first_match(r"`(convnext[_a-z0-9.]+)`|`(vit[_a-z0-9.]+)`", str, None)
    if backbone:
        p["backbone"] = backbone if isinstance(backbone, str) else backbone[0] or backbone[1]
    # More robust: look for timm.create_model("...") or similar
    m = re.search(r"convnext_base\.fb_in1k|vit_base_patch16_224\.[a-z_]+|convnext_tiny\.[a-z_]+",
                   text, flags=re.IGNORECASE)
    if m:
        p["backbone"] = m.group(0).lower()

    p["lr"]               = _first_match(r"`?lr[=:]?\s*(\d*\.?\d+e?-?\d*)`?",       float, p["lr"])
    p["weight_decay"]     = _first_match(r"weight[_ ]?decay[= :]+(\d*\.?\d+)",      float, p["weight_decay"])
    p["batch_size"]       = _first_match(r"batch[_ ]?size[= :]+(\d+)",              int,   p["batch_size"])
    p["epochs"]           = _first_match(r"Epochs[^\d]*(\d+)",                       int,   p["epochs"])
    p["warmup_epochs"]    = _first_match(r"warmup[= :]+(\d+)",                      int,   p["warmup_epochs"])
    p["layer_decay"]      = _first_match(r"layer[_ ]?decay[ =:]+(\d*\.?\d+)",        float, p["layer_decay"])
    p["label_smoothing"]  = _first_match(r"label[_ ]?smoothing[^\d]*(\d*\.?\d+)",    float, p["label_smoothing"])
    p["mixup_alpha"]      = _first_match(r"Mixup[^0-9]*(\d*\.?\d+)",                 float, p["mixup_alpha"])
    p["cutmix_alpha"]     = _first_match(r"CutMix[^0-9]*(\d*\.?\d+)",                float, p["cutmix_alpha"])
    p["randaugment_n"]    = _first_match(r"RandAugment[^N]*N[= :]+(\d+)",            int,   p["randaugment_n"])
    p["randaugment_m"]    = _first_match(r"RandAugment[^M]*M[= :]+(\d+)",            int,   p["randaugment_m"])
    p["ema_decay"]        = _first_match(r"EMA[^0-9.]*(\d\.\d{3,5})",                float, p["ema_decay"])

    # Target accuracy (for reporting context)
    m = re.search(r"Target\s*Top-?1[^0-9]*(\d+)[\s-]*(\d+)%", text, flags=re.IGNORECASE)
    if m:
        p["_target_top1_range"] = [int(m.group(1)), int(m.group(2))]

    return p


# ----------------------------------------------------------------------
# Translator — Plan dict → Alchemist nas_worker.py job JSON
# ----------------------------------------------------------------------
def plan_to_alchemist_job(plan: dict, dataset: str, data_path: str,
                          num_classes: int, trial_id: int = 0) -> dict:
    """Convert AutoML-Agent plan into the job schema accepted by nas_worker.py.

    Schema derived from `alchemist/nas_worker.py` get/get calls.
    """
    job = {
        "trial_id": trial_id,
        "task": {
            "name": dataset,
            "data_path": data_path,
            "num_classes": num_classes,
            "eval_metric": "top1_accuracy",
        },
        "arch": {
            "backbone": plan["backbone"],
            "head_type": plan.get("head_type", "linear"),
            "head_hidden": plan.get("head_hidden", 512),
            "head_dropout": plan.get("head_dropout", 0.3),
            "drop_path_rate": plan.get("drop_path_rate", 0.1),
            "add_se": plan.get("add_se", False),
            "add_cbam": plan.get("add_cbam", False),
        },
        "config": {
            # schedule
            "epochs": plan["epochs"],
            "warmup_epochs": plan["warmup_epochs"],
            "lr": plan["lr"],
            "weight_decay": plan["weight_decay"],
            # batch / size
            "batch_size": plan["batch_size"],
            "img_size": plan.get("img_size", 224),
            # aug
            "randaugment": plan["randaugment"],
            "random_erasing": plan["random_erasing"],
            "mixup": plan["mixup"],
            "mixup_alpha": plan["mixup_alpha"],
            "cutmix": plan["cutmix"],
            "cutmix_alpha": plan["cutmix_alpha"],
            "label_smoothing": plan["label_smoothing"],
            # regularisation
            "ema": plan["ema"],
            "ema_decay": plan["ema_decay"],
            # LLRD (Alchemist uses backbone_lr_scale as coarse equivalent)
            "backbone_lr_scale": plan.get("layer_decay", 0.7),
        },
    }
    return job


# ----------------------------------------------------------------------
# EC2 execution via Alchemist pattern
# ----------------------------------------------------------------------
EC2_KEY = Path.home() / ".ssh/alchemist-gpu-key-use1.pem"
EC2_USER = "ubuntu"
EC2_REMOTE_DIR = "/home/ubuntu/alchemist"


def run_on_ec2(job_dict: dict, ec2_host: str, out_name: str,
               repo_root: Path) -> tuple[int, dict]:
    """Upload nas_worker.py + job JSON to EC2, run, pull result."""
    # 1. Ensure nas_worker.py is on EC2
    for f in ("nas_worker.py",):
        subprocess.run(
            ["scp", "-i", str(EC2_KEY), "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             str(repo_root / f), f"{EC2_USER}@{ec2_host}:{EC2_REMOTE_DIR}/{f}"],
            check=True, capture_output=True,
        )

    # 2. Upload job JSON
    job_local = repo_root / f"jobs/{out_name}.json"
    job_local.parent.mkdir(parents=True, exist_ok=True)
    job_local.write_text(json.dumps(job_dict, indent=2))
    subprocess.run(
        ["ssh", "-i", str(EC2_KEY), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         f"{EC2_USER}@{ec2_host}", f"mkdir -p {EC2_REMOTE_DIR}/jobs {EC2_REMOTE_DIR}/results"],
        check=True,
    )
    subprocess.run(
        ["scp", "-i", str(EC2_KEY), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         str(job_local), f"{EC2_USER}@{ec2_host}:{EC2_REMOTE_DIR}/jobs/{out_name}.json"],
        check=True, capture_output=True,
    )

    # 3. Execute remote (foreground with generous timeout)
    remote_cmd = (
        "source /opt/pytorch/bin/activate && "
        f"cd {EC2_REMOTE_DIR} && "
        f"python -u nas_worker.py "
        f"--job jobs/{out_name}.json "
        f"--output results/{out_name}.json 2>&1"
    )
    print(f"[run_on_ec2] launching on {ec2_host} …")
    t0 = time.time()
    proc = subprocess.run(
        ["ssh", "-i", str(EC2_KEY), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         "-o", "ServerAliveInterval=60",
         f"{EC2_USER}@{ec2_host}", remote_cmd],
        capture_output=True, text=True, timeout=6 * 3600,  # 6h hard cap
    )
    elapsed = time.time() - t0
    print(f"[run_on_ec2] rc={proc.returncode}  elapsed={elapsed/60:.1f}min")
    if proc.returncode != 0:
        print(f"[run_on_ec2] stderr tail:\n{proc.stderr[-2000:]}")
        return proc.returncode, {"status": "error", "stderr": proc.stderr[-2000:]}

    # 4. Pull results JSON back
    local_result = repo_root / f"results/{out_name}.json"
    local_result.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-i", str(EC2_KEY), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         f"{EC2_USER}@{ec2_host}:{EC2_REMOTE_DIR}/results/{out_name}.json",
         str(local_result)],
        check=True, capture_output=True,
    )
    return 0, json.loads(local_result.read_text())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
DATASETS = {
    "cifar100":   {"data_path": "/home/ubuntu/data/cifar100",    "num_classes": 100,
                    "log": "/tmp/amla_cifar100_run.log"},
    "butterfly":  {"data_path": "/home/ubuntu/data/butterfly",   "num_classes": 75,
                    "log": "/tmp/amla_butterfly_run.log"},
    "shopee":     {"data_path": "/home/ubuntu/data/shopee-iet",  "num_classes": 4,
                    "log": "/tmp/amla_shopee_run.log"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--ec2-host", default="3.90.166.150")
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--override-backbone", default=None,
                        help="force a specific timm backbone, bypassing plan extraction")
    parser.add_argument("--override-batch-size", type=int, default=None,
                        help="force a specific batch size (e.g. 64 to avoid OOM)")
    parser.add_argument("--override-ema", type=lambda s: s.lower() == "true", default=None,
                        help="force EMA on/off (true/false)")
    parser.add_argument("--override-epochs", type=int, default=None,
                        help="force a specific epoch count (plan extractor may misparse)")
    parser.add_argument("--override-warmup", type=int, default=None,
                        help="force a specific warmup epoch count")
    parser.add_argument("--override-lr", type=float, default=None,
                        help="force a specific base learning rate")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    d = DATASETS[args.dataset]

    print(f"=== Extracting AutoML-Agent plan from {d['log']} ===")
    plan = extract_plan(d["log"])
    if args.override_backbone:
        plan["backbone"] = args.override_backbone
    if args.override_batch_size is not None:
        plan["batch_size"] = args.override_batch_size
    if args.override_ema is not None:
        plan["ema"] = args.override_ema
    if args.override_epochs is not None:
        plan["epochs"] = args.override_epochs
    if args.override_warmup is not None:
        plan["warmup_epochs"] = args.override_warmup
    if args.override_lr is not None:
        plan["lr"] = args.override_lr
    print(json.dumps(plan, indent=2, default=str))

    print(f"\n=== Translating to Alchemist nas_worker job ===")
    job = plan_to_alchemist_job(plan, dataset=args.dataset,
                                 data_path=d["data_path"],
                                 num_classes=d["num_classes"],
                                 trial_id=args.trial_id)
    print(json.dumps(job, indent=2, default=str))

    print(f"\n=== Running on EC2 {args.ec2_host} ===")
    out_name = f"amla_plan_{args.dataset}_t{args.trial_id}"
    rc, result = run_on_ec2(job, args.ec2_host, out_name, repo_root)

    print(f"\n=== Result ===")
    print(json.dumps(result, indent=2, default=str))
    if rc == 0:
        top1 = result.get("top1_accuracy") or result.get("best_top1")
        if top1 is not None:
            print(f"\nFinal Top-1 on {args.dataset}: {top1:.2f}%")
    sys.exit(rc)


if __name__ == "__main__":
    main()
