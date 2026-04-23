"""Alchemist Detection Worker — COCO Object Detection training + evaluation.

Supports YOLO, RT-DETR, and torchvision detection models.
Uses ultralytics for YOLO/RT-DETR, torchvision for Faster-RCNN/DETR.

Usage:
    python detection_worker.py --job jobs/det_trial.json \
        --output jobs/det_result.json --progress jobs/det_progress.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detection_worker")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_detection_training(
    base_model: str,
    task: dict,
    config: dict,
    trial_id: int,
) -> dict:
    """Train an object detection model on COCO format data."""
    t0 = time.time()

    # Config
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 16)
    img_size = config.get("img_size", 640)
    lr = config.get("lr", 0.01)
    optimizer = config.get("optimizer", "auto")
    freeze_backbone = config.get("freeze_backbone", False)
    freeze_layers = config.get("freeze_layers", 0)
    augmentation = config.get("augmentation", "basic")
    device = config.get("device", "0")
    workers = config.get("workers", 4)

    data_path = task.get("data_path", "")  # COCO yaml or directory
    eval_metric = task.get("eval_metric", "mAP50-95")

    logger.info("=== Detection Training ===")
    logger.info("  Model: %s", base_model)
    logger.info("  Data: %s", data_path)
    logger.info("  Epochs: %d, Batch: %d, ImgSize: %d", epochs, batch_size, img_size)

    # Determine framework: ultralytics (YOLO/RT-DETR) or torchvision
    model_lower = base_model.lower()
    use_ultralytics = any(k in model_lower for k in ["yolo", "rtdetr", "rt-detr"])

    if use_ultralytics:
        return _train_ultralytics(
            base_model, data_path, config, trial_id, t0,
        )
    else:
        return _train_torchvision(
            base_model, data_path, config, trial_id, t0,
        )


def _train_ultralytics(
    model_name: str,
    data_path: str,
    config: dict,
    trial_id: int,
    t0: float,
) -> dict:
    """Train using ultralytics (YOLO, RT-DETR)."""
    from ultralytics import YOLO

    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 16)
    img_size = config.get("img_size", 640)
    lr = config.get("lr", 0.01)
    optimizer = config.get("optimizer", "auto")
    freeze_layers = config.get("freeze_layers", 0)
    device = config.get("device", "0")
    workers = config.get("workers", 4)
    patience = config.get("patience", 10)

    # Model mapping
    model_map = {
        "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt", "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",
        "yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt",
        "yolo11m": "yolo11m.pt", "yolo11l": "yolo11l.pt",
        "yolo11x": "yolo11x.pt",
        "rtdetr-l": "rtdetr-l.pt", "rtdetr-x": "rtdetr-x.pt",
    }
    model_file = model_map.get(model_name.lower(), model_name)

    logger.info("  Loading ultralytics model: %s", model_file)
    model = YOLO(model_file)

    # Build training args
    train_args = {
        "data": data_path,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "lr0": lr,
        "optimizer": optimizer,
        "device": device,
        "workers": workers,
        "patience": patience,
        "project": f"/home/ubuntu/checkpoints/detection",
        "name": f"trial{trial_id}",
        "exist_ok": True,
        "verbose": True,
    }

    if freeze_layers > 0:
        train_args["freeze"] = freeze_layers

    # Augmentation
    aug = config.get("augmentation", "basic")
    if aug == "advanced":
        train_args.update({
            "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.1,
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "degrees": 10.0, "translate": 0.1, "scale": 0.5,
            "fliplr": 0.5, "flipud": 0.0,
        })
    elif aug == "minimal":
        train_args.update({
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        })

    # SAM optimizer
    if config.get("use_sam", False):
        logger.info("  Note: SAM not natively supported in ultralytics, using default optimizer")

    # Progress callback
    _prog_path = os.environ.get("ALCHEMIST_PROGRESS_PATH")

    logger.info("  Starting ultralytics training...")
    try:
        results = model.train(**train_args)
    except Exception as e:
        logger.error("  Training failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "trial_id": trial_id,
        }

    # Extract metrics
    metrics = results.results_dict if hasattr(results, "results_dict") else {}

    # mAP values
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    precision = metrics.get("metrics/precision(B)", 0.0)
    recall = metrics.get("metrics/recall(B)", 0.0)

    # Validate on val set
    logger.info("  Running validation...")
    try:
        val_results = model.val(data=data_path, imgsz=img_size, device=device)
        map50 = val_results.box.map50
        map50_95 = val_results.box.map
        precision = val_results.box.mp
        recall = val_results.box.mr
    except Exception as e:
        logger.warning("  Validation failed: %s", e)

    elapsed = time.time() - t0

    # S3 backup
    s3_bucket = config.get("s3_backup_bucket", "")
    if s3_bucket:
        try:
            import subprocess
            ckpt_dir = f"/home/ubuntu/checkpoints/detection/trial{trial_id}"
            s3_prefix = f"det_trial{trial_id}"
            subprocess.run(
                ["aws", "s3", "sync", ckpt_dir, f"s3://{s3_bucket}/{s3_prefix}/"],
                timeout=120, capture_output=True,
            )
            logger.info("  [s3-backup] → s3://%s/%s/", s3_bucket, s3_prefix)
        except Exception as e:
            logger.warning("  [s3-backup] Failed: %s", e)

    logger.info("  Results: mAP50=%.4f, mAP50-95=%.4f, P=%.4f, R=%.4f",
                map50, map50_95, precision, recall)

    return {
        "status": "ok",
        "trial_id": trial_id,
        "score": round(float(map50_95) * 100, 2),
        "map50": round(float(map50) * 100, 2),
        "map50_95": round(float(map50_95) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "train_loss": 0.0,
        "elapsed_s": round(elapsed, 1),
        "config": config,
        "applied_techniques": {
            "model": model_name,
            "framework": "ultralytics",
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "lr": lr,
            "optimizer": optimizer,
            "augmentation": aug,
            "freeze_layers": freeze_layers,
        },
    }


def _train_torchvision(
    model_name: str,
    data_path: str,
    config: dict,
    trial_id: int,
    t0: float,
) -> dict:
    """Train using torchvision detection models (Faster-RCNN, DETR, etc.)."""
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        fasterrcnn_mobilenet_v3_large_fpn,
    )

    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 8)
    lr = config.get("lr", 0.005)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("  Loading torchvision model: %s", model_name)
    num_classes = config.get("num_classes", 91)  # COCO has 80 + background

    if "mobilenet" in model_name.lower():
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Replace head for custom num_classes if needed
    if num_classes != 91:
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    logger.info("  torchvision detection training not fully implemented yet")
    logger.info("  Use ultralytics (YOLO/RT-DETR) for best results")

    return {
        "status": "error",
        "error": "torchvision detection training not fully implemented. Use YOLO/RT-DETR.",
        "trial_id": trial_id,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Alchemist Detection Worker")
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", help="Path to write per-epoch progress JSON")
    args = parser.parse_args()

    if args.progress:
        os.environ["ALCHEMIST_PROGRESS_PATH"] = args.progress

    with open(args.job) as f:
        job = json.load(f)

    logger.info("Detection Job: model=%s, task=%s",
                job.get("base_model"), job.get("task", {}).get("name"))

    try:
        result = run_detection_training(
            base_model=job.get("base_model", "yolov8m"),
            task=job.get("task", {}),
            config=job.get("config", {}),
            trial_id=job.get("trial_id", 0),
        )
    except Exception as e:
        logger.error("Failed: %s", e, exc_info=True)
        result = {
            "status": "error",
            "error": str(e),
            "trial_id": job.get("trial_id", 0),
        }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Result: status=%s, mAP50-95=%s",
                result.get("status"), result.get("map50_95"))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
