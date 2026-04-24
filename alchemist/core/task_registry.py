"""Centralized Task Type Registry for Alchemist.

Every vision task type (classification, detection, segmentation, ...) registers
its metadata here: worker script, eval metric, model catalog, technique catalog,
default training config, and PwC dataset aliases.

Other agents (Benchmark, Research, Controller) and the Executor query this
registry instead of hardcoding task-specific logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskTypeMeta:
    """Metadata for a vision task type."""

    task_type: str
    worker_script: str
    eval_metric: str
    model_framework: str  # "timm", "ultralytics", "transformers", "torch_hub"
    known_models: list[dict[str, Any]] = field(default_factory=list)
    published_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    technique_catalog: dict[str, dict] = field(default_factory=dict)
    default_config: dict[str, Any] = field(default_factory=dict)
    default_priority_techs: list[str] = field(default_factory=list)
    dataset_aliases: list[str] = field(default_factory=list)
    benchmark_metrics: list[str] = field(default_factory=list)
    higher_is_better: bool = True
    # GPU-based model tiers: {min_gpu_gb: model_name}
    gpu_model_tiers: dict[int, str] = field(default_factory=dict)
    # Model upgrade path: {current: next_larger}
    model_upgrade_path: dict[str, str] = field(default_factory=dict)
    # Published ceilings: {model_name: expected_metric}
    model_ceilings: dict[str, float] = field(default_factory=dict)


# ─── Task Registry ───────────────────────────────────────────────────────────

TASK_REGISTRY: dict[str, TaskTypeMeta] = {}

# ─── Task name → type keyword mapping ────────────────────────────────────────

_TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    # Order matters: more specific types first, then general ones
    "pose_estimation": ["pose", "coco_pose", "mpii", "crowdpose", "humanpose"],
    "segmentation": ["segmentation", "seg", "ade20k", "cityscapes", "pascal_seg", "coco_seg", "stuff"],
    "detection": ["detection", "coco_detection", "voc", "objects365", "det", "openimages", "coco"],
    "video_classification": ["ucf101", "hmdb51", "kinetics", "ssv2", "diving48", "video_cls"],
    "vlm": ["llava_video", "video_mme", "vlm_training", "video_qa", "video_caption"],
    "classification": [],  # fallback — matches anything not caught above
}


def detect_task_type(task_name: str) -> str:
    """Infer task type from task name via keyword matching."""
    low = task_name.lower()
    for task_type, keywords in _TASK_TYPE_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return task_type
    return "classification"  # default fallback


def get_task_meta(task_type: str) -> TaskTypeMeta:
    """Get metadata for a task type. Falls back to classification."""
    if task_type not in TASK_REGISTRY:
        logger.warning("Unknown task type '%s', falling back to classification", task_type)
        task_type = "classification"
    return TASK_REGISTRY[task_type]


def get_task_meta_for_name(task_name: str) -> TaskTypeMeta:
    """Convenience: detect task type from name and return metadata."""
    return get_task_meta(detect_task_type(task_name))


# ═══════════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY["classification"] = TaskTypeMeta(
    task_type="classification",
    worker_script="train_worker.py",
    eval_metric="top1_accuracy",
    model_framework="timm",
    known_models=[
        {"name": "resnet50", "backend": "timm", "model_id": "resnet50.a1_in1k", "params_m": 26, "img_size": 224},
        {"name": "dinov2_vitb14", "backend": "timm", "model_id": "vit_base_patch14_dinov2.lvd142m", "params_m": 86, "img_size": 224},
        {"name": "dinov2_vits14", "backend": "timm", "model_id": "vit_small_patch14_dinov2.lvd142m", "params_m": 22, "img_size": 224},
        {"name": "vit_b16_dino", "backend": "timm", "model_id": "vit_base_patch16_224.dino", "params_m": 86, "img_size": 224},
        {"name": "vit_s16_dino", "backend": "timm", "model_id": "vit_small_patch16_224.dino", "params_m": 22, "img_size": 224},
        {"name": "vit_s16_supervised", "backend": "timm", "model_id": "vit_small_patch16_224.augreg_in21k_ft_in1k", "params_m": 22, "img_size": 224},
        {"name": "vitamin_s_clip", "backend": "timm", "model_id": "vitamin_small_224.datacomp1b_clip", "params_m": 22, "img_size": 224},
        {"name": "mvitv2_tiny", "backend": "timm", "model_id": "mvitv2_tiny.fb_in1k", "params_m": 24, "img_size": 224},
    ],
    published_scores={
        "resnet50":          {"linear_probe": 75.8, "knn": 68.5, "detection_ap": 38.0},
        "dinov2_vitb14":     {"linear_probe": 87.3, "knn": 82.1, "detection_ap": 47.2},
        "dinov2_vits14":     {"linear_probe": 82.2, "knn": 79.0, "detection_ap": 39.1},
        "vit_b16_dino":      {"linear_probe": 81.2, "knn": 76.1, "detection_ap": 42.3},
        "vit_s16_dino":      {"linear_probe": 78.0, "knn": 72.8, "detection_ap": 35.5},
        "vit_s16_supervised": {"linear_probe": 80.5, "knn": 74.2, "detection_ap": 40.1},
        "vitamin_s_clip":    {"linear_probe": 79.3, "knn": 75.5, "detection_ap": 38.8},
        "mvitv2_tiny":       {"linear_probe": 77.6, "knn": 71.3, "detection_ap": 34.2},
    },
    technique_catalog={
        "sam":                  {"optimizer": "sam", "sam_rho": 0.05},
        "sam_conservative":     {"optimizer": "sam", "sam_rho": 0.02},
        "sam_aggressive":       {"optimizer": "sam", "sam_rho": 0.1},
        "mixup_light":          {"mixup": True, "mixup_alpha": 0.2},
        "mixup_standard":       {"mixup": True, "mixup_alpha": 0.8},
        "cutmix":               {"cutmix": True, "cutmix_alpha": 1.0},
        "cutmix_light":         {"cutmix": True, "cutmix_alpha": 0.5},
        "randaugment":          {"randaugment": True},
        "random_erasing":       {"extra": {"random_erasing": True, "random_erasing_prob": 0.25}},
        "stochastic_depth":     {"extra": {"drop_path_rate": 0.1}},
        "stochastic_depth_strong": {"extra": {"drop_path_rate": 0.3}},
        "label_smoothing":      {"label_smoothing": 0.1},
        "ema":                  {"ema": True, "ema_decay": 0.9999},
        "llrd":                 {"backbone_lr_scale": 0.7},
        "weight_decay_light":   {"weight_decay": 0.01},
        "weight_decay_heavy":   {"weight_decay": 0.05},
        "cosine_restarts":      {"extra": {"lr_schedule": "cosine_restarts"}},
        "onecycle":             {"extra": {"lr_schedule": "onecycle"}},
        "longer_training_30ep": {"epochs": 30, "warmup_epochs": 3},
        "longer_training_50ep": {"epochs": 50, "warmup_epochs": 5},
        "se_attention":         {"extra": {"add_se": True}},
        "cbam_attention":       {"extra": {"add_cbam": True}},
        "self_attention_2d":    {"extra": {"add_self_attention": True, "self_attn_heads": 4}},
        "lora_attn":            {"extra": {"add_lora": True, "lora_rank": 8, "lora_targets": "attn"}},
        "lora_qkv":             {"extra": {"add_lora": True, "lora_rank": 4, "lora_targets": "qkv"}},
        "adapter_houlsby":      {"extra": {"add_adapter": True, "adapter_bottleneck": 64}},
    },
    default_config={
        "lr": 1e-4, "batch_size": 32, "epochs": 20,
        "freeze_backbone": False, "adapter": "linear_head",
        "optimizer": "sam", "sam_rho": 0.05,
        "mixup": True, "mixup_alpha": 0.3,
        "cutmix": True, "cutmix_alpha": 0.5,
        "randaugment": True, "label_smoothing": 0.1,
        "ema": True, "ema_decay": 0.9999,
        "warmup_epochs": 2, "backbone_lr_scale": 0.7,
        "weight_decay": 0.02,
    },
    default_priority_techs=[
        "stochastic_depth", "random_erasing", "sam_aggressive",
        "cosine_restarts", "longer_training_50ep",
    ],
    dataset_aliases=["CIFAR-100", "CIFAR-10", "ImageNet", "ImageNet-1K"],
    benchmark_metrics=["linear_probe", "knn"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Object Detection
# ═══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY["detection"] = TaskTypeMeta(
    task_type="detection",
    worker_script="detection_worker.py",
    eval_metric="mAP50-95",
    model_framework="ultralytics",
    known_models=[
        {"name": "yolov8n", "backend": "ultralytics", "model_id": "yolov8n.pt", "params_m": 3, "img_size": 640},
        {"name": "yolov8s", "backend": "ultralytics", "model_id": "yolov8s.pt", "params_m": 11, "img_size": 640},
        {"name": "yolov8m", "backend": "ultralytics", "model_id": "yolov8m.pt", "params_m": 26, "img_size": 640},
        {"name": "yolov8l", "backend": "ultralytics", "model_id": "yolov8l.pt", "params_m": 44, "img_size": 640},
        {"name": "yolov8x", "backend": "ultralytics", "model_id": "yolov8x.pt", "params_m": 68, "img_size": 640},
        {"name": "yolo11n", "backend": "ultralytics", "model_id": "yolo11n.pt", "params_m": 3, "img_size": 640},
        {"name": "yolo11s", "backend": "ultralytics", "model_id": "yolo11s.pt", "params_m": 10, "img_size": 640},
        {"name": "yolo11m", "backend": "ultralytics", "model_id": "yolo11m.pt", "params_m": 20, "img_size": 640},
        {"name": "yolo11l", "backend": "ultralytics", "model_id": "yolo11l.pt", "params_m": 26, "img_size": 640},
        {"name": "yolo11x", "backend": "ultralytics", "model_id": "yolo11x.pt", "params_m": 57, "img_size": 640},
        {"name": "rtdetr-l", "backend": "ultralytics", "model_id": "rtdetr-l.pt", "params_m": 32, "img_size": 640},
        {"name": "rtdetr-x", "backend": "ultralytics", "model_id": "rtdetr-x.pt", "params_m": 65, "img_size": 640},
    ],
    published_scores={
        "yolov8n": {"mAP50": 52.6, "mAP50_95": 37.3},
        "yolov8s": {"mAP50": 61.8, "mAP50_95": 44.9},
        "yolov8m": {"mAP50": 67.2, "mAP50_95": 50.2},
        "yolov8l": {"mAP50": 69.8, "mAP50_95": 52.9},
        "yolov8x": {"mAP50": 71.0, "mAP50_95": 53.9},
        "yolo11n": {"mAP50": 53.4, "mAP50_95": 39.5},
        "yolo11s": {"mAP50": 62.5, "mAP50_95": 47.0},
        "yolo11m": {"mAP50": 68.1, "mAP50_95": 51.5},
        "yolo11l": {"mAP50": 69.4, "mAP50_95": 53.4},
        "yolo11x": {"mAP50": 70.6, "mAP50_95": 54.7},
        "rtdetr-l": {"mAP50": 71.4, "mAP50_95": 53.0},
        "rtdetr-x": {"mAP50": 72.8, "mAP50_95": 54.8},
    },
    technique_catalog={
        "yolov8n": {"base_model": "yolov8n", "batch_size": 32},
        "yolov8s": {"base_model": "yolov8s", "batch_size": 32},
        "yolov8m": {"base_model": "yolov8m", "batch_size": 16},
        "yolov8l": {"base_model": "yolov8l", "batch_size": 16},
        "yolov8x": {"base_model": "yolov8x", "batch_size": 8},
        "yolo11m": {"base_model": "yolo11m", "batch_size": 16},
        "yolo11l": {"base_model": "yolo11l", "batch_size": 8},
        "yolo11x": {"base_model": "yolo11x", "batch_size": 8},
        "rtdetr_l": {"base_model": "rtdetr-l", "batch_size": 8},
        "rtdetr_x": {"base_model": "rtdetr-x", "batch_size": 4},
        "mosaic_on": {"extra": {"mosaic": 1.0}},
        "mosaic_off": {"extra": {"mosaic": 0.0}},
        "mixup_det": {"extra": {"mixup": 0.15}},
        "copy_paste": {"extra": {"copy_paste": 0.1}},
        "copy_paste_strong": {"extra": {"copy_paste": 0.3}},
        "multi_scale": {"extra": {"multi_scale": 0.5}},
        "random_erasing_det": {"extra": {"erasing": 0.4}},
        "resolution_640": {"img_size": 640},
        "resolution_800": {"img_size": 800},
        "resolution_1280": {"img_size": 1280},
        "sgd_optimizer": {"extra": {"optimizer": "SGD"}},
        "adamw_optimizer": {"extra": {"optimizer": "AdamW"}},
        "lr_high": {"lr": 0.02},
        "lr_low": {"lr": 0.005},
        "lr_very_low": {"lr": 0.001},
        "long_training": {"epochs": 100},
        "box_loss_high": {"extra": {"box": 10.0}},
        "cls_loss_low": {"extra": {"cls": 0.3}},
        "dfl_loss_high": {"extra": {"dfl": 2.0}},
        "freeze_backbone_10": {"extra": {"freeze": 10}},
        "cos_lr": {"extra": {"cos_lr": True}},
        "close_mosaic_10": {"extra": {"close_mosaic": 10}},
    },
    default_config={
        "base_model": "yolov8m", "batch_size": 16, "epochs": 50,
        "img_size": 640, "lr": 0.01, "optimizer": "auto",
        "augmentation": "advanced", "patience": 10,
        "device": "0", "workers": 4,
    },
    default_priority_techs=[
        "yolov8l", "yolov8x", "yolo11l", "yolo11x",
        "rtdetr_l", "rtdetr_x",
        "resolution_800", "long_training", "copy_paste",
        "mixup_det", "lr_low", "freeze_backbone_10", "multi_scale",
    ],
    dataset_aliases=["COCO", "COCO-2017", "PASCAL VOC", "Objects365"],
    benchmark_metrics=["mAP50_95", "mAP50"],
    gpu_model_tiers={
        40: "yolov8x",   # L40S(46GB), A100(80GB), H100(80GB)
        24: "yolov8l",   # A10G(24GB), RTX 4090(24GB)
        16: "yolov8m",   # T4(16GB)
        8: "yolov8s",    # RTX 3060(8GB)
        0: "yolov8n",
    },
    model_upgrade_path={
        "yolov8n": "yolov8s", "yolov8s": "yolov8m",
        "yolov8m": "yolov8l", "yolov8l": "yolov8x",
        "yolo11n": "yolo11s", "yolo11s": "yolo11m",
        "yolo11m": "yolo11l", "yolo11l": "yolo11x",
        "yolov8x": "rtdetr-l", "yolo11x": "rtdetr-l",
        "rtdetr-l": "rtdetr-x",
    },
    model_ceilings={
        "yolov8n": 37.3, "yolov8s": 44.9, "yolov8m": 50.2,
        "yolov8l": 52.9, "yolov8x": 53.9,
        "yolo11m": 51.5, "yolo11l": 53.4, "yolo11x": 54.7,
        "rtdetr-l": 53.0, "rtdetr-x": 54.8,
    },
)

# ═══════════════════════════════════════════════════════════════════════════════
# Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY["segmentation"] = TaskTypeMeta(
    task_type="segmentation",
    worker_script="segmentation_worker.py",
    eval_metric="mIoU",
    model_framework="ultralytics",
    known_models=[
        {"name": "yolov8n-seg", "backend": "ultralytics", "model_id": "yolov8n-seg.pt", "params_m": 4, "img_size": 640},
        {"name": "yolov8s-seg", "backend": "ultralytics", "model_id": "yolov8s-seg.pt", "params_m": 12, "img_size": 640},
        {"name": "yolov8m-seg", "backend": "ultralytics", "model_id": "yolov8m-seg.pt", "params_m": 27, "img_size": 640},
        {"name": "yolov8l-seg", "backend": "ultralytics", "model_id": "yolov8l-seg.pt", "params_m": 46, "img_size": 640},
        {"name": "yolov8x-seg", "backend": "ultralytics", "model_id": "yolov8x-seg.pt", "params_m": 71, "img_size": 640},
    ],
    published_scores={
        "yolov8n-seg": {"mIoU": 30.5, "mask_mAP50_95": 36.7},
        "yolov8s-seg": {"mIoU": 36.8, "mask_mAP50_95": 44.6},
        "yolov8m-seg": {"mIoU": 40.8, "mask_mAP50_95": 49.9},
        "yolov8l-seg": {"mIoU": 42.6, "mask_mAP50_95": 52.3},
        "yolov8x-seg": {"mIoU": 43.4, "mask_mAP50_95": 53.4},
    },
    technique_catalog={
        "yolov8m-seg": {"base_model": "yolov8m-seg", "batch_size": 16},
        "yolov8l-seg": {"base_model": "yolov8l-seg", "batch_size": 8},
        "yolov8x-seg": {"base_model": "yolov8x-seg", "batch_size": 4},
        "resolution_800": {"img_size": 800},
        "mosaic_on": {"extra": {"mosaic": 1.0}},
        "copy_paste": {"extra": {"copy_paste": 0.1}},
        "long_training": {"epochs": 100},
    },
    default_config={
        "base_model": "yolov8m-seg", "batch_size": 16, "epochs": 50,
        "img_size": 640, "lr": 0.01, "optimizer": "auto",
        "patience": 10, "device": "0", "workers": 4,
    },
    default_priority_techs=["yolov8l-seg", "yolov8x-seg", "resolution_800", "long_training"],
    dataset_aliases=["COCO-Seg", "ADE20K", "Cityscapes"],
    benchmark_metrics=["mIoU", "mask_mAP50_95"],
    gpu_model_tiers={40: "yolov8x-seg", 24: "yolov8l-seg", 16: "yolov8m-seg", 8: "yolov8s-seg", 0: "yolov8n-seg"},
)

# ═══════════════════════════════════════════════════════════════════════════════
# Pose Estimation
# ═══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY["pose_estimation"] = TaskTypeMeta(
    task_type="pose_estimation",
    worker_script="pose_worker.py",
    eval_metric="mAP",
    model_framework="ultralytics",
    known_models=[
        {"name": "yolov8n-pose", "backend": "ultralytics", "model_id": "yolov8n-pose.pt", "params_m": 3, "img_size": 640},
        {"name": "yolov8s-pose", "backend": "ultralytics", "model_id": "yolov8s-pose.pt", "params_m": 12, "img_size": 640},
        {"name": "yolov8m-pose", "backend": "ultralytics", "model_id": "yolov8m-pose.pt", "params_m": 26, "img_size": 640},
        {"name": "yolov8l-pose", "backend": "ultralytics", "model_id": "yolov8l-pose.pt", "params_m": 44, "img_size": 640},
        {"name": "yolov8x-pose", "backend": "ultralytics", "model_id": "yolov8x-pose.pt", "params_m": 69, "img_size": 640},
    ],
    published_scores={
        "yolov8n-pose": {"mAP": 50.4}, "yolov8s-pose": {"mAP": 60.0},
        "yolov8m-pose": {"mAP": 65.0}, "yolov8l-pose": {"mAP": 67.6},
        "yolov8x-pose": {"mAP": 69.2},
    },
    technique_catalog={
        "yolov8m-pose": {"base_model": "yolov8m-pose", "batch_size": 16},
        "yolov8l-pose": {"base_model": "yolov8l-pose", "batch_size": 8},
        "yolov8x-pose": {"base_model": "yolov8x-pose", "batch_size": 4},
        "resolution_800": {"img_size": 800},
        "long_training": {"epochs": 100},
    },
    default_config={
        "base_model": "yolov8m-pose", "batch_size": 16, "epochs": 50,
        "img_size": 640, "lr": 0.01, "optimizer": "auto",
        "patience": 10, "device": "0", "workers": 4,
    },
    default_priority_techs=["yolov8l-pose", "yolov8x-pose", "resolution_800", "long_training"],
    dataset_aliases=["COCO-Pose", "MPII", "CrowdPose"],
    benchmark_metrics=["mAP"],
    gpu_model_tiers={40: "yolov8x-pose", 24: "yolov8l-pose", 16: "yolov8m-pose", 8: "yolov8s-pose", 0: "yolov8n-pose"},
)

# ═══════════════════════════════════════════════════════════════════════════════
# Video Classification
# ═══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY["video_classification"] = TaskTypeMeta(
    task_type="video_classification",
    worker_script="video_worker.py",
    eval_metric="top1_accuracy",
    model_framework="torch_hub",
    known_models=[
        {"name": "vjepa2_vitl", "backend": "torch_hub", "model_id": "facebookresearch/vjepa2:vjepa2_vitl_16x384", "params_m": 305, "img_size": 384},
        {"name": "vjepa2_vith", "backend": "torch_hub", "model_id": "facebookresearch/vjepa2:vjepa2_vith_16x384", "params_m": 632, "img_size": 384},
        {"name": "videomae_base", "backend": "transformers", "model_id": "MCG-NJU/videomae-base", "params_m": 87, "img_size": 224},
        {"name": "videomae_large", "backend": "transformers", "model_id": "MCG-NJU/videomae-large", "params_m": 305, "img_size": 224},
    ],
    published_scores={
        "vjepa2_vitl": {"ucf101": 92.9, "hmdb51": 82.0},
        "vjepa2_vith": {"ucf101": 93.5, "hmdb51": 83.2},
        "videomae_base": {"ucf101": 91.3, "hmdb51": 76.8},
        "videomae_large": {"ucf101": 95.4, "hmdb51": 81.3},
    },
    technique_catalog={
        "video_8frames": {"extra": {"num_frames": 8, "frame_stride": 4}},
        "video_16frames": {"extra": {"num_frames": 16, "frame_stride": 4}},
        "video_32frames": {"extra": {"num_frames": 32, "frame_stride": 2}},
        "video_dense_sampling": {"extra": {"num_frames": 16, "frame_stride": 2}},
        "video_sparse_sampling": {"extra": {"num_frames": 8, "frame_stride": 8}},
        "longer_training_50ep": {"epochs": 50, "warmup_epochs": 5},
    },
    default_config={
        "lr": 1e-4, "batch_size": 8, "epochs": 20,
        "num_frames": 16, "frame_stride": 4,
        "warmup_epochs": 2, "weight_decay": 0.01,
    },
    default_priority_techs=["video_16frames", "video_32frames", "longer_training_50ep"],
    dataset_aliases=["UCF-101", "HMDB-51", "Kinetics-400", "Something-Something V2"],
    benchmark_metrics=["top1_accuracy"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Video Language Model (VLM)
# ═══════════════════════════════════════════════════════════════════════════════

TASK_REGISTRY["vlm"] = TaskTypeMeta(
    task_type="vlm",
    worker_script="vlm_worker.py",
    eval_metric="accuracy",
    model_framework="transformers",
    known_models=[
        {"name": "qwen3.5-9b", "backend": "transformers", "model_id": "Qwen/Qwen3.5-9B", "params_m": 9000, "img_size": 384},
        {"name": "llava-video-7b", "backend": "transformers", "model_id": "lmms-lab/LLaVA-Video-7B", "params_m": 7000, "img_size": 384},
    ],
    published_scores={
        "qwen3.5-9b": {"video_mme": 47.0},
        "llava-video-7b": {"video_mme": 42.0},
    },
    technique_catalog={},
    default_config={
        "lr": 2e-5, "batch_size": 1, "epochs": 1,
        "grad_accum": 16, "num_frames": 16,
        "lora_r": 32, "lora_alpha": 64,
    },
    default_priority_techs=[],
    dataset_aliases=["LLaVA-Video-178K", "Video-MME"],
    benchmark_metrics=["accuracy"],
)


# ─── Utility: select model for GPU ───────────────────────────────────────────

def select_model_for_gpu(task_meta: TaskTypeMeta, gpu_gb: float | None = None) -> str:
    """Select the largest model that fits the available GPU.

    Args:
        task_meta: Task metadata with gpu_model_tiers.
        gpu_gb: GPU memory in GB. If None, tries local torch.cuda first,
                then falls back to the largest tier (assumes remote GPU is large).
    """
    if not task_meta.gpu_model_tiers:
        return task_meta.default_config.get("base_model", "")

    if gpu_gb is None:
        # Try local GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_gb = torch.cuda.get_properties(0).total_memory / 1e9
        except Exception:
            pass

    if gpu_gb is None or gpu_gb == 0:
        # No GPU info available (likely running locally with remote executor)
        # Default to largest tier — remote GPUs are typically large
        largest_tier = max(task_meta.gpu_model_tiers.keys())
        model = task_meta.gpu_model_tiers[largest_tier]
        logger.info("[task_registry] No local GPU — assuming remote large GPU → %s", model)
        return model

    # Pick largest model whose tier threshold <= GPU memory
    for min_gb in sorted(task_meta.gpu_model_tiers.keys(), reverse=True):
        if gpu_gb >= min_gb:
            model = task_meta.gpu_model_tiers[min_gb]
            logger.info("[task_registry] GPU %.0fGB → %s", gpu_gb, model)
            return model

    # Fallback to smallest
    smallest_tier = min(task_meta.gpu_model_tiers.keys())
    return task_meta.gpu_model_tiers[smallest_tier]

    # Fallback
    return task_meta.default_config.get("base_model", "")
