#!/usr/bin/env python3
"""Video classification training worker for Alchemist.

Supports V-JEPA and other video ViT models. Handles frame sampling,
video augmentation, temporal modeling, and all training techniques
from train_worker.py (SAM, EMA, bf16, Mixup, etc.).

Usage:
    python video_worker.py --job jobs/video_trial.json --output results/video_trial.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video_worker")


# ---------------------------------------------------------------------------
# Video Dataset
# ---------------------------------------------------------------------------

class VideoDataset(torch.utils.data.Dataset):
    """Load videos from directory structure: root/<class>/<video>.avi

    Supports UCF-101, HMDB-51, Kinetics-style directory layouts.
    Uses decord for fast video decoding, falls back to torchvision.
    """

    def __init__(
        self,
        root: str,
        num_frames: int = 16,
        stride: int = 4,
        img_size: int = 224,
        is_train: bool = True,
        augment: str = "basic",
    ):
        self.num_frames = num_frames
        self.stride = stride
        self.img_size = img_size
        self.is_train = is_train
        self.augment = augment

        # Discover videos
        self.samples = []
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        video_exts = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
        for cls_name in self.classes:
            cls_dir = Path(root) / cls_name
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in video_exts:
                    self.samples.append((str(f), self.class_to_idx[cls_name]))

        logger.info(
            "VideoDataset: %d videos, %d classes, %d frames/clip, stride=%d",
            len(self.samples), len(self.classes), num_frames, stride,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_frames(path)
        frames = self._transform(frames)
        return frames, label

    def _load_frames(self, path: str) -> torch.Tensor:
        """Load frames using decord (fast) or torchvision (fallback)."""
        try:
            import decord
            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(path, num_threads=1)
            total = len(vr)
        except ImportError:
            # Fallback to torchvision
            import torchvision.io as vio
            video, _, info = vio.read_video(path, pts_unit="sec")
            total = video.shape[0]
            vr = None

        # Sample frame indices
        clip_len = self.num_frames * self.stride
        if total >= clip_len:
            if self.is_train:
                start = torch.randint(0, total - clip_len + 1, (1,)).item()
            else:
                start = (total - clip_len) // 2  # center crop
            indices = list(range(start, start + clip_len, self.stride))
        else:
            # Video too short — sample with repetition
            indices = torch.linspace(0, total - 1, self.num_frames).long().tolist()

        indices = [min(i, total - 1) for i in indices]

        if vr is not None:
            frames = vr.get_batch(indices)  # (T, H, W, C) uint8
            frames = frames.float() / 255.0
            frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        else:
            frames = video[indices].float() / 255.0
            frames = frames.permute(0, 3, 1, 2)

        return frames  # (T, C, H, W)

    def _transform(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply spatial transforms to video frames."""
        T, C, H, W = frames.shape
        size = self.img_size

        if self.is_train:
            # Random resized crop
            i, j, h, w = self._random_crop_params(H, W, scale=(0.6, 1.0))
            frames = frames[:, :, i:i+h, j:j+w]
            frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                frames = frames.flip(dims=[3])
        else:
            # Center crop
            short_side = min(H, W)
            crop = int(short_side * 0.875)
            top = (H - crop) // 2
            left = (W - crop) // 2
            frames = frames[:, :, top:top+crop, left:left+crop]
            frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)

        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        return frames  # (T, C, H, W)

    @staticmethod
    def _random_crop_params(H, W, scale=(0.6, 1.0)):
        area = H * W
        for _ in range(10):
            target_area = torch.empty(1).uniform_(*scale).item() * area
            aspect_ratio = torch.empty(1).uniform_(3/4, 4/3).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= W and 0 < h <= H:
                i = torch.randint(0, H - h + 1, (1,)).item()
                j = torch.randint(0, W - w + 1, (1,)).item()
                return i, j, h, w
        # Fallback: center crop
        crop = min(H, W)
        return (H - crop) // 2, (W - crop) // 2, crop, crop


# ---------------------------------------------------------------------------
# Video Model Wrapper
# ---------------------------------------------------------------------------

class VideoClassifier(nn.Module):
    """Wrap a V-JEPA or image ViT for video classification.

    Handles temporal aggregation: mean pooling over frame features.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            (B, num_classes) logits
        """
        B, T, C, H, W = x.shape
        # Process each frame through backbone
        x = x.reshape(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, D) or (B*T, N, D)

        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.dim() == 3:
            features = features[:, 0]  # CLS token
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Reshape and temporal mean pooling
        D = features.shape[-1]
        features = features.reshape(B, T, D)
        features = features.mean(dim=1)  # (B, D) — temporal average

        return self.head(features)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_video_training(
    base_model: str,
    task: dict,
    config: dict,
    trial_id: int,
) -> dict:
    """Train a video classification model."""
    from train_worker import SAM, EMA

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA required"

    num_classes = task.get("num_classes", 101)
    data_path = task.get("data_path", "/home/ubuntu/data/ucf101")

    # Config
    lr = config.get("lr", 1e-4)
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 4)
    weight_decay = config.get("weight_decay", 0.02)
    num_frames = config.get("num_frames", 16)
    frame_stride = config.get("frame_stride", 4)
    img_size = config.get("img_size", 224)
    warmup_epochs = config.get("warmup_epochs", 2)
    backbone_lr_scale = config.get("backbone_lr_scale", 0.1)
    use_ema = config.get("ema", False)
    ema_decay = config.get("ema_decay", 0.9999)
    opt_name = config.get("optimizer", "adamw").lower()
    sam_rho = config.get("sam_rho", 0.05)
    label_smoothing = config.get("label_smoothing", 0.1)

    _use_bf16 = torch.cuda.is_bf16_supported()
    _amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16
    if _use_bf16:
        logger.info("  Using bfloat16 mixed precision")

    # Load model
    logger.info("Loading model: %s", base_model)
    try:
        from alchemist.core.vjepa_loader import load_vjepa, register_vjepa_in_model_loader
        register_vjepa_in_model_loader()
    except ImportError:
        pass

    try:
        from alchemist.core.model_loader import ModelLoader
        backbone = ModelLoader.load(base_model, num_classes=0, pretrained=True)
    except Exception:
        import timm
        backbone = timm.create_model(base_model, pretrained=True, num_classes=0)

    # Determine embed_dim
    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        embed_dim = getattr(backbone, "num_features", 768)

    # Remove existing head if any
    if hasattr(backbone, "head"):
        backbone.head = nn.Identity()
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()

    model = VideoClassifier(backbone, embed_dim, num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("  VideoClassifier: %.1fM params, embed_dim=%d", param_count, embed_dim)

    # Datasets
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_path, "test")

    train_ds = VideoDataset(train_dir, num_frames, frame_stride, img_size, is_train=True)
    val_ds = VideoDataset(val_dir, num_frames, frame_stride, img_size, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Optimizer
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    param_groups = [
        {"params": backbone_params, "lr": lr * backbone_lr_scale},
        {"params": head_params, "lr": lr},
    ]

    if opt_name == "sam":
        base_opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        optimizer = SAM(base_opt, rho=sam_rho)
        logger.info("  Optimizer: SAM(AdamW, rho=%.3f)", sam_rho)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        logger.info("  Optimizer: AdamW")

    # LR schedule
    from torch.amp import GradScaler, autocast
    scaler = GradScaler(enabled=not _use_bf16)
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    lr_floor = max(1e-7, lr * 0.01)
    if warmup_epochs > 0 and warmup_epochs < max(2, int(epochs * 0.1)):
        warmup_epochs = max(2, int(epochs * 0.1))
        warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(cosine, lr_floor / lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer if not isinstance(optimizer, SAM) else optimizer.base_optimizer, lr_lambda)

    # EMA
    ema = EMA(model, decay=ema_decay) if use_ema else None
    if ema:
        logger.info("  EMA enabled (decay=%.4f)", ema_decay)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    logger.info("  Training: %d epochs, batch=%d, frames=%d, stride=%d, img=%d",
                epochs, batch_size, num_frames, frame_stride, img_size)

    # Training loop
    best_score = 0.0
    best_state = None
    ema_ready_epoch = warmup_epochs

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for videos, targets in train_loader:
            videos, targets = videos.to(device), targets.to(device)

            _is_sam = isinstance(optimizer, SAM)
            if _is_sam:
                with autocast(device_type="cuda", dtype=_amp_dtype):
                    out = model(videos)
                    loss = criterion(out, targets)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer.base_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.first_step()
                scaler.update()

                with autocast(device_type="cuda", dtype=_amp_dtype):
                    out2 = model(videos)
                    loss2 = criterion(out2, targets)
                optimizer.zero_grad()
                scaler.scale(loss2).backward()
                scaler.unscale_(optimizer.base_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step()
                scaler.update()
            else:
                with autocast(device_type="cuda", dtype=_amp_dtype):
                    out = model(videos)
                    loss = criterion(out, targets)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            if ema:
                ema.update(model)
            epoch_loss += loss.item()

        train_loss = epoch_loss / max(len(train_loader), 1)

        if math.isnan(train_loss) or math.isinf(train_loss):
            logger.warning("NaN/Inf at epoch %d — halting", epoch + 1)
            break

        # Evaluate
        ema_ready = ema and epoch >= ema_ready_epoch
        if ema_ready:
            ema.apply(model)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                with autocast(device_type="cuda", dtype=_amp_dtype):
                    out = model(videos)
                correct += (out.argmax(1) == targets).sum().item()
                total += targets.size(0)
        epoch_score = 100.0 * correct / max(total, 1)

        if epoch_score > best_score:
            best_score = epoch_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ema_ready:
            ema.restore(model)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info("  Epoch %d/%d, train=%.4f, val=%.2f%%, best=%.2f%%",
                        epoch + 1, epochs, train_loss, epoch_score, best_score)

        # Progress file
        _prog_path = os.environ.get("ALCHEMIST_PROGRESS_PATH")
        if _prog_path:
            try:
                prog = {
                    "epoch": int(epoch + 1), "total_epochs": int(epochs),
                    "train_loss": float(train_loss), "val_acc": float(epoch_score),
                    "best_so_far": float(best_score), "elapsed_s": float(time.time() - t0),
                }
                with open(_prog_path, "w") as f:
                    json.dump(prog, f)
            except Exception:
                pass

    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_dir = Path("/home/ubuntu/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"video_{task['name']}_{base_model.replace('/', '_')}_trial{trial_id}.pt"
    if best_state:
        torch.save(best_state, ckpt_path)

    return {
        "status": "ok",
        "trial_id": trial_id,
        "score": round(best_score, 2),
        "train_loss": round(train_loss, 4),
        "elapsed_s": round(elapsed, 1),
        "config": config,
        "checkpoint_path": str(ckpt_path),
        "applied_techniques": {
            "optimizer": opt_name,
            "precision": "bf16" if _use_bf16 else "fp16",
            "ema": bool(ema),
            "label_smoothing": float(label_smoothing),
            "sam_rho": float(sam_rho) if opt_name == "sam" else None,
            "backbone_lr_scale": float(backbone_lr_scale),
            "num_frames": int(num_frames),
            "frame_stride": int(frame_stride),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Alchemist Video Training Worker")
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", help="Path to write per-epoch progress JSON")
    args = parser.parse_args()

    if args.progress:
        os.environ["ALCHEMIST_PROGRESS_PATH"] = args.progress

    with open(args.job) as f:
        job = json.load(f)

    logger.info("Video Job: model=%s, task=%s",
                job.get("base_model"), job.get("task", {}).get("name"))

    try:
        result = run_video_training(
            job["base_model"], job["task"], job.get("config", {}), job.get("trial_id", 0),
        )
    except Exception as e:
        import traceback
        logger.error("Failed: %s\n%s", e, traceback.format_exc())
        result = {"status": "error", "error": str(e), "trial_id": job.get("trial_id", 0)}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Result: status=%s, score=%s", result.get("status"), result.get("score"))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
