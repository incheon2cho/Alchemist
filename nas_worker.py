#!/usr/bin/env python3
"""NAS Training Worker — Neural Architecture Search + HP optimization on GPU.

Supports:
- Architecture search: depth, width, block type, attention, head design
- Training tricks: OneCycleLR, Mixup, CutMix, RandAugment, Label Smoothing, EMA
- 8-hour budget-aware execution

Usage:
    python nas_worker.py --job jobs/nas_1.json --output results/nas_1.json

Job JSON:
{
  "command": "train",
  "trial_id": 1,
  "task": {"name": "cifar100", "data_path": "/home/ubuntu/data/cifar100", "num_classes": 100},
  "arch": {
    "backbone": "resnet50",       // resnet18/34/50/101, wide_resnet50_2, resnext50_32x4d, efficientnet_b0/b2, convnext_tiny
    "width_mult": 1.0,           // channel width multiplier
    "add_se": false,             // SE blocks
    "add_cbam": false,           // CBAM attention
    "head_type": "mlp",          // linear, mlp, mlp_deep
    "head_hidden": 512,          // head hidden dim
    "head_dropout": 0.3,
    "drop_path_rate": 0.0
  },
  "config": {
    "lr": 0.001, "epochs": 50, "batch_size": 128,
    "weight_decay": 0.05, "label_smoothing": 0.1,
    "lr_schedule": "onecycle",
    "backbone_lr_scale": 0.1,
    "warmup_epochs": 5,
    "mixup": true, "mixup_alpha": 0.2,
    "cutmix": true, "cutmix_alpha": 1.0,
    "randaugment": true, "random_erasing": true,
    "ema": false, "ema_decay": 0.999
  }
}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nas_worker")


# ============================================================
# Architecture Builder
# ============================================================

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        # Channel
        avg_w = self.fc(self.avg_pool(x).view(B, C))
        max_w = self.fc(self.max_pool(x).view(B, C))
        ch_attn = torch.sigmoid(avg_w + max_w).view(B, C, 1, 1)
        x = x * ch_attn
        # Spatial
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True)[0]
        sp_attn = self.spatial(torch.cat([avg_s, max_s], dim=1))
        return x * sp_attn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1)
        return x * w


def build_head(in_features: int, num_classes: int, arch: dict) -> nn.Module:
    """Build classification head."""
    head_type = arch.get("head_type", "linear")
    hidden = arch.get("head_hidden", 512)
    drop = arch.get("head_dropout", 0.3)

    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes),
        )
    elif head_type == "mlp_deep":
        return nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(1024, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(drop * 0.7),
            nn.Linear(hidden, num_classes),
        )
    else:
        return nn.Linear(in_features, num_classes)


def build_model(arch: dict, num_classes: int, device: torch.device) -> nn.Module:
    """Build model from architecture specification."""
    backbone_name = arch.get("backbone", "resnet50")
    drop_path = arch.get("drop_path_rate", 0.0)

    # Map common names to timm model IDs
    TIMM_MAP = {
        "resnet18": "resnet18.a1_in1k",
        "resnet34": "resnet34.a1_in1k",
        "resnet50": "resnet50.a1_in1k",
        "resnet101": "resnet101.a1_in1k",
        "wide_resnet50": "wide_resnet50_2.tv2_in1k",
        "wide_resnet101": "wide_resnet101_2.tv2_in1k",
        "resnext50": "resnext50_32x4d.a1h_in1k",
        "resnext101": "resnext101_32x8d.tv2_in1k",
        "efficientnet_b0": "efficientnet_b0.ra_in1k",
        "efficientnet_b2": "efficientnet_b2.ra_in1k",
        "efficientnet_b3": "tf_efficientnet_b3.aa_in1k",
        "convnext_tiny": "convnext_tiny.fb_in1k",
        "convnext_small": "convnext_small.fb_in1k",
        "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
        "densenet121": "densenet121.tv_in1k",
        "regnetx_032": "regnetx_032.tv2_in1k",
        "regnety_032": "regnety_032.tv2_in1k",
    }

    # Special handling for Vision Mamba (not in timm)
    if backbone_name.startswith("vim_"):
        from vision_mamba import Vim
        vim_configs = {
            "vim_tiny":  {"dim": 192, "depth": 12, "dt_rank": 24, "dim_inner": 192, "d_state": 192},
            "vim_small": {"dim": 384, "depth": 24, "dt_rank": 48, "dim_inner": 384, "d_state": 384},
        }
        cfg = vim_configs.get(backbone_name, vim_configs["vim_small"])
        model = Vim(
            **cfg,
            num_classes=num_classes,
            image_size=224,
            patch_size=16,
            channels=3,
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("Vision Mamba: %s, %.1fM params (no pretrained, train from scratch)", backbone_name, param_count)
        return model

    model_id = TIMM_MAP.get(backbone_name, backbone_name)
    logger.info("Building model: %s (timm: %s)", backbone_name, model_id)

    model = timm.create_model(
        model_id, pretrained=True, num_classes=0,
        drop_path_rate=drop_path,
    )
    embed_dim = model.num_features

    # Inject attention modules for ResNet-family
    add_se = arch.get("add_se", False)
    add_cbam = arch.get("add_cbam", False)

    if add_se or add_cbam:
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name, None)
            if layer is None:
                continue
            new_blocks = []
            for block in layer:
                new_blocks.append(block)
                if hasattr(block, "conv3"):
                    ch = block.conv3.out_channels
                elif hasattr(block, "conv2"):
                    ch = block.conv2.out_channels
                else:
                    continue
                if add_cbam:
                    new_blocks.append(CBAM(ch))
                elif add_se:
                    new_blocks.append(SEBlock(ch))
            setattr(model, layer_name, nn.Sequential(*new_blocks))
        attn_type = "CBAM" if add_cbam else "SE"
        logger.info("  %s attention injected", attn_type)

    # Build head
    head = build_head(embed_dim, num_classes, arch)
    full_model = nn.Sequential(model, head)

    param_count = sum(p.numel() for p in full_model.parameters()) / 1e6
    logger.info("  Params: %.1fM, embed_dim: %d, head: %s",
                param_count, embed_dim, arch.get("head_type", "linear"))

    return full_model.to(device)


# ============================================================
# Data
# ============================================================

def build_loaders(task: dict, config: dict):
    """Build CIFAR-100 data loaders with augmentation."""
    img_size = config.get("img_size", 224)
    bs = config.get("batch_size", 128)

    train_tfms = [
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
    ]
    if config.get("randaugment"):
        train_tfms.append(T.RandAugment(num_ops=2, magnitude=9))
    else:
        train_tfms.append(T.ColorJitter(0.3, 0.3, 0.3, 0.1))

    train_tfms.extend([
        T.ToTensor(),
        T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    if config.get("random_erasing"):
        train_tfms.append(T.RandomErasing(p=0.25))

    val_tfms = [
        T.Resize(int(img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ]

    data_path = Path(task["data_path"])
    train_ds = ImageFolder(data_path / "train", T.Compose(train_tfms))
    val_ds = ImageFolder(data_path / "val", T.Compose(val_tfms))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ============================================================
# Training Utilities
# ============================================================

def mixup_data(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut = math.sqrt(1.0 - lam)
    ch, cw = int(H * cut), int(W * cut)
    cy, cx = torch.randint(0, H, (1,)).item(), torch.randint(0, W, (1,)).item()
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    x_c = x.clone()
    x_c[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_c, y, y[idx], lam


class EMA:
    """Exponential Moving Average — batch-level update."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
        self.step_count = 0

    def update(self, model):
        self.step_count += 1
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ============================================================
# Training Loop
# ============================================================

def run_training(task: dict, arch: dict, config: dict, trial_id: int) -> dict:
    """Full training pipeline."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = task.get("num_classes", 100)

    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 50)
    weight_decay = config.get("weight_decay", 0.05)
    label_smoothing = config.get("label_smoothing", 0.1)
    warmup_epochs = config.get("warmup_epochs", 5)
    backbone_lr_scale = config.get("backbone_lr_scale", 0.1)
    use_mixup = config.get("mixup", False)
    use_cutmix = config.get("cutmix", False)
    mixup_alpha = config.get("mixup_alpha", 0.2)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)
    use_ema = config.get("ema", False)
    ema_decay = config.get("ema_decay", 0.999)

    # Build model and data
    model = build_model(arch, num_classes, device)
    train_loader, val_loader = build_loaders(task, config)

    # Freeze backbone if backbone_lr_scale is 0 (linear probe mode)
    is_linear_probe = (backbone_lr_scale == 0.0 or backbone_lr_scale == 0)
    children = list(model.children())

    if is_linear_probe and len(children) >= 2:
        # True freeze: extract features once, train head only
        logger.info("  LINEAR PROBE MODE: freezing backbone, extracting features once")
        backbone = children[0]
        head = children[1]
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        # Extract features
        def extract_feats(loader):
            feats, labels = [], []
            with torch.no_grad():
                for imgs, tgts in loader:
                    imgs = imgs.to(device)
                    f = backbone(imgs).float()  # ensure float32
                    feats.append(f.cpu())
                    labels.append(tgts)
            return torch.cat(feats), torch.cat(labels)

        train_feats, train_labels = extract_feats(train_loader)
        val_feats, val_labels = extract_feats(val_loader)
        logger.info("  Features extracted: train=%s, val=%s", train_feats.shape, val_feats.shape)

        feat_train_loader = DataLoader(
            torch.utils.data.TensorDataset(train_feats, train_labels),
            batch_size=512, shuffle=True, drop_last=True,
        )
        feat_val_loader = DataLoader(
            torch.utils.data.TensorDataset(val_feats, val_labels),
            batch_size=512,
        )

        head = head.to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_score = 0.0
        best_state = None
        train_loss = val_loss = 0.0

        for epoch in range(epochs):
            head.train()
            epoch_loss = 0.0
            for feat, tgt in feat_train_loader:
                feat, tgt = feat.to(device), tgt.to(device)
                out = head(feat)
                loss = criterion(out, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            train_loss = epoch_loss / len(feat_train_loader)

            head.eval()
            correct = total = 0
            with torch.no_grad():
                for feat, tgt in feat_val_loader:
                    feat, tgt = feat.to(device), tgt.to(device)
                    out = head(feat)
                    correct += (out.argmax(1) == tgt).sum().item()
                    total += tgt.size(0)
            epoch_score = 100.0 * correct / total
            if epoch_score > best_score:
                best_score = epoch_score
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logger.info("  Epoch %d/%d, loss=%.4f, val=%.2f%%, best=%.2f%%",
                            epoch + 1, epochs, train_loss, epoch_score, best_score)

        elapsed = time.time() - t0
        ckpt_dir = Path("/home/ubuntu/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        backbone_name = arch.get("backbone", "unknown").replace(".", "_")
        ckpt_path = ckpt_dir / f"lp_{backbone_name}_trial{trial_id}.pt"
        if best_state:
            torch.save(best_state, ckpt_path)
        param_count = sum(p.numel() for p in head.parameters()) / 1e6

        return {
            "status": "ok", "trial_id": trial_id,
            "score": round(best_score, 2),
            "train_loss": round(train_loss, 4), "val_loss": round(train_loss, 4),
            "elapsed_s": round(elapsed, 1),
            "arch": arch, "config": config,
            "params_m": round(param_count, 1),
            "checkpoint_path": str(ckpt_path),
        }

    # Non-frozen: differential LR fine-tuning (original path)
    if len(children) >= 2:
        backbone_params = list(children[0].parameters())
        head_params = list(children[1].parameters())
        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
            {"params": head_params, "lr": lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler()

    # LR schedule
    lr_schedule = config.get("lr_schedule", "onecycle")
    total_steps = epochs * len(train_loader)

    if lr_schedule == "onecycle":
        max_lrs = [lr * backbone_lr_scale, lr] if len(children) >= 2 else [lr]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=total_steps,
            pct_start=config.get("onecycle_pct_start", 0.3),
            anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
        )
        step_per_batch = True
    elif lr_schedule == "cosine":
        warmup_steps = warmup_epochs * len(train_loader)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-6 / lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        step_per_batch = False

    logger.info("  LR: %s, epochs: %d, batch: %d", lr_schedule, epochs, config.get("batch_size", 128))

    # EMA
    ema = EMA(model, decay=ema_decay) if use_ema else None
    ema_warmup = max(warmup_epochs, 5) if ema else 0

    # Training
    best_score = 0.0
    best_state = None
    train_loss = val_loss = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, tgts in train_loader:
            imgs, tgts = imgs.to(device), tgts.to(device)

            # Mixup / CutMix
            apply_mix = use_mixup or use_cutmix
            if apply_mix:
                r = torch.rand(1).item()
                if use_cutmix and r < 0.5:
                    imgs, tgts_a, tgts_b, lam = cutmix_data(imgs, tgts, cutmix_alpha)
                elif use_mixup:
                    imgs, tgts_a, tgts_b, lam = mixup_data(imgs, tgts, mixup_alpha)
                else:
                    tgts_a, tgts_b, lam = tgts, tgts, 1.0

            with autocast(device_type="cuda"):
                out = model(imgs)
                if apply_mix:
                    loss = lam * criterion(out, tgts_a) + (1 - lam) * criterion(out, tgts_b)
                else:
                    loss = criterion(out, tgts)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if step_per_batch:
                scheduler.step()
            epoch_loss += loss.item()

            # EMA update every batch
            if ema:
                ema.update(model)

        if not step_per_batch:
            scheduler.step()
        train_loss = epoch_loss / len(train_loader)

        # Evaluate
        ema_ready = ema and epoch >= ema_warmup
        if ema_ready:
            ema.apply(model)

        model.eval()
        correct = total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(device), tgts.to(device)
                with autocast(device_type="cuda"):
                    out = model(imgs)
                val_loss_sum += criterion(out, tgts).item()
                correct += (out.argmax(1) == tgts).sum().item()
                total += tgts.size(0)
        val_loss = val_loss_sum / len(val_loader)
        epoch_score = 100.0 * correct / total

        if epoch_score > best_score:
            best_score = epoch_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ema_ready:
            ema.restore(model)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info("  Epoch %d/%d, train=%.4f, val=%.2f%%, best=%.2f%%",
                        epoch + 1, epochs, train_loss, epoch_score, best_score)

    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_dir = Path("/home/ubuntu/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backbone = arch.get("backbone", "unknown")
    ckpt_path = ckpt_dir / f"nas_{backbone}_trial{trial_id}.pt"
    if best_state:
        torch.save(best_state, ckpt_path)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "status": "ok",
        "trial_id": trial_id,
        "score": round(best_score, 2),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "elapsed_s": round(elapsed, 1),
        "arch": arch,
        "config": config,
        "params_m": round(param_count, 1),
        "checkpoint_path": str(ckpt_path),
    }


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="NAS Training Worker")
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.job) as f:
        job = json.load(f)

    logger.info("Job: trial=%d, backbone=%s",
                job.get("trial_id", 0), job.get("arch", {}).get("backbone", "?"))

    try:
        result = run_training(
            task=job["task"], arch=job["arch"],
            config=job["config"], trial_id=job["trial_id"],
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
