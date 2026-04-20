#!/usr/bin/env python3
"""Training worker — runs on AWS GPU instance.

Supports advanced training: Mixup, CutMix, RandomErasing, label smoothing,
warmup, architecture modification, differential LR, mixed precision.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_worker")

import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """Lightweight self-attention block to inject into ResNet."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)  # B, HW, C
        tokens = self.norm(tokens)
        out, _ = self.attn(tokens, tokens, tokens)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
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
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


def modify_resnet_architecture(model, config: dict, num_classes: int):
    """Modify ResNet-50 architecture based on config.

    Supports:
    - width_mult: scale channel widths (e.g., 1.5 for wider)
    - add_se: inject SE blocks after each bottleneck
    - add_attention: inject self-attention after layer3/layer4
    - head_type: 'mlp_head' for MLP classifier, 'linear' for standard
    - drop_rate: dropout rate in head
    """
    import torch.nn as nn

    add_se = config.get("add_se", False)
    add_attention = config.get("add_attention", False)
    head_type = config.get("adapter", "none")
    drop_rate = config.get("drop_rate", 0.0)

    # Inject SE blocks as registered submodules (so .to(device) moves them)
    if add_se:
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, name, None)
            if layer is None:
                continue
            for i, block in enumerate(layer):
                channels = block.conv3.out_channels if hasattr(block, "conv3") else block.conv2.out_channels
                se = SEBlock(channels)
                # Register as submodule so .to(device) works
                block.add_module(f"se_block", se)
                original_forward = block.forward

                def make_se_forward(orig, blk):
                    def new_forward(x):
                        out = orig(x)
                        return blk.se_block(out)
                    return new_forward

                block.forward = make_se_forward(original_forward, block)
        logger.info("  SE blocks injected into all layers")

    # Inject self-attention after layer4 as registered submodule
    if add_attention:
        for layer_name in ["layer4"]:
            layer = getattr(model, layer_name, None)
            if layer is None:
                continue
            last_block = layer[-1]
            channels = last_block.conv3.out_channels if hasattr(last_block, "conv3") else last_block.conv2.out_channels
            attn = SelfAttentionBlock(channels, num_heads=4)
            last_block.add_module("self_attn_block", attn)
            original_forward = last_block.forward

            def make_attn_forward(orig, blk):
                def new_forward(x):
                    out = orig(x)
                    return blk.self_attn_block(out)
                return new_forward

            last_block.forward = make_attn_forward(original_forward, last_block)
        logger.info("  Self-attention block injected into layer4")

    # Replace classifier head
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
    elif hasattr(model, "head") and hasattr(model.head, "in_features"):
        in_features = model.head.in_features
    else:
        in_features = model.num_features

    if head_type == "mlp_head":
        new_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(drop_rate or 0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop_rate or 0.2),
            nn.Linear(512, num_classes),
        )
    elif head_type == "lora":
        new_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(drop_rate or 0.3),
            nn.Linear(256, num_classes),
        )
    else:
        new_head = nn.Linear(in_features, num_classes)

    if hasattr(model, "fc"):
        model.fc = new_head
    elif hasattr(model, "head"):
        model.head = new_head

    return model


def build_model(base_model: str, num_classes: int, config: dict, device):
    """Build model with optional architecture modifications."""
    import timm

    freeze = config.get("freeze_backbone", True)
    adapter = config.get("adapter", "none")
    drop_rate = config.get("drop_rate", 0.0)
    drop_path = config.get("drop_path_rate", 0.0)

    if freeze:
        model = timm.create_model(base_model, pretrained=True, num_classes=0)
        embed_dim = model.num_features
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        if adapter == "mlp_head":
            head = nn.Sequential(
                nn.Linear(embed_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.GELU(),
                nn.Dropout(drop_rate or 0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(drop_rate or 0.2),
                nn.Linear(512, num_classes),
            ).to(device)
        elif adapter == "lora":
            head = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(drop_rate or 0.3),
                nn.Linear(256, num_classes),
            ).to(device)
        else:
            head = nn.Linear(embed_dim, num_classes).to(device)

        return model, head, embed_dim
    else:
        # Full fine-tuning with architecture modification
        model = timm.create_model(
            base_model, pretrained=True, num_classes=num_classes,
            drop_rate=drop_rate, drop_path_rate=drop_path,
        )

        # Apply architecture modifications (SE, attention, head replacement)
        is_resnet = "resnet" in base_model.lower()
        if is_resnet:
            model = modify_resnet_architecture(model, config, num_classes)
            logger.info("  Architecture mods: SE=%s, Attn=%s, Head=%s",
                        config.get("add_se"), config.get("add_attention"), adapter)

        model = model.to(device)
        embed_dim = num_classes
        return model, None, embed_dim


def build_loaders(task: dict, config: dict):
    """Build data loaders with advanced augmentation."""
    import torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    img_size = config.get("img_size", 224)
    bs = config.get("batch_size", 128)
    use_randaug = config.get("randaugment", False)
    use_erasing = config.get("random_erasing", False)

    # Train transforms
    train_tfms = [
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
    ]
    if use_randaug:
        train_tfms.append(T.RandAugment(num_ops=2, magnitude=9))
    else:
        train_tfms.extend([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])
    train_tfms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    if use_erasing:
        train_tfms.append(T.RandomErasing(p=0.25))

    val_tfms = [
        T.Resize(int(img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ]

    data_path = Path(task["data_path"])
    train_ds = ImageFolder(data_path / "train", T.Compose(train_tfms))
    val_ds = ImageFolder(data_path / "val", T.Compose(val_tfms))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation."""
    import torch
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation."""
    import torch
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape

    cut_ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy = torch.randint(0, H, (1,)).item()
    cx = torch.randint(0, W, (1,)).item()
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)

    x_clone = x.clone()
    x_clone[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_clone, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        """Replace model params with EMA params for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.cpu().clone() for k, v in self.shadow.items()}


def extract_features(model, loader, device):
    """Extract features from frozen backbone."""
    import torch
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = imgs.to(device)
            feat = model(imgs)
            features.append(feat.cpu())
            labels.append(tgts)
    return torch.cat(features), torch.cat(labels)


def run_training(base_model: str, task: dict, config: dict, trial_id: int) -> dict:
    """Run training with all advanced techniques."""
    import torch
    import torch.nn as nn
    from torch.amp import GradScaler, autocast

    # Use bfloat16 if GPU supports it (Ampere+: A10G, A100, H100).
    # bf16 has fp32-like exponent range → no overflow → no NaN collapse
    # on long training runs. fp16 overflows after ~10 epochs on large models.
    _use_bf16 = torch.cuda.is_bf16_supported()
    _amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16
    if _use_bf16:
        logger.info("  Using bfloat16 mixed precision (no GradScaler needed)")

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = task.get("num_classes", 100)

    freeze = config.get("freeze_backbone", True)
    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 30)
    weight_decay = config.get("weight_decay", 0.01)
    label_smoothing = config.get("label_smoothing", 0.1)
    warmup_epochs = config.get("warmup_epochs", 5)
    use_mixup = config.get("mixup", False)
    mixup_alpha = config.get("mixup_alpha", 0.2)
    use_cutmix = config.get("cutmix", False)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)
    backbone_lr_scale = config.get("backbone_lr_scale", 0.1)
    use_ema = config.get("ema", False)
    ema_decay = config.get("ema_decay", 0.999)

    model, head, embed_dim = build_model(base_model, num_classes, config, device)
    train_loader, val_loader = build_loaders(task, config)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler(enabled=not _use_bf16)  # bf16 doesn't need scaler

    if freeze:
        # --- Feature extraction mode ---
        logger.info("Feature extraction mode (frozen backbone)")
        train_feat, train_labels = extract_features(model, train_loader, device)
        val_feat, val_labels = extract_features(model, val_loader, device)

        feat_ds = torch.utils.data.TensorDataset(train_feat, train_labels)
        feat_loader = torch.utils.data.DataLoader(feat_ds, batch_size=256, shuffle=True, drop_last=True)
        val_feat_ds = torch.utils.data.TensorDataset(val_feat, val_labels)
        val_feat_loader = torch.utils.data.DataLoader(val_feat_ds, batch_size=256)

        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_loss = 0.0
        best_score = 0.0
        best_state = None
        for epoch in range(epochs):
            head.train()
            epoch_loss = 0.0
            for feat, tgt in feat_loader:
                feat, tgt = feat.to(device), tgt.to(device)
                out = head(feat)
                loss = criterion(out, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            train_loss = epoch_loss / len(feat_loader)

            # Eval
            head.eval()
            correct = total = 0
            with torch.no_grad():
                for feat, tgt in val_feat_loader:
                    feat, tgt = feat.to(device), tgt.to(device)
                    correct += (head(feat).argmax(1) == tgt).sum().item()
                    total += tgt.size(0)
            epoch_score = 100.0 * correct / total
            if epoch_score > best_score:
                best_score = epoch_score
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logger.info("  Epoch %d/%d, loss=%.4f, val=%.2f%%, best=%.2f%%",
                            epoch + 1, epochs, train_loss, epoch_score, best_score)

        score = best_score
        val_loss = train_loss  # approximate
        ckpt_state = best_state or head.state_dict()
    else:
        # --- Full fine-tuning mode ---
        logger.info("Full fine-tuning with advanced techniques")
        train_model = model

        # EMA
        ema = EMA(train_model, decay=ema_decay) if use_ema else None
        if ema:
            logger.info("  EMA enabled (decay=%.4f)", ema_decay)

        # Differential LR
        if hasattr(train_model, 'fc'):
            head_params = list(train_model.fc.parameters())
            head_ids = {id(p) for p in head_params}
            backbone_params = [p for p in train_model.parameters() if id(p) not in head_ids]
            param_groups = [
                {"params": backbone_params, "lr": lr * backbone_lr_scale},
                {"params": head_params, "lr": lr},
            ]
        elif hasattr(train_model, 'head'):
            head_params = list(train_model.head.parameters())
            head_ids = {id(p) for p in head_params}
            backbone_params = [p for p in train_model.parameters() if id(p) not in head_ids]
            param_groups = [
                {"params": backbone_params, "lr": lr * backbone_lr_scale},
                {"params": head_params, "lr": lr},
            ]
        else:
            param_groups = [{"params": train_model.parameters(), "lr": lr}]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # LR Scheduling strategy
        lr_strategy = config.get("lr_schedule", "warmup_cosine")
        min_lr = config.get("min_lr", 1e-6)
        total_steps = epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)
        step_per_batch = True  # step scheduler every batch

        if lr_strategy == "onecycle":
            # OneCycleLR: ramp up then down, very effective for fine-tuning
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[lr * backbone_lr_scale, lr] if len(param_groups) > 1 else [lr],
                total_steps=total_steps,
                pct_start=config.get("onecycle_pct_start", 0.3),
                anneal_strategy="cos",
                div_factor=config.get("onecycle_div_factor", 25),
                final_div_factor=config.get("onecycle_final_div_factor", 1e4),
            )
            logger.info("  LR schedule: OneCycleLR (max_lr=%s, pct_start=%.1f)",
                        lr, config.get("onecycle_pct_start", 0.3))

        elif lr_strategy == "cosine_restarts":
            # Cosine Annealing with Warm Restarts (SGDR)
            restart_period = config.get("restart_period", max(epochs // 3, 10))
            restart_mult = config.get("restart_mult", 2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=restart_period, T_mult=restart_mult, eta_min=min_lr,
            )
            step_per_batch = False  # step per epoch
            logger.info("  LR schedule: CosineWarmRestarts (T0=%d, Tmult=%d)",
                        restart_period, restart_mult)

        elif lr_strategy == "cosine_min":
            # Warmup + Cosine with min_lr floor
            def lr_lambda(step):
                if step < warmup_steps and warmup_steps > 0:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                return max(cosine, min_lr / lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            logger.info("  LR schedule: Warmup+Cosine (min_lr=%s)", min_lr)

        else:
            # Default: warmup + cosine with floor.
            # Auto-scale warmup if too short relative to total epochs
            # (avoids instability from abrupt warmup→decay transition in long runs).
            if warmup_epochs > 0 and warmup_epochs < max(2, int(epochs * 0.1)):
                warmup_epochs = max(2, int(epochs * 0.1))
                warmup_steps = warmup_epochs * steps_per_epoch
                logger.info("  [auto-adjust] warmup scaled to %d epochs (10%% of %d)",
                            warmup_epochs, epochs)
            # LR floor: never decay below 1e-7 (prevents NaN from near-zero updates)
            lr_floor = max(1e-7, lr * 0.01)
            def lr_lambda(step):
                if step < warmup_steps and warmup_steps > 0:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                return max(cosine, lr_floor / lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            logger.info("  LR schedule: Warmup+Cosine (warmup=%d ep, floor=%.1e)", warmup_epochs, lr_floor)

        train_loss = 0.0
        val_loss = 0.0
        best_score = 0.0
        best_state = None

        for epoch in range(epochs):
            train_model.train()
            epoch_loss = 0.0
            for imgs, tgts in train_loader:
                imgs, tgts = imgs.to(device), tgts.to(device)

                # Apply Mixup or CutMix
                apply_mix = use_mixup or use_cutmix
                if apply_mix:
                    r = torch.rand(1).item()
                    if use_cutmix and r < 0.5:
                        imgs, tgts_a, tgts_b, lam = cutmix_data(imgs, tgts, cutmix_alpha)
                    elif use_mixup:
                        imgs, tgts_a, tgts_b, lam = mixup_data(imgs, tgts, mixup_alpha)
                    else:
                        tgts_a, tgts_b, lam = tgts, tgts, 1.0

                with autocast(device_type="cuda", dtype=_amp_dtype):
                    out = train_model(imgs)
                    if apply_mix:
                        loss = mixup_criterion(criterion, out, tgts_a, tgts_b, lam)
                    else:
                        loss = criterion(out, tgts)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if step_per_batch:
                    scheduler.step()
                epoch_loss += loss.item()

            if not step_per_batch:
                scheduler.step(epoch)
            train_loss = epoch_loss / len(train_loader)

            # NaN guard: detect diverged training and halt early.
            if math.isnan(train_loss) or math.isinf(train_loss):
                logger.warning("  NaN/Inf detected at epoch %d — halting training", epoch + 1)
                break

            # Update EMA after each epoch
            if ema:
                ema.update(train_model)

            # Evaluate (use EMA weights only after warmup period to let shadow stabilize)
            ema_ready = ema and epoch >= max(warmup_epochs, 10)
            if ema_ready:
                ema.apply(train_model)
            train_model.eval()
            correct = total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for imgs, tgts in val_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    with autocast(device_type="cuda", dtype=_amp_dtype):
                        out = train_model(imgs)
                    val_loss_sum += criterion(out, tgts).item()
                    correct += (out.argmax(1) == tgts).sum().item()
                    total += tgts.size(0)
            val_loss = val_loss_sum / len(val_loader)
            epoch_score = 100.0 * correct / total

            if epoch_score > best_score:
                best_score = epoch_score
                if ema:
                    best_state = {k: v.cpu().clone() for k, v in train_model.state_dict().items()}
                else:
                    best_state = {k: v.cpu().clone() for k, v in train_model.state_dict().items()}

            if ema_ready:
                ema.restore(train_model)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                logger.info("  Epoch %d/%d, train=%.4f, val=%.2f%%, best=%.2f%%",
                            epoch + 1, epochs, train_loss, epoch_score, best_score)

            # Write per-epoch progress file for Controller early-stop monitoring.
            import os as _os2
            _prog_path = _os2.environ.get("ALCHEMIST_PROGRESS_PATH")
            if _prog_path:
                try:
                    import json as _json
                    _prog = {
                        "epoch": int(epoch + 1),
                        "total_epochs": int(epochs),
                        "train_loss": float(train_loss),
                        "val_acc": float(epoch_score),
                        "best_so_far": float(best_score),
                        "elapsed_s": float(time.time() - t0),
                    }
                    with open(_prog_path, "w") as _pf:
                        _json.dump(_prog, _pf)
                except Exception:
                    pass

        score = best_score
        ckpt_state = best_state or train_model.state_dict()

    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_dir = Path("/home/ubuntu/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{task['name']}_{base_model.replace('.', '_')}_trial{trial_id}.pt"
    torch.save(ckpt_state, ckpt_path)

    return {
        "status": "ok",
        "trial_id": trial_id,
        "score": round(score, 2),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "elapsed_s": round(elapsed, 1),
        "config": config,
        "checkpoint_path": str(ckpt_path),
    }


def run_baseline(base_model: str, task: dict) -> dict:
    config = {
        "freeze_backbone": True,
        "adapter": "linear_head",
        "lr": 0.01,
        "epochs": 5,
        "batch_size": 128,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
    }
    result = run_training(base_model, task, config, trial_id=0)
    result["command"] = "baseline"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Alchemist Training Worker")
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress",
                        help="Path to write per-epoch progress JSON for Controller early-stop.")
    args = parser.parse_args()

    if args.progress:
        # Make the progress path available to run_training via env var
        import os as _os
        _os.environ["ALCHEMIST_PROGRESS_PATH"] = args.progress

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"Job file not found: {job_path}", file=sys.stderr)
        return 1

    with open(job_path) as f:
        job = json.load(f)

    logger.info("Job: %s (model=%s, task=%s)",
                job.get("command"), job.get("base_model"), job.get("task", {}).get("name"))

    try:
        if job.get("command") == "baseline":
            result = run_baseline(job["base_model"], job["task"])
        else:
            result = run_training(job["base_model"], job["task"], job["config"], job["trial_id"])
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
