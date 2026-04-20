"""VisionArchModifier — architecture-agnostic module injection for any timm model.

Automatically discovers model structure via named_modules() and injects
vision optimization modules (SE, CBAM, LoRA, Adapter, Self-Attention)
at appropriate locations without hardcoding layer names.

Supports: ResNet, ConvNeXt, SwinV2, ViT, MaxViT, EfficientNet, etc.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injectable Modules
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention via global pool → FC → sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module: channel + spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ch_attn = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        x = x * ch_attn
        # Spatial attention
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        sp_attn = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        return x * sp_attn


class SelfAttention2D(nn.Module):
    """Multi-head self-attention for 2D feature maps (HxW → sequence → attn → reshape)."""

    def __init__(self, channels: int, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        # Reshape to sequence
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        seq = self.norm(seq)
        qkv = self.qkv(seq).reshape(b, h * w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        out = self.proj(out)
        # Residual + reshape back
        out = seq + out
        return out.transpose(1, 2).reshape(b, c, h, w)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for nn.Linear — injects trainable low-rank matrices
    while freezing the original weight. Works for both ViT attention projections
    and CNN 1x1 convolutions reshaped as linear.

    forward: y = W_frozen @ x + (B @ A) @ x * scaling
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


class AdapterBlock(nn.Module):
    """Bottleneck adapter (Houlsby et al., 2019) — inject after any layer.
    down-project → nonlinearity → up-project → residual.
    """

    def __init__(self, channels: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(channels, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        # Handle both (B, C, H, W) and (B, N, C) inputs
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            out = self.up(self.act(self.down(x)))
            out = (residual.flatten(2).transpose(1, 2) + out).transpose(1, 2).reshape(b, c, h, w)
            return out
        else:
            return residual + self.up(self.act(self.down(x)))


# ---------------------------------------------------------------------------
# Universal Architecture Modifier
# ---------------------------------------------------------------------------

class VisionArchModifier:
    """Architecture-agnostic module injector for any timm model.

    Usage::

        modifier = VisionArchModifier(model)
        modifier.inject_se(reduction=16)           # SE after conv stages
        modifier.inject_lora(rank=8, targets="qkv") # LoRA on attention layers
        modifier.inject_adapter(bottleneck=64)      # Adapter after transformer blocks
        modifier.inject_self_attention(num_heads=4)  # Self-attn after last conv stage
        modifier.inject_cbam()                       # CBAM after conv stages
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._analyze_structure()

    def _analyze_structure(self):
        """Discover model structure: find conv stages, transformer blocks, linear layers."""
        self.conv_stages = []     # (name, module, out_channels)
        self.transformer_blocks = []  # (name, module)
        self.linear_layers = []   # (name, module)
        self.attn_projections = []  # (name, module) — qkv/proj in attention

        for name, module in self.model.named_modules():
            # Detect conv stages (Conv2d with stride or large channels)
            if isinstance(module, nn.Conv2d) and module.out_channels >= 64:
                self.conv_stages.append((name, module, module.out_channels))

            # Detect transformer blocks (modules containing both attn and mlp/ffn)
            children = {n for n, _ in module.named_children()}
            if any(k in children for k in ("attn", "self_attn", "attention")) and \
               any(k in children for k in ("mlp", "ffn", "ls2", "drop_path2")):
                self.transformer_blocks.append((name, module))

            # Detect linear layers (for LoRA)
            if isinstance(module, nn.Linear) and module.in_features >= 64:
                self.linear_layers.append((name, module))
                # Detect attention projections specifically
                if any(k in name.lower() for k in ("qkv", "q_proj", "k_proj", "v_proj",
                                                     "in_proj", "out_proj", "proj")):
                    self.attn_projections.append((name, module))

        logger.info(
            "Arch analysis: %d conv stages, %d transformer blocks, "
            "%d linear layers, %d attn projections",
            len(self.conv_stages), len(self.transformer_blocks),
            len(self.linear_layers), len(self.attn_projections),
        )

    def inject_se(self, reduction: int = 16, max_injections: int = 4) -> int:
        """Inject SE blocks after the last N conv layers (stage boundaries)."""
        # Pick the last `max_injections` conv stages by channel size (feature-rich)
        targets = self.conv_stages[-max_injections:]
        count = 0
        for name, conv_module, channels in targets:
            se = SEBlock(channels, reduction)
            self._wrap_module_output(name, conv_module, se)
            count += 1
        if count:
            logger.info("  Injected %d SE blocks (reduction=%d)", count, reduction)
        return count

    def inject_cbam(self, reduction: int = 16, max_injections: int = 4) -> int:
        """Inject CBAM blocks after the last N conv layers."""
        targets = self.conv_stages[-max_injections:]
        count = 0
        for name, conv_module, channels in targets:
            cbam = CBAMBlock(channels, reduction)
            self._wrap_module_output(name, conv_module, cbam)
            count += 1
        if count:
            logger.info("  Injected %d CBAM blocks", count)
        return count

    def inject_self_attention(self, num_heads: int = 4, max_injections: int = 2) -> int:
        """Inject self-attention after the last N conv stages.
        Useful for adding attention to pure CNN models (ConvNeXt, ResNet, EfficientNet).
        """
        targets = self.conv_stages[-max_injections:]
        count = 0
        for name, conv_module, channels in targets:
            if channels % num_heads != 0:
                num_heads = min(num_heads, channels)
                while channels % num_heads != 0 and num_heads > 1:
                    num_heads -= 1
            attn = SelfAttention2D(channels, num_heads)
            self._wrap_module_output(name, conv_module, attn)
            count += 1
        if count:
            logger.info("  Injected %d self-attention blocks (heads=%d)", count, num_heads)
        return count

    def inject_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        targets: str = "attn",  # "attn" | "all" | "qkv"
    ) -> int:
        """Replace Linear layers with LoRA-augmented versions.

        Args:
            targets: "attn" = attention projections only, "all" = all linear layers,
                     "qkv" = only qkv projections.
        """
        if targets == "attn":
            candidates = self.attn_projections
        elif targets == "qkv":
            candidates = [(n, m) for n, m in self.attn_projections
                          if any(k in n.lower() for k in ("qkv", "q_proj", "k_proj", "v_proj", "in_proj"))]
        else:
            candidates = self.linear_layers

        count = 0
        for name, linear in candidates:
            lora = LoRALinear(linear, rank=rank, alpha=alpha)
            self._replace_module(name, lora)
            count += 1
        if count:
            logger.info("  Injected LoRA (rank=%d, alpha=%.1f) into %d layers [%s]",
                        rank, alpha, count, targets)
        return count

    def inject_adapter(self, bottleneck: int = 64, max_injections: int = 6) -> int:
        """Inject adapter blocks after transformer blocks (Houlsby-style)."""
        targets = self.transformer_blocks[-max_injections:]
        count = 0
        for name, block in targets:
            # Infer hidden dimension from block's MLP/FFN
            hidden_dim = None
            for child_name, child in block.named_modules():
                if isinstance(child, nn.Linear):
                    hidden_dim = child.in_features
                    break
            if hidden_dim is None:
                continue
            adapter = AdapterBlock(hidden_dim, bottleneck)
            self._wrap_module_output(name, block, adapter)
            count += 1
        if count:
            logger.info("  Injected %d adapter blocks (bottleneck=%d)", count, bottleneck)
        return count

    def _wrap_module_output(self, name: str, target: nn.Module, wrapper: nn.Module):
        """Wrap a module's forward so wrapper is applied to its output (residual-safe)."""
        # Register wrapper as submodule for .to(device) and state_dict
        safe_name = name.replace(".", "_") + "_injected"
        self.model.add_module(safe_name, wrapper)
        original_forward = target.forward

        def new_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            return wrapper(out)

        target.forward = new_forward

    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a named module in the model hierarchy."""
        parts = name.split(".")
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    def summary(self) -> dict[str, Any]:
        """Return summary of injectable points found."""
        return {
            "conv_stages": len(self.conv_stages),
            "transformer_blocks": len(self.transformer_blocks),
            "linear_layers": len(self.linear_layers),
            "attn_projections": len(self.attn_projections),
        }


# ---------------------------------------------------------------------------
# Helper: apply modifications from config dict
# ---------------------------------------------------------------------------

def apply_arch_modifications(model: nn.Module, config: dict) -> nn.Module:
    """Apply architecture modifications based on config keys.

    Config keys:
        add_se: bool — inject SE blocks
        add_cbam: bool — inject CBAM blocks
        add_self_attention: bool — inject self-attention (for CNNs)
        add_lora: bool — inject LoRA into attention layers
        lora_rank: int — LoRA rank (default 8)
        lora_targets: str — "attn" | "all" | "qkv"
        add_adapter: bool — inject Houlsby adapters
        adapter_bottleneck: int — adapter bottleneck dim (default 64)
    """
    if not any(config.get(k) for k in ("add_se", "add_cbam", "add_self_attention",
                                         "add_lora", "add_adapter")):
        return model

    modifier = VisionArchModifier(model)

    if config.get("add_se"):
        modifier.inject_se(reduction=config.get("se_reduction", 16))

    if config.get("add_cbam"):
        modifier.inject_cbam(reduction=config.get("cbam_reduction", 16))

    if config.get("add_self_attention"):
        modifier.inject_self_attention(
            num_heads=config.get("self_attn_heads", 4),
        )

    if config.get("add_lora"):
        modifier.inject_lora(
            rank=config.get("lora_rank", 8),
            alpha=config.get("lora_alpha", 16.0),
            targets=config.get("lora_targets", "attn"),
        )

    if config.get("add_adapter"):
        modifier.inject_adapter(
            bottleneck=config.get("adapter_bottleneck", 64),
        )

    return model
