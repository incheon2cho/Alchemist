"""C-Abstractor — Locality-enhanced visual token compressor.

Based on Honeybee (Cha et al., CVPR 2024). Uses convolutional bottleneck
blocks with Squeeze-Excitation to compress visual tokens while preserving
spatial locality. This bridges a vision encoder (e.g., V-JEPA 2.1) to an
LLM (e.g., Qwen) by projecting and compressing visual representations.

Architecture:
    (B, N, D_vision)
    → reshape to 2D grid (B, C, H, W)
    → L ResNet bottleneck blocks (with SE)
    → AdaptiveAvgPool2d to (h, w) where h*w = num_output_tokens
    → L ResNet bottleneck blocks
    → flatten to (B, num_output_tokens, C)
    → Linear projection to D_llm
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.squeeze(x).view(B, C)
        s = self.excite(s).view(B, C, 1, 1)
        return x * s


class ResBottleneckBlock(nn.Module):
    """ResNet bottleneck block with optional Squeeze-Excitation.

    Conv1x1(down) → BN → GELU → Conv3x3 → BN → GELU → Conv1x1(up) → BN [+ SE] + residual
    """

    def __init__(self, channels: int, bottleneck_ratio: int = 4, use_se: bool = True):
        super().__init__()
        mid = max(channels // bottleneck_ratio, 16)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return self.act(out + residual)


class CAbstractor(nn.Module):
    """C-Abstractor: compress visual tokens via convolutional bottleneck + pooling.

    Args:
        in_dim: vision encoder output dimension (e.g., 1408 for V-JEPA 2.1 ViT-g)
        out_dim: LLM hidden dimension (e.g., 2560 for Qwen3.6-35B-A3B)
        num_output_tokens: number of compressed visual tokens (e.g., 64)
        hidden_channels: intermediate channel dimension for conv blocks
        num_blocks_before: number of ResNet blocks before pooling
        num_blocks_after: number of ResNet blocks after pooling
        use_se: use Squeeze-Excitation in bottleneck blocks
    """

    def __init__(
        self,
        in_dim: int = 1408,
        out_dim: int = 2560,
        num_output_tokens: int = 64,
        hidden_channels: int = 512,
        num_blocks_before: int = 3,
        num_blocks_after: int = 3,
        use_se: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_output_tokens = num_output_tokens
        self.hidden_channels = hidden_channels

        # Compute output spatial grid (must be square root of num_output_tokens)
        self.pool_h = int(math.sqrt(num_output_tokens))
        self.pool_w = num_output_tokens // self.pool_h
        assert self.pool_h * self.pool_w == num_output_tokens, (
            f"num_output_tokens ({num_output_tokens}) must be factorizable into h*w grid"
        )

        # Input projection: (B, N, in_dim) → (B, hidden_channels, H, W)
        self.input_proj = nn.Linear(in_dim, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)

        # Pre-pooling conv blocks
        self.blocks_before = nn.Sequential(
            *[ResBottleneckBlock(hidden_channels, use_se=use_se) for _ in range(num_blocks_before)]
        )

        # Spatial compression
        self.pool = nn.AdaptiveAvgPool2d((self.pool_h, self.pool_w))

        # Post-pooling conv blocks
        self.blocks_after = nn.Sequential(
            *[ResBottleneckBlock(hidden_channels, use_se=use_se) for _ in range(num_blocks_after)]
        )

        # Output projection: hidden_channels → out_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_dim),
        )

    def _tokens_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, C) token sequence to (B, C, H, W) spatial grid.

        Pads if N is not a perfect square.
        """
        B, N, C = x.shape
        h = int(math.sqrt(N))
        w = h
        # Adjust to fit: find closest h, w such that h * w >= N
        while h * w < N:
            w += 1
        # Pad if needed
        if h * w > N:
            pad = h * w - N
            x = F.pad(x, (0, 0, 0, pad))  # pad sequence dim
        return x.permute(0, 2, 1).reshape(B, C, h, w)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Compress visual tokens.

        Args:
            visual_tokens: (B, N, D_vision) from vision encoder

        Returns:
            (B, num_output_tokens, D_llm) compressed tokens for LLM
        """
        # Project to hidden channels
        x = self.input_proj(visual_tokens)  # (B, N, hidden_channels)
        x = self.input_norm(x)

        # Reshape to spatial grid for convolution
        x = self._tokens_to_grid(x)  # (B, hidden_channels, H, W)

        # Pre-pooling conv blocks
        x = self.blocks_before(x)

        # Spatial compression
        x = self.pool(x)  # (B, hidden_channels, pool_h, pool_w)

        # Post-pooling conv blocks
        x = self.blocks_after(x)

        # Flatten and project to LLM dimension
        B = x.shape[0]
        x = x.reshape(B, self.hidden_channels, -1).permute(0, 2, 1)  # (B, num_output_tokens, C)
        x = self.output_proj(x)  # (B, num_output_tokens, out_dim)

        return x
