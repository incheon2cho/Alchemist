"""V-JEPA Model Loader — load pretrained V-JEPA models from facebookresearch/jepa.

V-JEPA (Video JEPA) provides self-supervised ViT models that can be used
as strong image classification backbones. Since the repo lacks hubconf.py,
this module handles cloning, importing, and weight loading directly.

Pretrained weights (from README):
  - ViT-L/16: https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar
  - ViT-H/16: https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
  - ViT-H/16-384: https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar

Usage::

    from alchemist.core.vjepa_loader import load_vjepa
    model = load_vjepa("vit_huge", num_classes=100, img_size=224)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "alchemist" / "model_repos"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VJEPA_REPO = "facebookresearch/jepa"
VJEPA_DIR = _CACHE_DIR / "facebookresearch_jepa"

# Pretrained weight URLs from V-JEPA README
VJEPA_WEIGHTS = {
    "vit_large": "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar",
    "vit_huge": "https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar",
    "vit_huge_384": "https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar",
}

# Model configs matching facebookresearch/jepa/src/models/vision_transformer.py
VJEPA_CONFIGS = {
    "vit_large": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    "vit_huge": dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16),
    "vit_huge_384": dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16),
}


def _ensure_repo_cloned() -> Path:
    """Clone the V-JEPA repo if not cached."""
    if not VJEPA_DIR.exists():
        logger.info("[V-JEPA] Cloning facebookresearch/jepa ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             f"https://github.com/{VJEPA_REPO}.git", str(VJEPA_DIR)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone V-JEPA repo: {result.stderr[:300]}")
    return VJEPA_DIR


def _download_weights(variant: str) -> Path:
    """Download pretrained weights if not cached."""
    url = VJEPA_WEIGHTS.get(variant)
    if url is None:
        raise ValueError(f"Unknown V-JEPA variant: {variant}. Available: {list(VJEPA_WEIGHTS.keys())}")

    cache_path = _CACHE_DIR / f"vjepa_{variant}.pth.tar"
    if cache_path.exists():
        return cache_path

    logger.info("[V-JEPA] Downloading weights for %s ...", variant)
    import urllib.request
    urllib.request.urlretrieve(url, str(cache_path))
    logger.info("[V-JEPA] Downloaded: %s (%.1f MB)", cache_path.name, cache_path.stat().st_size / 1e6)
    return cache_path


def load_vjepa(
    variant: str = "vit_huge",
    num_classes: int = 100,
    img_size: int = 224,
    pretrained: bool = True,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Load a V-JEPA model with optional pretrained weights + classification head.

    Args:
        variant: "vit_large" | "vit_huge" | "vit_huge_384"
        num_classes: number of output classes
        img_size: input image size (224 or 384)
        pretrained: load pretrained V-JEPA weights
        drop_path_rate: stochastic depth rate

    Returns:
        nn.Module with .forward(x) → (batch, num_classes) logits
    """
    repo_dir = _ensure_repo_cloned()

    # Add repo src to path for import
    src_dir = str(repo_dir / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Import the VisionTransformer and builder functions
    try:
        from models.vision_transformer import VisionTransformer
        import models.vision_transformer as vit_module
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import V-JEPA VisionTransformer: {e}. "
            f"Ensure {repo_dir}/src/models/vision_transformer.py exists."
        )

    # Get the builder function (vit_large, vit_huge, etc.)
    builder_name = variant.replace("_384", "")  # vit_huge_384 → vit_huge
    builder = getattr(vit_module, builder_name, None)
    if builder is None:
        raise ValueError(
            f"V-JEPA builder '{builder_name}' not found. "
            f"Available: {[f for f in dir(vit_module) if f.startswith('vit_')]}"
        )

    # Build the model
    model = builder(
        img_size=[img_size],
        drop_path_rate=drop_path_rate,
    )
    embed_dim = VJEPA_CONFIGS.get(variant, {}).get("embed_dim", 1280)
    logger.info("[V-JEPA] Built %s (embed_dim=%d, img_size=%d)", variant, embed_dim, img_size)

    # Load pretrained weights
    if pretrained:
        weight_path = _download_weights(variant)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

        # V-JEPA checkpoints store weights under 'target_encoder' or 'encoder'
        state_dict = None
        for key in ["target_encoder", "encoder", "model", "state_dict"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint

        # Clean up key prefixes (e.g., "module." from DDP)
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("backbone.", "")
            cleaned[k] = v

        # Load weights (ignore missing classification head)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        logger.info(
            "[V-JEPA] Loaded pretrained weights: %d matched, %d missing, %d unexpected",
            len(cleaned) - len(unexpected), len(missing), len(unexpected),
        )

    # Add classification head (V-JEPA encoder doesn't have one)
    model.head = nn.Linear(embed_dim, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    # Wrap forward to handle V-JEPA's output format
    original_forward = model.forward

    def classification_forward(x):
        # V-JEPA forward may return features without head
        features = original_forward(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        # Global average pooling if needed (B, N, D) → (B, D)
        if features.dim() == 3:
            features = features[:, 0]  # CLS token or features.mean(dim=1)
        return model.head(features)

    model.forward = classification_forward
    logger.info("[V-JEPA] Added classification head (%d → %d)", embed_dim, num_classes)

    # Remove from sys.path
    if src_dir in sys.path:
        sys.path.remove(src_dir)

    return model


# Register V-JEPA in ModelLoader
def register_vjepa_in_model_loader():
    """Patch ModelLoader to recognize V-JEPA model IDs."""
    try:
        from alchemist.core.model_loader import ModelLoader

        _original_load = ModelLoader.load.__func__

        @staticmethod
        def load_with_vjepa(model_id, num_classes=100, pretrained=True, **kwargs):
            # Check if this is a V-JEPA model
            if model_id.lower().startswith("vjepa_") or model_id.lower().startswith("v-jepa"):
                variant = model_id.lower().replace("vjepa_", "").replace("v-jepa_", "").replace("v-jepa-", "")
                variant_map = {
                    "vitl16": "vit_large", "vit_large": "vit_large", "vitl": "vit_large",
                    "vith16": "vit_huge", "vit_huge": "vit_huge", "vith": "vit_huge",
                    "vith16_384": "vit_huge_384", "vit_huge_384": "vit_huge_384",
                }
                v = variant_map.get(variant, variant)
                img_size = 384 if "384" in variant else 224
                return load_vjepa(
                    v, num_classes=num_classes, img_size=img_size,
                    pretrained=pretrained,
                    drop_path_rate=kwargs.get("drop_path_rate", 0.0),
                )
            return _original_load(model_id, num_classes=num_classes, pretrained=pretrained, **kwargs)

        ModelLoader.load = load_with_vjepa
        logger.info("[V-JEPA] Registered in ModelLoader")
    except ImportError:
        pass
