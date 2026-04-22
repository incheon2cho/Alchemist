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
    num_frames: int = 16,
    tubelet_size: int = 2,
) -> nn.Module:
    """Load a V-JEPA model with optional pretrained weights + classification head.

    Args:
        variant: "vit_large" | "vit_huge" | "vit_huge_384"
        num_classes: number of output classes
        img_size: input image size (224 or 384)
        pretrained: load pretrained V-JEPA weights
        drop_path_rate: stochastic depth rate
        num_frames: number of video frames (16 for pretrained checkpoint)
        tubelet_size: temporal patch size (2 for pretrained checkpoint)

    Returns:
        nn.Module with .forward(x) → (batch, num_classes) logits
    """
    repo_dir = _ensure_repo_cloned()

    # Add repo root AND repo src to path for import
    # V-JEPA uses both `from src.models.xxx` and `from models.xxx`
    repo_root = str(repo_dir)
    src_dir = str(repo_dir / "src")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
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

    # Build the model — V-JEPA expects img_size as int, not list
    # Must pass num_frames and tubelet_size to match pretrained checkpoint
    model = builder(
        img_size=img_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
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
        # Global average pooling over all patch tokens (B, N, D) → (B, D)
        # V-JEPA is self-supervised — CLS token is NOT optimized for classification.
        # Mean pooling over all tokens gives much better linear probe accuracy.
        if features.dim() == 3:
            features = features.mean(dim=1)
        return model.head(features)

    model.forward = classification_forward
    logger.info("[V-JEPA] Added classification head (%d → %d)", embed_dim, num_classes)

    # Remove from sys.path
    if src_dir in sys.path:
        sys.path.remove(src_dir)

    return model


# ---------------------------------------------------------------------------
# V-JEPA 2 — HuggingFace-based loader
# ---------------------------------------------------------------------------

# V-JEPA 2 model variants on HuggingFace
VJEPA2_MODELS = {
    "vjepa2_vitb": "facebook/vjepa2-vitb-fpc64-256",
    "vjepa2_vitl": "facebook/vjepa2-vitl-fpc64-256",
    "vjepa2_vitg": "facebook/vjepa2-vitg-fpc64-256",
    "vjepa2_vitG": "facebook/vjepa2-vitG-fpc64-384",
    # V-JEPA 2.1 (higher resolution)
    "vjepa2.1_vitb": "facebook/vjepa2.1-vitb-fpc64-384",
    "vjepa2.1_vitl": "facebook/vjepa2.1-vitl-fpc64-384",
    "vjepa2.1_vitg": "facebook/vjepa2.1-vitg-fpc64-384",
    "vjepa2.1_vitG": "facebook/vjepa2.1-vitG-fpc64-384",
}

VJEPA2_EMBED_DIMS = {
    "vitb": 768, "vitl": 1024, "vitg": 1408, "vitG": 1664,
}


def load_vjepa2(
    variant: str = "vjepa2_vitl",
    num_classes: int = 101,
    pretrained: bool = True,
    for_vlm: bool = False,
) -> nn.Module:
    """Load V-JEPA 2 model from HuggingFace.

    Args:
        variant: model ID, e.g. "vjepa2_vitl", "vjepa2.1_vitg"
        num_classes: number of output classes
        pretrained: load pretrained weights
        for_vlm: if True, return raw encoder without classification head.
                 Output is (B, N, D) token features for VLM pipelines.

    Returns:
        nn.Module with .forward(x) → logits or raw features
        Input x: (B, C, T, H, W) video tensor
    """
    # Resolve HuggingFace model ID
    hf_id = VJEPA2_MODELS.get(variant)
    if hf_id is None:
        # Try direct HuggingFace ID
        if variant.startswith("facebook/"):
            hf_id = variant
        else:
            raise ValueError(
                f"Unknown V-JEPA2 variant '{variant}'. "
                f"Available: {list(VJEPA2_MODELS.keys())}"
            )

    logger.info("[V-JEPA2] Loading %s from HuggingFace...", hf_id)

    # Try torch.hub first (lighter dependency)
    # Note: V-JEPA2 hub returns (encoder, predictor) tuple — take encoder only
    try:
        hub_name = variant.replace(".", "_").replace("vjepa2_", "vjepa2_vit_").replace("vjepa2_1_", "vjepa2_1_vit_")
        # Normalize: vjepa2_vitl -> vjepa2_vit_large, etc.
        name_map = {
            "vjepa2_vit_vitl": "vjepa2_vit_large", "vjepa2_vit_vitb": "vjepa2_vit_base",
            "vjepa2_vit_vitg": "vjepa2_vit_giant", "vjepa2_vit_vitG": "vjepa2_vit_giant",
            "vjepa2_vit_vith": "vjepa2_vit_huge",
            "vjepa2_1_vit_vitl": "vjepa2_1_vit_large_384",
            "vjepa2_1_vit_vitb": "vjepa2_1_vit_base_384",
            "vjepa2_1_vit_vitg": "vjepa2_1_vit_giant_384",
            "vjepa2_1_vit_vitG": "vjepa2_1_vit_gigantic_384",
        }
        hub_name = name_map.get(hub_name, hub_name)
        result = torch.hub.load('facebookresearch/vjepa2', hub_name, pretrained=pretrained)
        # Hub returns (encoder, predictor) tuple — extract encoder
        if isinstance(result, tuple):
            model = result[0]
        else:
            model = result
        embed_dim = getattr(model, "embed_dim", 1024)
        logger.info("[V-JEPA2] Loaded via torch.hub: %s (embed_dim=%d)", hub_name, embed_dim)
    except Exception as hub_err:
        logger.info("[V-JEPA2] torch.hub failed (%s), trying HuggingFace transformers...", hub_err)
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
            # Detect embed_dim from config
            config = getattr(model, "config", None)
            embed_dim = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", 1024)
            logger.info("[V-JEPA2] Loaded via HuggingFace (embed_dim=%d)", embed_dim)
        except ImportError:
            raise RuntimeError(
                "V-JEPA2 requires either torch.hub access or `pip install transformers`. "
                f"torch.hub error: {hub_err}"
            )

    # Detect embed_dim from variant name if not found
    if embed_dim is None:
        for key, dim in VJEPA2_EMBED_DIMS.items():
            if key in variant:
                embed_dim = dim
                break
        if embed_dim is None:
            embed_dim = 1024

    model.embed_dim = embed_dim
    model.is_video = True

    # VLM mode: return raw encoder features (B, N, D) without head
    if for_vlm:
        original_forward = model.forward

        def vlm_forward(x):
            output = original_forward(x)
            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state
            elif isinstance(output, (tuple, list)):
                return output[0]
            return output

        model.forward = vlm_forward
        # Freeze all parameters for VLM use
        for p in model.parameters():
            p.requires_grad = False
        logger.info("[V-JEPA2] VLM mode: %s (frozen, embed_dim=%d)", hf_id, embed_dim)
        return model

    # Classification mode: add head + mean pooling
    model.head = nn.Linear(embed_dim, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    original_forward = model.forward

    def classification_forward(x):
        output = original_forward(x)
        if hasattr(output, "last_hidden_state"):
            features = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            features = output[0]
        else:
            features = output
        if features.dim() == 3:
            features = features.mean(dim=1)
        return model.head(features)

    model.forward = classification_forward
    logger.info("[V-JEPA2] Ready: %s → %d classes (embed_dim=%d)", hf_id, num_classes, embed_dim)
    return model


# Register V-JEPA in ModelLoader
def register_vjepa_in_model_loader():
    """Patch ModelLoader to recognize V-JEPA model IDs."""
    try:
        try:
            from model_loader import ModelLoader
        except ImportError:
            from alchemist.core.model_loader import ModelLoader

        _original_load = ModelLoader.load

        @staticmethod
        def load_with_vjepa(model_id, num_classes=100, pretrained=True, **kwargs):
            # Check V-JEPA 2 first (vjepa2_xxx or vjepa2.1_xxx)
            if model_id.lower().startswith("vjepa2"):
                return load_vjepa2(
                    model_id, num_classes=num_classes, pretrained=pretrained,
                )
            # Check V-JEPA 1
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
                    num_frames=kwargs.get("num_frames", 16),
                    tubelet_size=kwargs.get("tubelet_size", 2),
                )
            return _original_load(model_id, num_classes=num_classes, pretrained=pretrained, **kwargs)

        ModelLoader.load = load_with_vjepa
        logger.info("[V-JEPA] Registered in ModelLoader")
    except ImportError:
        pass
