"""ModelLoader — load vision models from multiple registries.

Supports:
  1. timm: timm.create_model(name, pretrained=True)
  2. torch.hub: torch.hub.load('owner/repo', 'model_name')
  3. GitHub direct: clone repo + import model module dynamically

This bridges the gap between "Benchmark Agent found a model on GitHub"
and "train_worker can actually load and train it."
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "alchemist" / "model_repos"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ModelLoader:
    """Unified model loader across timm, torch.hub, and GitHub repos."""

    @staticmethod
    def load(
        model_id: str,
        num_classes: int = 100,
        pretrained: bool = True,
        **kwargs,
    ) -> nn.Module:
        """Load a model from the best available source.

        Args:
            model_id: one of:
                - timm model name (e.g., "swinv2_base_window12to16_192to256")
                - torch.hub spec (e.g., "github:owner/repo:model_fn")
                - GitHub URL (e.g., "https://github.com/owner/repo")
            num_classes: number of output classes
            pretrained: load pretrained weights
        """
        # 1. Try timm first (fastest, most reliable)
        model = ModelLoader._try_timm(model_id, num_classes, pretrained, **kwargs)
        if model is not None:
            return model

        # 2. Try torch.hub (github:owner/repo:entry_point format)
        if model_id.startswith("github:") or "/" in model_id:
            model = ModelLoader._try_torch_hub(model_id, num_classes, pretrained, **kwargs)
            if model is not None:
                return model

        # 3. Try GitHub clone + dynamic import
        if "/" in model_id:
            model = ModelLoader._try_github_clone(model_id, num_classes, pretrained, **kwargs)
            if model is not None:
                return model

        raise RuntimeError(
            f"Could not load model '{model_id}' from any source "
            f"(timm, torch.hub, GitHub). Check model name or repo URL."
        )

    @staticmethod
    def _try_timm(
        model_id: str, num_classes: int, pretrained: bool, **kwargs,
    ) -> nn.Module | None:
        """Try loading from timm registry."""
        try:
            import timm
            known = set(timm.list_models())
            # Try exact match
            if model_id in known:
                model = timm.create_model(
                    model_id, pretrained=pretrained, num_classes=num_classes, **kwargs,
                )
                logger.info("[ModelLoader] loaded from timm: %s", model_id)
                return model
            # Try base name (without variant suffix)
            base = model_id.split(".")[0]
            if base in known:
                model = timm.create_model(
                    base, pretrained=pretrained, num_classes=num_classes, **kwargs,
                )
                logger.info("[ModelLoader] loaded from timm (base): %s", base)
                return model
        except Exception as e:
            logger.debug("[ModelLoader] timm failed for %s: %s", model_id, e)
        return None

    @staticmethod
    def _try_torch_hub(
        model_id: str, num_classes: int, pretrained: bool, **kwargs,
    ) -> nn.Module | None:
        """Try loading via torch.hub.load().

        Formats:
          - "github:owner/repo:entry_point"
          - "owner/repo" (auto-detect entry point from hubconf.py)
        """
        try:
            if model_id.startswith("github:"):
                parts = model_id.replace("github:", "").split(":")
                repo = parts[0]
                entry = parts[1] if len(parts) > 1 else None
            elif "/" in model_id and not model_id.startswith("http"):
                # Could be "owner/repo" or "owner/repo:entry"
                if ":" in model_id:
                    repo, entry = model_id.split(":", 1)
                else:
                    repo = model_id
                    entry = None
            else:
                return None

            # If no entry point specified, try to discover from hubconf.py
            if entry is None:
                entry = ModelLoader._discover_hub_entry(repo)
                if entry is None:
                    logger.debug("[ModelLoader] no hubconf entry found for %s", repo)
                    return None

            logger.info("[ModelLoader] loading from torch.hub: %s / %s", repo, entry)
            model = torch.hub.load(
                repo, entry,
                pretrained=pretrained,
                num_classes=num_classes,
                trust_repo=True,
                **kwargs,
            )
            logger.info("[ModelLoader] loaded from torch.hub: %s:%s", repo, entry)
            return model

        except Exception as e:
            logger.debug("[ModelLoader] torch.hub failed for %s: %s", model_id, e)
        return None

    @staticmethod
    def _discover_hub_entry(repo: str) -> str | None:
        """Try to discover the entry point function from a repo's hubconf.py."""
        try:
            # Download hubconf.py content via GitHub raw URL
            import urllib.request
            owner_repo = repo.replace("github:", "")
            url = f"https://raw.githubusercontent.com/{owner_repo}/main/hubconf.py"
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    content = resp.read().decode()
            except Exception:
                # Try master branch
                url = f"https://raw.githubusercontent.com/{owner_repo}/master/hubconf.py"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    content = resp.read().decode()

            # Parse function definitions — look for model-like functions
            import re
            functions = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
            # Filter out private/utility functions
            model_fns = [f for f in functions
                         if not f.startswith('_') and f not in ('dependencies',)]
            if model_fns:
                logger.info("[ModelLoader] discovered hub entries: %s", model_fns[:5])
                return model_fns[0]  # Return first model function
        except Exception as e:
            logger.debug("[ModelLoader] hubconf discovery failed for %s: %s", repo, e)
        return None

    @staticmethod
    def _try_github_clone(
        model_id: str, num_classes: int, pretrained: bool, **kwargs,
    ) -> nn.Module | None:
        """Clone a GitHub repo and try to import the model dynamically.

        Looks for common patterns:
          - models/model.py with a build_model() or create_model() function
          - model.py at repo root
          - __init__.py with model registration
        """
        try:
            owner_repo = model_id.replace("https://github.com/", "").rstrip("/")
            if "/" not in owner_repo:
                return None

            repo_dir = _CACHE_DIR / owner_repo.replace("/", "_")

            # Clone if not cached
            if not repo_dir.exists():
                logger.info("[ModelLoader] cloning %s ...", owner_repo)
                result = subprocess.run(
                    ["git", "clone", "--depth", "1",
                     f"https://github.com/{owner_repo}.git", str(repo_dir)],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    logger.warning("[ModelLoader] git clone failed: %s", result.stderr[:200])
                    return None

            # Add repo to Python path temporarily
            sys.path.insert(0, str(repo_dir))

            # Try common model loading patterns
            model = None

            # Pattern 1: hubconf.py exists
            hubconf = repo_dir / "hubconf.py"
            if hubconf.exists():
                model = torch.hub.load(
                    str(repo_dir), "model",
                    source="local", pretrained=pretrained,
                    trust_repo=True,
                )

            # Pattern 2: models/ directory with build function
            if model is None:
                for candidate in ["models/model.py", "model.py", "models/__init__.py"]:
                    fpath = repo_dir / candidate
                    if fpath.exists():
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("_repo_model", str(fpath))
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)

                        # Try common function names
                        for fn_name in ("build_model", "create_model", "get_model",
                                         "astroformer", "model"):
                            fn = getattr(mod, fn_name, None)
                            if callable(fn):
                                try:
                                    model = fn(num_classes=num_classes, pretrained=pretrained)
                                    break
                                except Exception:
                                    try:
                                        model = fn(num_classes=num_classes)
                                        break
                                    except Exception:
                                        continue
                    if model is not None:
                        break

            # Clean up sys.path
            sys.path.remove(str(repo_dir))

            if model is not None:
                logger.info("[ModelLoader] loaded from GitHub clone: %s", owner_repo)
                return model

        except Exception as e:
            logger.debug("[ModelLoader] GitHub clone failed for %s: %s", model_id, e)
        return None

    @staticmethod
    def resolve_model_info(model_id: str) -> dict[str, Any]:
        """Check where a model can be loaded from without actually loading it.

        Returns dict with:
          source: "timm" | "torch_hub" | "github_clone" | "unavailable"
          loadable: bool
          entry_point: str (for torch.hub)
          repo_url: str (for GitHub)
        """
        info: dict[str, Any] = {
            "model_id": model_id,
            "source": "unavailable",
            "loadable": False,
        }

        # Check timm
        try:
            import timm
            known = set(timm.list_models())
            base = model_id.split(".")[0]
            if model_id in known or base in known:
                info["source"] = "timm"
                info["loadable"] = True
                return info
        except Exception:
            pass

        # Check torch.hub
        if "/" in model_id:
            entry = ModelLoader._discover_hub_entry(
                model_id.replace("github:", "").split(":")[0]
            )
            if entry:
                info["source"] = "torch_hub"
                info["loadable"] = True
                info["entry_point"] = entry
                return info

        # Check GitHub clone feasibility
        if "/" in model_id:
            owner_repo = model_id.replace("https://github.com/", "").rstrip("/")
            try:
                import urllib.request
                url = f"https://api.github.com/repos/{owner_repo}"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Alchemist-Vision-Agent/1.0"
                })
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        info["source"] = "github_clone"
                        info["loadable"] = True
                        info["repo_url"] = f"https://github.com/{owner_repo}"
                        return info
            except Exception:
                pass

        return info
