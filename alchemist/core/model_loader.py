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
        """Clone a GitHub repo and auto-discover the model architecture.

        Fully generic — no hardcoded model names. Discovery strategy:
          1. hubconf.py → torch.hub.load(local) with auto-discovered entry
          2. Scan all .py files for nn.Module subclasses → instantiate largest
          3. Try factory functions (build_model, create_model, get_model, etc.)
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
            model = None

            # Strategy 1: hubconf.py → auto-discover entry points
            hubconf = repo_dir / "hubconf.py"
            if hubconf.exists() and model is None:
                import re
                content = hubconf.read_text(errors="ignore")
                entry_fns = [
                    fn for fn in re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
                    if not fn.startswith('_') and fn != 'dependencies'
                ]
                for fn_name in entry_fns:
                    try:
                        model = torch.hub.load(
                            str(repo_dir), fn_name,
                            source="local", pretrained=pretrained,
                            num_classes=num_classes, trust_repo=True,
                        )
                        logger.info("[ModelLoader] loaded via hubconf entry: %s", fn_name)
                        break
                    except Exception:
                        try:
                            model = torch.hub.load(
                                str(repo_dir), fn_name,
                                source="local", trust_repo=True,
                            )
                            logger.info("[ModelLoader] loaded via hubconf (no num_classes): %s", fn_name)
                            break
                        except Exception:
                            continue

            # Strategy 2: Scan .py files for nn.Module subclasses + factory functions
            if model is None:
                model = ModelLoader._scan_and_load(repo_dir, num_classes, pretrained)

            # Clean up sys.path
            if str(repo_dir) in sys.path:
                sys.path.remove(str(repo_dir))

            if model is not None:
                logger.info("[ModelLoader] loaded from GitHub clone: %s", owner_repo)
                return model

        except Exception as e:
            logger.debug("[ModelLoader] GitHub clone failed for %s: %s", model_id, e)
            # Clean up sys.path on error
            repo_dir_str = str(_CACHE_DIR / model_id.replace("https://github.com/", "").rstrip("/").replace("/", "_"))
            if repo_dir_str in sys.path:
                sys.path.remove(repo_dir_str)
        return None

    @staticmethod
    def _scan_and_load(
        repo_dir: Path, num_classes: int, pretrained: bool,
    ) -> nn.Module | None:
        """Scan repo's Python files to auto-discover model classes and factory functions.

        Strategy:
          1. Find all .py files (prioritize models/, model.py, network.py)
          2. Parse for nn.Module subclass definitions
          3. Try factory functions: build_model, create_model, get_model, make_model
          4. If no factory, instantiate the largest nn.Module subclass directly
        """
        import importlib.util
        import re

        # Prioritized file search order
        search_paths = []
        for pattern in [
            "models/*.py", "model.py", "network.py", "nets/*.py",
            "architectures/*.py", "src/models/*.py", "src/model.py",
            "*.py",  # fallback: scan all root .py files
        ]:
            search_paths.extend(sorted(repo_dir.glob(pattern)))

        # Deduplicate while preserving order
        seen_files = set()
        unique_paths = []
        for p in search_paths:
            if p.name.startswith("_") or p.name in ("setup.py", "train.py", "test.py",
                                                       "evaluate.py", "demo.py", "hubconf.py"):
                continue
            if p not in seen_files:
                seen_files.add(p)
                unique_paths.append(p)

        # Common factory function names (generic, no model-specific names)
        FACTORY_NAMES = [
            "build_model", "create_model", "get_model", "make_model",
            "build_network", "create_network", "get_network",
            "model", "net", "network",
        ]

        for fpath in unique_paths[:15]:  # limit scan to avoid slow imports
            try:
                spec = importlib.util.spec_from_file_location(
                    f"_repo_{fpath.stem}", str(fpath),
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                # Try factory functions first
                for fn_name in FACTORY_NAMES:
                    fn = getattr(mod, fn_name, None)
                    if not callable(fn):
                        continue
                    for call_args in [
                        {"num_classes": num_classes, "pretrained": pretrained},
                        {"num_classes": num_classes},
                        {"n_classes": num_classes},
                        {},
                    ]:
                        try:
                            result = fn(**call_args)
                            if isinstance(result, nn.Module):
                                logger.info(
                                    "[ModelLoader] loaded via %s:%s(%s)",
                                    fpath.name, fn_name, call_args,
                                )
                                return result
                        except Exception:
                            continue

                # Try instantiating nn.Module subclasses directly
                module_classes = []
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name, None)
                    if (isinstance(attr, type) and issubclass(attr, nn.Module)
                            and attr is not nn.Module
                            and not attr_name.startswith("_")):
                        module_classes.append((attr_name, attr))

                # Sort by name length descending (heuristic: longer name = more specific model)
                module_classes.sort(key=lambda x: len(x[0]), reverse=True)

                for cls_name, cls in module_classes[:5]:
                    for call_args in [
                        {"num_classes": num_classes},
                        {"n_classes": num_classes},
                        {"num_class": num_classes},
                        {},
                    ]:
                        try:
                            instance = cls(**call_args)
                            if isinstance(instance, nn.Module):
                                param_count = sum(p.numel() for p in instance.parameters())
                                if param_count > 100_000:  # skip tiny test modules
                                    logger.info(
                                        "[ModelLoader] instantiated %s:%s (%.1fM params)",
                                        fpath.name, cls_name, param_count / 1e6,
                                    )
                                    return instance
                        except Exception:
                            continue

            except Exception as e:
                logger.debug("[ModelLoader] scan %s failed: %s", fpath.name, type(e).__name__)
                continue

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
