#!/usr/bin/env python3
"""Launcher: AutoML-Agent on Shopee-IET product image classification.

Fair comparison with Alchemist:
  - Same hardware (A10G×1, via EC2 remote execution)
  - Same time budget (2h — reflecting the very small dataset size, 640 train)
  - Same LLM (Claude CLI via proxy)
  - No architectural / technique hints
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMLA_ROOT = HERE / "automl-agent"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AMLA_ROOT))
os.chdir(str(AMLA_ROOT))   # AutoML-Agent relies on cwd being the repo root

import remote_execute_ec2  # noqa: F401
from agent_manager import AgentManager  # noqa: E402


EC2_DATA = "/home/ubuntu/data/shopee-iet"

USER_PROMPT = f"""\
Build a model to classify Shopee product images into their category.

## EXECUTION ENVIRONMENT — READ FIRST
Planning happens on a CPU-only orchestration host. Training runs on a
REMOTE GPU server (AWS g5.xlarge: 1× NVIDIA A10G 24GB VRAM,
**4 vCPU, 16GB RAM**) reached automatically via execute_script().

CRITICAL RULES:
1. The dataset already exists ON THE REMOTE GPU at:
       /home/ubuntu/data/shopee-iet/{{train,val,test}}/<class>/*.jpg
   Reference this path verbatim. Do NOT download or materialize locally.
2. A `FileNotFoundError` for /home/ubuntu/data/shopee-iet on the LOCAL
   orchestration box is EXPECTED — that path lives only on the GPU server.
3. Save checkpoints to ./agent_workspace/trained_models/ (auto-rsynced back).
4. Spend AT MOST 2 plan revisions before invoking execute_script.
5. Do NOT import gradio, streamlit, or any UI-only library.

## REMOTE HARDWARE LIMITS — READ CAREFULLY
4 vCPU + 16GB RAM is small relative to the A10G. Use:
- `num_workers <= 2` (NEVER more; the dataset is tiny anyway)
- `persistent_workers = False`
- `prefetch_factor = 2`
- Total resident memory under ~12 GB.

## GPU IS MANDATORY — NO CPU FALLBACK
After `import torch`, the FIRST line of your script must be:
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    device = torch.device("cuda")
Do NOT write `device = "cuda" if torch.cuda.is_available() else "cpu"`.
CPU training on this box is forbidden — raise instead.

Task:
- Dataset: Shopee-IET (Kaggle demo-shopee-iet-competition).
- 4 classes (product categories): BabyPants, BabyShirt,
  womencasualshoes, womenchiffontop.
- Training set: 640 images (160 per class) at {EC2_DATA}/train/<class>/*.jpg
- Validation set: 160 images (40 per class) at {EC2_DATA}/val/<class>/*.jpg
- Additional held-out test set (NOT to be used for selection): 80 images at {EC2_DATA}/test/
- Image sizes vary (not pre-resized).
- Evaluation metric: Top-1 accuracy on the validation set.
  Report the single accuracy number (percentage, 2 decimal places).

Constraints:
- Hardware: a single NVIDIA A10G (24GB VRAM), ~90GB disk for checkpoints.
- Time budget: up to 2 hours of wall-clock training/eval time.
- ImageNet-1K AND ImageNet-21K pretrained models are allowed
  (e.g., from timm or HuggingFace; convnext_base.fb_in22k_ft_in1k,
  vit_base_patch16_224.augreg_in21k_ft_in1k, etc.). External
  large-scale corpora (JFT, LAION, LVD-142M, proprietary datasets)
  are NOT allowed.
- You MUST NOT use any separately held-out test split for model
  selection or tuning; use only the train + val directories provided.
- Submit a single final model; ensembling of models you train is
  allowed, TTA is allowed, but the compute budget applies to the
  total (training + TTA + ensemble).
- Save the best model checkpoint under ./agent_workspace/trained_models/.

Deliverable:
- A training script, an evaluation script, and the final Top-1 accuracy.
- Describe your design choices (architecture, optimizer, augmentation,
  schedule) and why.

The dataset is already available at {EC2_DATA}.
Please plan, write, run, and evaluate the complete pipeline.
"""


def main() -> None:
    workspace = AMLA_ROOT / "agent_workspace_shopee"
    (workspace / "datasets").mkdir(parents=True, exist_ok=True)
    (workspace / "trained_models").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AutoML-Agent × Claude CLI × EC2  —  Shopee-IET pipeline")
    print("=" * 70)
    print(f"  DATA (EC2): {EC2_DATA}")
    print("=" * 70)

    manager = AgentManager(
        "image_classification",
        llm="gpt-4",
        interactive=False,
        data_path=str(workspace / "datasets"),
        n_revise=2,
    )
    manager.initiate_chat(USER_PROMPT)


if __name__ == "__main__":
    main()
