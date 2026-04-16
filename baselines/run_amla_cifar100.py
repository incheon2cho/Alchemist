#!/usr/bin/env python3
"""End-to-end launcher: AutoML-Agent solves CIFAR-100 with Claude CLI + EC2.

Preconditions (done once):
  1. Claude CLI installed on local machine and authenticated.
  2. EC2 reachable via SSH key (see AMLA_EC2_* in remote_execute_ec2.py).
  3. Proxy server running:
         python baselines/claude_cli_proxy.py --port 8001 &
  4. AutoML-Agent requirements installed in conda env `amla`.

Usage:
    conda activate amla
    python baselines/run_amla_cifar100.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- 1. Paths & PYTHONPATH --------------------------------------------------
HERE = Path(__file__).resolve().parent
AMLA_ROOT = HERE / "automl-agent"
if not AMLA_ROOT.is_dir():
    raise SystemExit(f"AutoML-Agent not found at {AMLA_ROOT}")

sys.path.insert(0, str(HERE))        # for remote_execute_ec2.py
sys.path.insert(0, str(AMLA_ROOT))   # for agent_manager / operation_agent / ...

# AutoML-Agent expects cwd == repo root (uses os.getcwd() in prompt_agent)
os.chdir(str(AMLA_ROOT))

# ---- 2. Patch execution.py to run scripts on EC2 ---------------------------
# (side-effect: monkey-patches operation_agent.execution.execute_script)
import remote_execute_ec2  # noqa: F401

# ---- 3. Import AutoML-Agent's top-level manager ----------------------------
from agent_manager import AgentManager  # noqa: E402

# ---- 4. User prompt describing the task ------------------------------------
# Designed for a fair head-to-head comparison with Alchemist on CIFAR-100:
#   - Same hardware (A10G×1), same time budget (8h wall-clock)
#   - Same data (train/val, no external dataset)
#   - Same pretraining allowance (ImageNet-1K only)
#   - No hints about specific architectures/techniques (let the agent decide)
CIFAR100_USER_PROMPT = """\
Build a model to classify images from the CIFAR-100 dataset.

## EXECUTION ENVIRONMENT — READ FIRST
You are planning on a CPU-only orchestration host. The actual training
runs on a REMOTE GPU server (AWS g5.xlarge: 1× NVIDIA A10G 24GB VRAM,
**4 vCPU, 16GB RAM**) reached automatically when you call
`execute_script(...)`. The remote server has CUDA-ready PyTorch.

CRITICAL RULES:
1. The dataset already exists ON THE REMOTE GPU at:
       /home/ubuntu/data/cifar100/{{train,val}}/<class>/*.jpg
   Your generated training/eval scripts MUST reference this path
   verbatim. Do NOT attempt to download CIFAR-100, do NOT materialize
   it under agent_workspace/datasets/, do NOT try local fallback paths.
2. Every Python script you generate WILL execute on the remote GPU,
   not locally. So the remote paths above are the ones it sees.
3. A `FileNotFoundError` for /home/ubuntu/data/cifar100 on the LOCAL
   orchestration box is EXPECTED — that path lives only on the GPU
   server. Trust it; do not "fix" it by switching to a local path.
4. Save checkpoints to ./agent_workspace/trained_models/  and reports
   to ./reports/  — these are auto-rsynced back from the GPU after
   each script run.
5. Spend AT MOST 2 plan revisions before invoking execute_script.
   Iterate on actual training results, not on the plan.
6. Do NOT import `gradio`, `streamlit`, or any other UI-only library.

## REMOTE HARDWARE LIMITS — READ CAREFULLY
The g5.xlarge has only **4 vCPU and 16GB RAM**, which is small relative
to the A10G GPU. Aggressive DataLoader settings will starve the system
and crash the SSH connection. **Use the following limits**:

- `num_workers <= 2` (NEVER 4, 6, or 8)
- `persistent_workers = False` (avoid permanent worker memory)
- `prefetch_factor = 2`  (default; do NOT set higher)
- `pin_memory = True` is OK
- Avoid `multiprocessing.spawn` outside the DataLoader.
- Total resident process memory should stay under ~12 GB.

For ImageNet-1K-pretrained backbones (ConvNeXt-Base, ViT-Base, etc.),
batch_size 128 in bf16 fits easily on 24GB VRAM with these settings.
Pretrained weights download from HuggingFace once is OK (~150-300 MB).

## GPU IS MANDATORY — NO CPU FALLBACK
The A10G is available and must be used. The wrapper verifies CUDA
before running your script (pre-flight fails with exit=42 if CUDA
isn't accessible). Inside your script, the FIRST thing you do after
`import torch` must be:

    import torch
    assert torch.cuda.is_available(), "CUDA is required; no CPU fallback allowed"
    device = torch.device("cuda")           # always cuda, never conditional
    # Do NOT write:  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Do NOT write:  if not torch.cuda.is_available(): device = "cpu"
    # That pattern causes silent CPU training on ~4 vCPU → it is forbidden.

Use `torch.autocast("cuda", dtype=torch.bfloat16)` (never `"cpu"`).
Use `.to("cuda", non_blocking=True)` for tensors/model/data.
If CUDA is not available, raise immediately — the wrapper will catch
it and retry. CPU training on this box yields no usable result.

## Task
- Dataset: CIFAR-100 (100 fine-grained classes, 32x32 RGB natural images).
- Training set: 50,000 images (500 per class) at
  {root}/train/<class>/*.jpg     (REMOTE PATH — do not localize)
- Validation set: 10,000 images (100 per class) at
  {root}/val/<class>/*.jpg
- Evaluation metric: Top-1 accuracy on the validation set.
  Report the single accuracy number (percentage, 2 decimal places).
  No test set; selection and reporting use validation only.

## Constraints
- Hardware: single NVIDIA A10G (24GB VRAM).
- Time budget: 8 hours wall-clock for training + eval combined.
- ImageNet-1K AND ImageNet-21K pretrained backbones are allowed
  (timm or HuggingFace; e.g., convnext_base.fb_in22k_ft_in1k,
  vit_base_patch16_224.augreg_in21k_ft_in1k). NOT allowed: external
  large-scale corpora (JFT, LAION, LVD-142M, proprietary datasets).
- Do NOT use any held-out data beyond the provided validation split.
- Single final model; ensembling and TTA permitted within the budget.

## Deliverable
- A training script, an evaluation script, the final Top-1 accuracy,
  and a brief rationale (architecture, optimizer, augmentation, schedule).

Please plan briefly (≤2 revisions), then write the scripts and CALL
execute_script to run them on the remote GPU. The dataset is at
{data_path} (remote path — do not change to a local path).
"""

# ---- 5. Paths for data + workspace -----------------------------------------
# Local agent_workspace; generated code runs on EC2 via rsync + ssh.
WORKSPACE = AMLA_ROOT / "agent_workspace"
(WORKSPACE / "datasets").mkdir(parents=True, exist_ok=True)
(WORKSPACE / "trained_models").mkdir(parents=True, exist_ok=True)

# EC2 has CIFAR-100 at /home/ubuntu/data/cifar100; we tell the agent this path.
# The agent's generated code runs on EC2, so it should use the EC2 path directly.
EC2_CIFAR100 = "/home/ubuntu/data/cifar100"
data_path_for_prompt = EC2_CIFAR100

user_prompt = CIFAR100_USER_PROMPT.format(
    root=EC2_CIFAR100,
    data_path=EC2_CIFAR100,
)

# ---- 6. Run the AutoML-Agent manager ---------------------------------------
def main() -> None:
    print("=" * 70)
    print("AutoML-Agent × Claude CLI × EC2  —  CIFAR-100 pipeline")
    print("=" * 70)
    print(f"  AMLA_ROOT  : {AMLA_ROOT}")
    print(f"  WORKSPACE  : {WORKSPACE}")
    print(f"  DATA (EC2) : {EC2_CIFAR100}")
    print(f"  PROXY      : {os.environ.get('AMLA_PROXY', 'http://127.0.0.1:8001/v1')}")
    print(f"  EC2_HOST   : {os.environ.get('AMLA_EC2_HOST', 'default (see remote_execute_ec2.py)')}")
    print("=" * 70)

    manager = AgentManager(
        "image_classification",      # task type → loads prompt_pool/image_classification.py
        llm="gpt-4",                 # routed to Claude via proxy
        interactive=False,
        data_path=str(WORKSPACE / "datasets"),  # AutoML-Agent expects a local dir
        n_revise=2,                  # cap plan-revision loop to avoid endless refinement
    )
    manager.initiate_chat(user_prompt)


if __name__ == "__main__":
    main()
