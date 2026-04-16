#!/usr/bin/env python3
"""Remote execution shim for AutoML-Agent's `operation_agent/execution.py`.

Monkey-patches `execute_script()` so that generated Python scripts run on a
remote EC2 instance (with GPU) instead of the local machine. The LLM-driven
orchestration (agents, planning, code generation) continues to run locally,
using the Claude CLI via the proxy.

Flow for each `execute_script(script_name, work_dir, device)` call:

    1. rsync `work_dir` ─► EC2:{REMOTE_BASE}/{relpath(work_dir)}
    2. ssh EC2 "source /opt/pytorch/bin/activate
                 && cd {remote_work_dir}
                 && CUDA_VISIBLE_DEVICES={device} python -u {script_name}"
    3. rsync EC2:{remote_work_dir}/  ─►  work_dir   (pulls back models/logs)
    4. Return (return_code, 'The script has been executed...\n<output>')

Environment variables (read once at import, overridable):

    AMLA_EC2_HOST      default: alchemist-ec2        (~/.ssh/config alias or IP)
    AMLA_EC2_USER      default: ubuntu
    AMLA_EC2_KEY       default: ~/.ssh/alchemist-gpu-key-use1.pem
    AMLA_EC2_REMOTE    default: /home/ubuntu/amla_workspace
    AMLA_EC2_ACTIVATE  default: source /opt/pytorch/bin/activate
    AMLA_EC2_TIMEOUT   default: 7200 (seconds)

Usage (two equivalent options):

    # Option A — apply patch at Python startup
    import remote_execute_ec2   # noqa: F401  (side-effect: patches execute_script)

    # Option B — call the function directly
    from remote_execute_ec2 import execute_script_remote
    rc, out = execute_script_remote("train.py", "./agent_workspace/exp/foo", device="0")
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger("amla.remote-exec")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ----------------------------------------------------------------------
# Configuration (read from env at import)
# ----------------------------------------------------------------------
EC2_HOST     = os.environ.get("AMLA_EC2_HOST",     "3.90.166.150")
EC2_USER     = os.environ.get("AMLA_EC2_USER",     "ubuntu")
EC2_KEY      = os.path.expanduser(
               os.environ.get("AMLA_EC2_KEY",      "~/.ssh/alchemist-gpu-key-use1.pem"))
EC2_REMOTE   = os.environ.get("AMLA_EC2_REMOTE",   "/home/ubuntu/amla_workspace")
EC2_ACTIVATE = os.environ.get("AMLA_EC2_ACTIVATE", "source /opt/pytorch/bin/activate")
EC2_TIMEOUT  = int(os.environ.get("AMLA_EC2_TIMEOUT", "1800"))  # 30min per script
EC2_PREFLIGHT_TIMEOUT = int(os.environ.get("AMLA_EC2_PREFLIGHT_TIMEOUT", "60"))

# Project root = first ancestor of work_dir that contains `agent_workspace/`.
# We rsync only the relevant subtree to avoid uploading gigabytes.
def _resolve_project_paths(local_work_dir: str) -> tuple[Path, Path, str]:
    """Return (local_root, local_work_abs, remote_work_rel).

    local_root:      anchor for rsync (usually the parent of `agent_workspace`)
    local_work_abs:  absolute path of `work_dir`
    remote_work_rel: path relative to `EC2_REMOTE` (stable across runs)
    """
    local_work_abs = Path(local_work_dir).resolve()
    # Walk up until we find `agent_workspace` (AutoML-Agent's convention)
    local_root = local_work_abs
    while local_root.parent != local_root:
        if (local_root / "agent_workspace").is_dir():
            break
        if local_root.name == "agent_workspace":
            local_root = local_root.parent
            break
        local_root = local_root.parent
    if local_root == Path("/") or not (local_root / "agent_workspace").exists():
        # Fallback: use work_dir itself as the root
        local_root = local_work_abs if local_work_abs.is_dir() else local_work_abs.parent

    try:
        remote_rel = str(local_work_abs.relative_to(local_root))
    except ValueError:
        remote_rel = local_work_abs.name
    return local_root, local_work_abs, remote_rel


# ----------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------
def _ssh_base() -> list[str]:
    return [
        "ssh",
        "-i", EC2_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=30",
        f"{EC2_USER}@{EC2_HOST}",
    ]


def _rsync_base() -> list[str]:
    return [
        "rsync", "-az", "--delete",
        "-e", f"ssh -i {EC2_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
    ]


def _run(cmd: list[str], timeout: int | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


# ----------------------------------------------------------------------
# Main entry — drop-in for AutoML-Agent's execute_script()
# ----------------------------------------------------------------------
_PATCH_HEADER = (
    "# [remote_execute_ec2] AUTO-INJECTED: enforce GPU-only + fixed paths\n"
    "import os as _alch_os, sys as _alch_sys\n"
    "_alch_os.environ['CUDA_VISIBLE_DEVICES'] = _alch_os.environ.get('CUDA_VISIBLE_DEVICES', '0')\n"
    "import torch as _alch_torch\n"
    "assert _alch_torch.cuda.is_available(), 'CUDA required (post-patched). No CPU fallback.'\n"
    "print(f'[alch-preflight] cuda={_alch_torch.cuda.is_available()} '\n"
    "      f'name={_alch_torch.cuda.get_device_name(0)}', flush=True)\n"
    "del _alch_os, _alch_sys, _alch_torch\n"
    "# --- original agent script follows ---\n"
)


def _patch_agent_script(local_path: "Path") -> None:
    """Mutate the agent-generated script in-place to:

    1. Inject a GPU-required preflight header (prevents silent CPU fallback).
    2. Replace torchvision CIFAR-100 download with the pre-staged ImageFolder
       at /home/ubuntu/data/cifar100/{train,val}.
    3. Rewrite common relative-dataset paths to absolute EC2 paths.
    4. Force `download=True` → `download=False` so no network fetch is attempted.

    Patches are applied conservatively: each rewrite is idempotent and text-only.
    """
    try:
        src = local_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("patch: could not read %s (%s)", local_path, e)
        return

    orig = src

    # Skip re-patching
    if "[remote_execute_ec2] AUTO-INJECTED" in src:
        return

    # Replacement 1 — force CUDA, strip CPU fallback one-liners
    for pattern, repl in [
        ('torch.device("cuda" if torch.cuda.is_available() else "cpu")',
         'torch.device("cuda")  # [alch-patch] cpu fallback removed'),
        ('torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')',
         'torch.device("cuda")  # [alch-patch] cpu fallback removed'),
        ('"cuda" if torch.cuda.is_available() else "cpu"',
         '"cuda"  # [alch-patch]'),
        ("'cuda' if torch.cuda.is_available() else 'cpu'",
         '"cuda"  # [alch-patch]'),
    ]:
        src = src.replace(pattern, repl)

    # Replacement 2 — force-disable torchvision download (no network needed)
    src = src.replace("download=True", "download=False  # [alch-patch] use pre-staged data")

    # Replacement 3 — common dataset path assignments → absolute EC2 path
    for pattern, repl in [
        ('DATASET_PATH = "_experiments/datasets"',
         'DATASET_PATH = "/home/ubuntu/data"  # [alch-patch]'),
        ("DATASET_PATH = '_experiments/datasets'",
         "DATASET_PATH = '/home/ubuntu/data'  # [alch-patch]"),
        ('"./agent_workspace/datasets/cifar100"',
         '"/home/ubuntu/data/cifar100"  # [alch-patch]'),
        ("'./agent_workspace/datasets/cifar100'",
         "'/home/ubuntu/data/cifar100'  # [alch-patch]"),
        ('"agent_workspace/datasets/cifar100"',
         '"/home/ubuntu/data/cifar100"  # [alch-patch]'),
    ]:
        src = src.replace(pattern, repl)

    # Replacement 4 — torchvision.datasets.CIFAR100(...) → ImageFolder
    # Heuristic rewrite: replace `datasets.CIFAR100(<path>, train=True, ...)` with
    # ImageFolder("/home/ubuntu/data/cifar100/train"). The agent then still gets
    # a dataset object with (image, label) tuples.
    import re as _re
    def _cifar_to_imagefolder(m):
        is_train = "train=True" in m.group(0)
        split = "train" if is_train else "val"
        return (
            f'datasets.ImageFolder("/home/ubuntu/data/cifar100/{split}")'
            f'  # [alch-patch]'
        )
    src = _re.sub(
        r"datasets\.CIFAR100\([^)]*\)",
        _cifar_to_imagefolder,
        src,
    )

    # Always prepend the GPU-preflight header so we fail fast on CUDA loss
    src = _PATCH_HEADER + src

    if src != orig:
        local_path.write_text(src, encoding="utf-8")
        log.info("patched agent script: %s (CPU/download/paths normalised)", local_path.name)


def execute_script_remote(script_name: str, work_dir: str = ".", device: str = "0") -> tuple[int, str]:
    """Run `script_name` located inside `work_dir` on the configured EC2 host.

    NEW (v2): transfer ONLY the patched script file, not the whole agent
    workspace. EC2 already has the base environment (PyTorch, timm, data)
    under /opt/pytorch and /home/ubuntu/data, so the script is standalone.
    Outputs (./agent_workspace/trained_models, ./reports) are created on
    EC2 and pulled back selectively after run.

    Matches the return signature of AutoML-Agent's local `execute_script`:
        (return_code: int, observation: str)
    """
    local_root, local_work_abs, remote_rel = _resolve_project_paths(work_dir)
    local_script = local_work_abs / script_name
    if not local_script.exists():
        msg = f"The file {script_name} does not exist at {local_work_abs}."
        return -1, msg

    # Auto-patch the agent-generated script to enforce GPU + data paths
    try:
        _patch_agent_script(local_script)
    except Exception as e:
        log.warning("patch failed (continuing): %s", e)

    # Clean per-call workdir on EC2 — just holds the script + its outputs.
    # The actual file name is preserved for any agent logic that parses it.
    script_basename = Path(script_name).name
    remote_work_dir = f"{EC2_REMOTE}/run"
    remote_script_path = f"{remote_work_dir}/{script_basename}"

    log.info("remote exec (v2: script-only): script=%s device=%s", script_basename, device)

    # 1. ensure remote work dir + expected output subdirs exist
    mk_cmd = (
        f"mkdir -p {shlex.quote(remote_work_dir)}/agent_workspace/trained_models "
        f"{shlex.quote(remote_work_dir)}/reports "
        f"{shlex.quote(remote_work_dir)}/logs"
    )
    mkdir_rc, _, mkdir_err = _run(_ssh_base() + [mk_cmd])
    if mkdir_rc != 0:
        return -1, f"Failed to create remote dir: {mkdir_err}"

    # 2. scp ONLY the patched script (a few KB) instead of the whole workspace.
    scp_cmd = [
        "scp", "-i", EC2_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        str(local_script),
        f"{EC2_USER}@{EC2_HOST}:{remote_script_path}",
    ]
    rc, _, err = _run(scp_cmd, timeout=60)
    if rc != 0:
        return -1, f"scp script failed (rc={rc}): {err[:500]}"
    log.info("uploaded script (%d bytes) → %s:%s",
             local_script.stat().st_size, EC2_HOST, remote_script_path)

    # 3a. Pre-flight: verify CUDA + data path reachable, with short timeout.
    # Run SEPARATELY from the agent script so a hang in the script itself
    # doesn't block detecting infra issues.
    preflight_cmd = (
        f"{EC2_ACTIVATE} && python -c "
        "'import torch, os, sys; "
        "ok = torch.cuda.is_available(); "
        "name = torch.cuda.get_device_name(0) if ok else None; "
        "data_ok = os.path.isdir(\"/home/ubuntu/data\"); "
        "print(f\"[preflight] cuda={ok} name={name} data_root={data_ok}\"); "
        "sys.exit(0 if (ok and data_ok) else 42)'"
    )
    pf_rc, pf_out, pf_err = _run(
        _ssh_base() + ["bash", "-lc", shlex.quote(preflight_cmd)],
        timeout=EC2_PREFLIGHT_TIMEOUT,
    )
    log.info("preflight: rc=%d stdout=%r", pf_rc, pf_out.strip()[:200])
    if pf_rc != 0:
        return -1, (
            "The script has been executed. Here is the output:\n"
            f"[preflight] FAILED rc={pf_rc}\n{pf_out}\n{pf_err}\n"
            "Abort: GPU or data unavailable on remote host."
        )

    # 3b. Run the agent script with FORCE_CUDA + bounded timeout (EC2_TIMEOUT).
    # On timeout, we explicitly kill the remote process group.
    remote_cmd = (
        f"{EC2_ACTIVATE} && cd {shlex.quote(remote_work_dir)} && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(device)} "
        f"FORCE_CUDA=1 HF_HUB_DISABLE_TELEMETRY=1 "
        f"timeout --kill-after=30 {EC2_TIMEOUT} "
        f"python -u {shlex.quote(script_basename)}"
    )
    rc, stdout, stderr = _run(_ssh_base() + ["bash", "-lc", shlex.quote(remote_cmd)],
                              timeout=EC2_TIMEOUT + 120)
    if rc == 124:
        log.warning("script exceeded remote timeout (%ds) — killed", EC2_TIMEOUT)
        stderr = (stderr or "") + f"\n[remote-exec] script killed after {EC2_TIMEOUT}s timeout"
    # Log rc + tails so we can see why a run failed without re-reading agent logs
    _tail = lambda s, n=600: (s or "")[-n:].replace("\n", " ⏎ ")
    log.info("script done: rc=%d stdout_tail=%r", rc, _tail(stdout))
    if rc != 0 or not stdout:
        log.warning("script stderr_tail=%r", _tail(stderr))

    # 4. pull back ONLY output artifacts (trained_models, reports, logs).
    # Skip the script itself and any pyc caches — we already have the source.
    pull = [x for x in _rsync_base() if x != "--delete"]
    for sub in ("agent_workspace/trained_models", "reports", "logs"):
        down_src = f"{EC2_USER}@{EC2_HOST}:{remote_work_dir}/{sub}/"
        down_dst = str(local_work_abs / sub) + "/"
        # Ensure local subdirs exist
        Path(down_dst).mkdir(parents=True, exist_ok=True)
        _run(pull + [down_src, down_dst], timeout=300)

    if rc != 0:
        observation = stderr or stdout or f"(exit {rc})"
    else:
        observation = stdout or stderr or "(empty output)"
    # Preserve AutoML-Agent's wording
    return rc, "The script has been executed. Here is the output:\n" + observation


# ----------------------------------------------------------------------
# Monkey-patch — apply when this module is imported
# ----------------------------------------------------------------------
def _apply_patch() -> None:
    """Patch every place AutoML-Agent has bound `execute_script`.

    `from X import execute_script` copies the function reference into the
    importing module's namespace at import time. Patching only the source
    module misses these copies, so we hunt them all down.
    """
    targets: list[tuple[str, str]] = [
        # (module_path, attribute_name)
        ("operation_agent.execution", "execute_script"),  # source
        ("operation_agent",            "execute_script"),  # `from .execution import execute_script`
        ("experiments.execution",      "execute_script"),  # twin source
        ("experiments.evaluation",     "execute_script"),  # `from .execution import execute_script`
    ]

    import importlib
    patched_count = 0
    for mod_path, attr in targets:
        try:
            mod = importlib.import_module(mod_path)
        except ModuleNotFoundError as e:
            log.warning("module %s not importable (%s); skipping", mod_path, e)
            continue
        if not hasattr(mod, attr):
            log.warning("module %s has no attr '%s'; skipping", mod_path, attr)
            continue
        if getattr(mod, "_amla_remote_patched_" + attr, False):
            continue
        setattr(mod, "_orig_" + attr, getattr(mod, attr))
        setattr(mod, attr, execute_script_remote)
        setattr(mod, "_amla_remote_patched_" + attr, True)
        patched_count += 1
        log.info("patched %s.%s -> remote EC2", mod_path, attr)

    log.info("remote-exec patch complete: %d call sites rebound", patched_count)


_apply_patch()


# ----------------------------------------------------------------------
# CLI entry for quick manual testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Remote execute a Python script on EC2.")
    p.add_argument("script")
    p.add_argument("--work-dir", default=".")
    p.add_argument("--device", default="0")
    args = p.parse_args()
    rc, out = execute_script_remote(args.script, args.work_dir, args.device)
    print(f"[rc={rc}]")
    print(out)
