"""Training executor abstraction — local simulation vs remote GPU.

Supports:
  - LocalExecutor: in-process simulation (default, for testing)
  - AWSExecutor: submits training jobs to a remote GPU instance via SSH
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alchemist.core.schemas import TrialConfig, TrialResult, UserTask

logger = logging.getLogger(__name__)


def _flatten_config(config: "TrialConfig") -> dict:
    """Convert TrialConfig to a flat dict, merging `extra` into top level.

    train_worker.py reads all keys with flat ``config.get(key)`` calls, so
    the nested ``extra`` dict must be hoisted to the top level before the job
    JSON is sent to the remote worker.
    """
    flat = asdict(config)
    extra = flat.pop("extra", {}) or {}
    flat.update(extra)
    return flat


class TrainingExecutor(ABC):
    """Abstract training executor interface."""

    @abstractmethod
    def run_trial(
        self,
        trial_id: int,
        base_model: str,
        task: UserTask,
        config: TrialConfig,
    ) -> TrialResult:
        """Execute a single training trial and return the result."""

    @abstractmethod
    def evaluate_baseline(
        self,
        base_model: str,
        task: UserTask,
    ) -> float:
        """Evaluate base model without modification (baseline score)."""


class LocalExecutor(TrainingExecutor):
    """Simulated local executor (no GPU required). For testing."""

    def run_trial(
        self,
        trial_id: int,
        base_model: str,
        task: UserTask,
        config: TrialConfig,
    ) -> TrialResult:
        import random

        base = 65.0
        lr_bonus = {1e-4: 2.0, 3e-4: 4.0, 1e-3: 3.0, 3e-3: 0.0}
        base += lr_bonus.get(config.lr, 1.0)

        if not config.freeze_backbone:
            base += 5.0

        adapter_bonus = {"linear_head": 3.0, "lora": 4.0, "none": 0.0}
        base += adapter_bonus.get(config.adapter, 0.0)

        model_bonus = {
            "dinov2_vitb14": 8.0, "dinov2_vits14": 5.0,
            "vit_b16_dino": 4.0, "vit_s16_supervised": 3.0,
        }
        base += model_bonus.get(base_model, 2.0)

        score = base + random.uniform(-3, 3)
        score = min(score, 99.0)
        elapsed = config.epochs * 2.0 + random.uniform(0, 5)

        return TrialResult(
            trial_id=trial_id,
            config=config,
            score=score,
            train_loss=random.uniform(0.1, 0.5),
            val_loss=random.uniform(0.2, 0.6),
            elapsed_s=elapsed,
        )

    def evaluate_baseline(self, base_model: str, task: UserTask) -> float:
        import random

        base_scores = {
            "dinov2_vitb14": 72.0, "dinov2_vits14": 68.0,
            "vit_b16_dino": 67.0, "vit_s16_supervised": 65.0,
            "vitamin_s_clip": 66.0,
        }
        return base_scores.get(base_model, 60.0) + random.uniform(-3, 3)


class AWSExecutor(TrainingExecutor):
    """Remote executor that submits training jobs to an AWS GPU instance via SSH.

    Prerequisites:
      - SSH key-based auth configured for the target host
      - train_worker.py deployed to remote_work_dir on the AWS instance
      - PyTorch, timm, etc. installed on the remote environment
      - Training data accessible on the remote instance (e.g. S3 mount or EBS)

    Usage:
        executor = AWSExecutor(
            host="ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com",
            key_path="~/.ssh/my-gpu-key.pem",
            remote_work_dir="/home/ubuntu/alchemist",
            remote_python="conda run -n torch python",
        )
    """

    def __init__(
        self,
        host: str,
        key_path: str | None = None,
        remote_work_dir: str = "/home/ubuntu/alchemist",
        remote_python: str = "python3",
        poll_interval: int = 10,
        ssh_timeout: int = 10,
    ):
        self.host = host
        self.key_path = key_path
        self.remote_work_dir = remote_work_dir
        self.remote_python = remote_python
        self.poll_interval = poll_interval
        self.ssh_timeout = ssh_timeout
        self._remote_gpu_gb: float | None = None  # cached

    def get_remote_gpu_gb(self) -> float:
        """Query remote GPU memory via SSH (cached after first call)."""
        if self._remote_gpu_gb is not None:
            return self._remote_gpu_gb
        try:
            result = self._ssh_cmd(
                "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                mb = float(result.stdout.strip().split("\n")[0])
                self._remote_gpu_gb = mb / 1024.0
                logger.info("Remote GPU: %.0fGB (%s)", self._remote_gpu_gb, self.host)
                return self._remote_gpu_gb
        except Exception as e:
            logger.warning("Failed to query remote GPU: %s", e)
        self._remote_gpu_gb = 46.0  # L40S default
        return self._remote_gpu_gb

    @staticmethod
    def _select_worker(job: dict) -> str:
        """Select the appropriate worker script based on task type.

        Uses the centralized TaskRegistry to resolve task_name → worker_script.
        """
        from alchemist.core.task_registry import get_task_meta_for_name
        task_name = job.get("task", {}).get("name", "")
        meta = get_task_meta_for_name(task_name)
        return meta.worker_script

    def _ssh_cmd(self, command: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        """Run a command on the remote host via SSH."""
        cmd = ["ssh"]
        if self.key_path:
            cmd += ["-i", str(Path(self.key_path).expanduser())]
        cmd += [
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.ssh_timeout}",
            self.host,
            command,
        ]
        logger.debug("SSH: %s", command)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _scp_to_remote(self, local_path: str, remote_path: str) -> None:
        """Copy a file to the remote host."""
        cmd = ["scp"]
        if self.key_path:
            cmd += ["-i", str(Path(self.key_path).expanduser())]
        cmd += [
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"{self.host}:{remote_path}",
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)

    def run_trial(
        self,
        trial_id: int,
        base_model: str,
        task: UserTask,
        config: TrialConfig,
        early_stop_fn=None,
    ) -> TrialResult:
        job = {
            "command": "train",
            "trial_id": trial_id,
            "base_model": base_model,
            "task": {
                "name": task.name,
                "description": task.description,
                "data_path": task.data_path,
                "num_classes": task.num_classes,
                "eval_metric": task.eval_metric,
            },
            "config": _flatten_config(config),
        }
        return self._submit_and_wait(job, early_stop_fn=early_stop_fn)

    def evaluate_baseline(self, base_model: str, task: UserTask) -> float:
        # Use task-appropriate defaults from TaskRegistry
        from alchemist.core.task_registry import get_task_meta_for_name
        task_meta = get_task_meta_for_name(task.name)
        baseline_config = dict(task_meta.default_config) if task_meta.default_config else {}
        baseline_config["base_model"] = base_model
        # Auto-determine baseline epochs based on model size and estimated speed.
        # Goal: baseline should complete in ~30 minutes max.
        # Estimated epoch times on L40S for COCO (118K images):
        #   yolov8n/s: ~8min, yolov8m: ~12min, yolov8l: ~18min, yolov8x: ~25min
        #   yolo11*: similar to yolov8, rtdetr-l: ~90min, rtdetr-x: ~135min
        model_lower = base_model.lower()
        if "rtdetr" in model_lower:
            baseline_epochs = 1  # RT-DETR: 1.5-2h/epoch, 1 epoch is enough for baseline
        elif any(s in model_lower for s in ("11x", "8x")):
            baseline_epochs = 2  # Large YOLO: ~25min/epoch
        elif any(s in model_lower for s in ("11l", "8l")):
            baseline_epochs = 2  # Medium-large: ~18min/epoch
        else:
            baseline_epochs = 3  # Small/medium models: quick
        baseline_config["epochs"] = baseline_epochs
        logger.info(
            "[REASONING] Baseline evaluation config:\n"
            "  Model: %s\n"
            "  Epochs: %d (auto-determined: %s)\n"
            "  Config: batch=%s, lr=%s, img_size=%s\n"
            "  Rationale: %s",
            base_model, baseline_epochs,
            "rtdetr ~2h/ep → 1ep" if "rtdetr" in model_lower
            else "large YOLO ~25min/ep → 2ep" if any(s in model_lower for s in ("11x", "8x"))
            else "medium model → 3ep",
            baseline_config.get("batch_size"), baseline_config.get("lr"),
            baseline_config.get("img_size"),
            "Quick evaluation to establish baseline performance before multi-trial optimization",
        )

        job = {
            "command": "baseline",
            "trial_id": 0,
            "base_model": base_model,
            "task": {
                "name": task.name,
                "description": task.description,
                "data_path": task.data_path,
                "num_classes": task.num_classes,
                "eval_metric": task.eval_metric,
            },
            "config": baseline_config,
        }
        result = self._submit_and_wait(job)
        return result.score

    def _submit_and_wait(
        self,
        job: dict[str, Any],
        early_stop_fn=None,  # callable(progress: dict) -> (keep, reason)
    ) -> TrialResult:
        """Submit a training job to AWS and poll until completion.

        If ``early_stop_fn`` is given, the poll loop also reads the per-epoch
        progress file on the remote host and invokes the callback. When the
        callback returns ``keep=False``, the running train_worker PID is
        SSH-killed and a partial result is returned (status=ok, score = last
        val_acc, early_stopped=True).
        """
        job_id = f"trial_{job['trial_id']}_{int(time.time())}"
        job_file = f"{self.remote_work_dir}/jobs/{job_id}.json"
        result_file = f"{self.remote_work_dir}/jobs/{job_id}_result.json"
        progress_file = f"{self.remote_work_dir}/jobs/{job_id}_progress.json"

        # Ensure jobs directory exists
        self._ssh_cmd(f"mkdir -p {self.remote_work_dir}/jobs")

        # Write job file to remote
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(job, f)
            local_tmp = f.name
        try:
            self._scp_to_remote(local_tmp, job_file)
        finally:
            Path(local_tmp).unlink(missing_ok=True)

        # Launch training on remote (fire-and-forget via Popen).
        # No pipes held → no hang. Poll loop below detects completion.
        submit_cmd = [
            "ssh",
            "-i", str(Path(self.key_path).expanduser()),
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.ssh_timeout}",
            self.host,
            (
                f"cd {self.remote_work_dir} && "
                f"PYTHONUNBUFFERED=1 nohup {self.remote_python} "
                f"{self._select_worker(job)} "
                f"--job {job_file} --output {result_file} "
                f"--progress {progress_file} "
                f"> {self.remote_work_dir}/jobs/{job_id}.log 2>&1 "
                f"< /dev/null &"
            ),
        ]
        subprocess.Popen(
            submit_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # brief pause for SSH to deliver command
        logger.info("Submitted job %s to %s", job_id, self.host)

        # Poll for result (and mid-trial progress for early-stop)
        max_wait = 3600 * 4  # 4 hours max
        elapsed = 0
        last_progress_epoch = -1
        while elapsed < max_wait:
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

            # Early-stop monitoring: read progress.json from remote, invoke callback.
            if early_stop_fn is not None:
                prog_check = self._ssh_cmd(
                    f"cat {progress_file} 2>/dev/null", timeout=10,
                )
                if prog_check.returncode == 0 and prog_check.stdout.strip():
                    try:
                        prog = json.loads(prog_check.stdout.strip())
                        cur_epoch = int(prog.get("epoch", 0))
                        if cur_epoch != last_progress_epoch:
                            last_progress_epoch = cur_epoch
                            keep, reason = early_stop_fn(prog)
                            logger.info(
                                "[early-stop] trial %s epoch %d: keep=%s (%s)",
                                job_id, cur_epoch, keep, reason,
                            )
                            if not keep:
                                logger.warning(
                                    "[early-stop] KILL trial %s: %s", job_id, reason,
                                )
                                # SSH-kill train_worker for this job
                                self._ssh_cmd(
                                    f"pgrep -f 'train_worker.py --job.*{job_id}' "
                                    f"| xargs -r kill -9",
                                    timeout=15,
                                )
                                # Synthesize early-stopped result locally
                                return TrialResult(
                                    trial_id=job["trial_id"],
                                    config=TrialConfig(
                                        **{k: v for k, v in (job.get("config") or {}).items()
                                           if k in {f.name for f in
                                                    TrialConfig.__dataclass_fields__.values()}}
                                    ),
                                    score=float(prog.get("val_acc", 0.0)),
                                    train_loss=float(prog.get("train_loss", 0.0)),
                                    val_loss=0.0,
                                    elapsed_s=float(prog.get("elapsed_s", elapsed)),
                                )
                    except (ValueError, KeyError):
                        pass

            check = self._ssh_cmd(f"cat {result_file} 2>/dev/null", timeout=15)
            if check.returncode == 0 and check.stdout.strip():
                try:
                    data = json.loads(check.stdout.strip())
                    if data.get("status") == "error":
                        raise RuntimeError(
                            f"Remote training failed: {data.get('error', 'unknown')}"
                        )
                    cfg = data.get("config", {})
                    return TrialResult(
                        trial_id=data.get("trial_id", job["trial_id"]),
                        config=TrialConfig(
                            lr=cfg.get("lr", 1e-3),
                            batch_size=cfg.get("batch_size", 32),
                            epochs=cfg.get("epochs", 10),
                            weight_decay=cfg.get("weight_decay", 0.01),
                            scheduler=cfg.get("scheduler", "cosine"),
                            augmentation=cfg.get("augmentation", "basic"),
                            freeze_backbone=cfg.get("freeze_backbone", True),
                            adapter=cfg.get("adapter", "none"),
                        ),
                        score=data.get("score", 0.0),
                        train_loss=data.get("train_loss", 0.0),
                        val_loss=data.get("val_loss", 0.0),
                        elapsed_s=data.get("elapsed_s", 0.0),
                    )
                except json.JSONDecodeError:
                    continue  # Partial write, keep polling

            if elapsed % 60 < self.poll_interval:
                logger.info("Waiting for job %s... (%ds)", job_id, elapsed)

        raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")
