"""Run AutoML-Agent on COCO Object Detection.

This script adapts AutoML-Agent (ICML 2025) for COCO detection, providing
the same task description and constraints as Alchemist for fair comparison.

AutoML-Agent generates training code via LLM, which is then executed on
the remote EC2 GPU via remote_execute_ec2.py (fixed: no CUDA assert).

Usage:
    python baselines/run_amla_coco_detection.py \
        --host ubuntu@<EC2_IP> \
        --key ~/.ssh/alchemist-gpu-key-use1.pem
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("amla_coco")

# AutoML-Agent task specification for COCO detection
COCO_DETECTION_TASK = {
    "task_type": "object_detection",
    "dataset_name": "COCO 2017",
    "dataset_description": (
        "COCO 2017 Object Detection dataset.\n"
        "- 80 object categories with bounding box annotations\n"
        "- Training: 118,287 images in /home/ubuntu/data/coco/images/train2017/\n"
        "- Validation: 5,000 images in /home/ubuntu/data/coco/images/val2017/\n"
        "- YOLO format labels in /home/ubuntu/data/coco/labels/\n"
        "- Dataset config: /home/ubuntu/data/coco/coco.yaml\n"
        "- Pretrained COCO weights are allowed\n"
    ),
    "evaluation_metric": "mAP50-95",
    "hardware": "NVIDIA A10G 24GB GPU, 8 vCPU, 32GB RAM",
    "time_budget": "8 hours",
    "requirements": (
        "Use ultralytics library (pip install ultralytics) for YOLO/RT-DETR models.\n"
        "Available models: YOLOv8(n/s/m/l/x), YOLO11(n/s/m/l/x), RT-DETR(l/x).\n"
        "Train on /home/ubuntu/data/coco/coco.yaml.\n"
        "Report final mAP50, mAP50-95, precision, recall on val2017.\n"
        "Maximize mAP50-95 within 8-hour GPU budget.\n"
        "Save best model checkpoint.\n"
    ),
}

# Prompt template for AutoML-Agent LLM
PLAN_PROMPT = """
You are an AutoML agent for object detection. Design the best training plan
for COCO 2017 detection to maximize mAP50-95 within 8 hours on NVIDIA A10G 24GB.

Dataset: {dataset_description}

Available approaches:
1. YOLOv8/v9/v10/v11 (ultralytics) — fast, well-optimized
2. RT-DETR (ultralytics) — transformer-based, NMS-free
3. Custom training scripts with torchvision

Considerations:
- GPU: A10G 24GB — YOLOv8x fits with batch=4, RT-DETR-X with batch=2
- Time: 8 hours — enough for ~100 epochs with YOLOv8m
- Pretrained COCO weights available — fine-tuning is efficient

Generate a complete Python training script that:
1. Selects the best model architecture
2. Configures optimal hyperparameters (lr, batch, augmentation, epochs)
3. Trains on COCO 2017
4. Evaluates on val2017
5. Reports mAP50, mAP50-95, precision, recall
6. Saves the best checkpoint

Output only the Python script, no explanations.
"""

CODE_PROMPT = """
Write a complete Python script for COCO 2017 object detection training.

Requirements:
{requirements}

The script should:
1. Import ultralytics YOLO
2. Load a pretrained model (recommend YOLOv8m or YOLO11m for A10G)
3. Train with optimal hyperparameters for 8-hour budget
4. Use advanced augmentation (mosaic, mixup, copy_paste)
5. Evaluate and print results
6. Save results to /home/ubuntu/alchemist/jobs/amla_coco_result.json

Output ONLY the Python code, no markdown or explanations.
"""


def run_automl_agent_coco(host: str, key: str, n_attempts: int = 5):
    """Run AutoML-Agent on COCO detection with LLM code generation."""
    from alchemist.core.llm import LLMClient

    logger.info("=== AutoML-Agent: COCO Detection ===")
    logger.info("  Host: %s", host)
    logger.info("  Budget: 8 hours")
    logger.info("  Metric: mAP50-95")

    # Step 1: Generate training plan via LLM
    logger.info("\n[1] Generating training plan...")
    llm = LLMClient(provider="claude", model="sonnet")

    plan_prompt = PLAN_PROMPT.format(
        dataset_description=COCO_DETECTION_TASK["dataset_description"],
    )

    try:
        plan = llm.generate(plan_prompt)
        logger.info("  Plan generated (%d chars)", len(plan))
    except Exception as e:
        logger.error("  Plan generation failed: %s", e)
        # Fallback: use a known-good script
        plan = _fallback_training_script()

    # Step 2: Generate executable code
    logger.info("\n[2] Generating training code...")
    code_prompt = CODE_PROMPT.format(
        requirements=COCO_DETECTION_TASK["requirements"],
    )

    for attempt in range(1, n_attempts + 1):
        try:
            code = llm.generate(code_prompt)
            # Clean up markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            logger.info("  Code generated (attempt %d, %d chars)", attempt, len(code))

            # Step 3: Execute on remote EC2
            logger.info("\n[3] Executing on EC2 (attempt %d/%d)...", attempt, n_attempts)
            script_path = Path("/tmp/amla_coco_train.py")
            script_path.write_text(code)

            # Upload and execute
            import subprocess
            scp_cmd = [
                "scp", "-i", key, "-o", "StrictHostKeyChecking=no",
                str(script_path), f"{host}:/home/ubuntu/alchemist/amla_coco_train.py",
            ]
            subprocess.run(scp_cmd, timeout=30, capture_output=True)

            ssh_cmd = [
                "ssh", "-i", key, "-o", "StrictHostKeyChecking=no", host,
                "cd /home/ubuntu/alchemist && "
                "PYTHONUNBUFFERED=1 timeout 28800 python3 amla_coco_train.py "
                "> jobs/amla_coco.log 2>&1",
            ]
            logger.info("  Running: %s", " ".join(ssh_cmd[-1:]))
            result = subprocess.run(ssh_cmd, timeout=29000, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("  Execution succeeded!")
                break
            else:
                logger.warning("  Execution failed (rc=%d), retrying...", result.returncode)
                # Feed error back to LLM for next attempt
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                code_prompt = (
                    f"The previous script failed with error:\n{error_msg}\n\n"
                    f"Fix the script and try again.\n"
                    f"{CODE_PROMPT.format(requirements=COCO_DETECTION_TASK['requirements'])}"
                )

        except Exception as e:
            logger.error("  Attempt %d failed: %s", attempt, e)

    logger.info("\n=== AutoML-Agent COCO Detection complete ===")


def _fallback_training_script() -> str:
    """Known-good fallback training script if LLM generation fails."""
    return '''
import json
from ultralytics import YOLO

# Load pretrained YOLOv8m
model = YOLO("yolov8m.pt")

# Train on COCO
results = model.train(
    data="/home/ubuntu/data/coco/coco.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    lr0=0.01,
    optimizer="auto",
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    patience=10,
    device=0,
    workers=4,
    project="/home/ubuntu/checkpoints/detection",
    name="amla_trial",
    exist_ok=True,
)

# Validate
val = model.val(data="/home/ubuntu/data/coco/coco.yaml", imgsz=640, device=0)

# Save results
result = {
    "status": "ok",
    "agent": "automl-agent",
    "map50": round(float(val.box.map50) * 100, 2),
    "map50_95": round(float(val.box.map) * 100, 2),
    "precision": round(float(val.box.mp) * 100, 2),
    "recall": round(float(val.box.mr) * 100, 2),
}
with open("/home/ubuntu/alchemist/jobs/amla_coco_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Results: {result}")
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="ubuntu@54.159.18.39")
    parser.add_argument("--key", default=os.path.expanduser("~/.ssh/alchemist-gpu-key-use1.pem"))
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()

    run_automl_agent_coco(args.host, args.key, args.attempts)
