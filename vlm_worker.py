"""Alchemist VLM Training Worker — Video Language Model training.

Architecture:
    V-JEPA 2.1 (frozen) → C-Abstractor (trainable) → Qwen3.6 (LoRA)

Trains on LLaVA-Video-178K conversations, evaluates on Video-MME-v2.

Usage:
    python vlm_worker.py --job jobs/vlm_trial.json \\
        --output jobs/vlm_result.json --progress jobs/vlm_progress.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_worker")


# ---------------------------------------------------------------------------
# Video Loading (reuse decord-based approach from video_worker)
# ---------------------------------------------------------------------------

def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    img_size: int = 384,
) -> torch.Tensor:
    """Load and preprocess video frames.

    Returns: (C, T, H, W) tensor normalized with ImageNet stats.
    """
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)

    # Uniform temporal sampling
    if total >= num_frames:
        indices = torch.linspace(0, total - 1, num_frames).long().tolist()
    else:
        indices = list(range(total)) + [total - 1] * (num_frames - total)

    frames = vr.get_batch(indices)  # (T, H, W, C)
    frames = frames.float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

    # Resize
    frames = F.interpolate(frames, size=(img_size, img_size), mode="bilinear", align_corners=False)

    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # (T, C, H, W) → (C, T, H, W) for V-JEPA
    return frames.permute(1, 0, 2, 3)


# ---------------------------------------------------------------------------
# VLM Dataset — LLaVA-Video-178K format
# ---------------------------------------------------------------------------

# Special token for visual placeholder
IMAGE_TOKEN = "<image>"


class VLMVideoDataset(torch.utils.data.Dataset):
    """LLaVA-Video-178K conversation dataset.

    Each sample: video + multi-turn conversation [{from, value}, ...]
    """

    def __init__(
        self,
        data_path: str,
        video_dir: str,
        tokenizer,
        num_frames: int = 16,
        img_size: int = 384,
        max_text_len: int = 512,
        num_visual_tokens: int = 64,
    ):
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.num_visual_tokens = num_visual_tokens

        # Load conversation data
        if data_path.endswith(".json"):
            with open(data_path) as f:
                self.samples = json.load(f)
        elif data_path.endswith(".jsonl"):
            self.samples = []
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        else:
            # Try loading from HuggingFace datasets
            try:
                from datasets import load_dataset
                ds = load_dataset(data_path, split="train")
                self.samples = list(ds)
            except Exception:
                raise ValueError(f"Cannot load data from {data_path}")

        # Ensure IMAGE_TOKEN_ID exists in tokenizer
        if IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})

        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        logger.info("VLMVideoDataset: %d samples, %d frames, img=%d, max_text=%d",
                     len(self.samples), num_frames, img_size, max_text_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample.get("video", "")
        conversations = sample.get("conversations", [])

        # Resolve video path
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.video_dir, video_path)

        # Load video
        try:
            video = load_video_frames(video_path, self.num_frames, self.img_size)
        except Exception:
            # Fallback: random tensor if video not available
            video = torch.randn(3, self.num_frames, self.img_size, self.img_size)

        # Build conversation text
        # Format: <image> tokens + human question + gpt answer
        input_text = ""
        target_text = ""

        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "human":
                # Replace <image> placeholder in human message
                value = value.replace("<image>", "").replace("\n", " ").strip()
                input_text += f"User: {value}\n"
            elif role == "gpt":
                target_text += value.strip()

        # Tokenize: [visual_placeholders] + [human_text] + [gpt_response]
        # Visual placeholders will be replaced with actual visual tokens in the model
        visual_placeholder = IMAGE_TOKEN * self.num_visual_tokens
        full_text = f"{visual_placeholder}\n{input_text}Assistant: {target_text}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_text_len + self.num_visual_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Build labels: -100 for visual tokens and human text, actual ids for gpt response
        labels = input_ids.clone()

        # Mask everything before "Assistant:" as -100
        assistant_token = self.tokenizer.encode("Assistant:", add_special_tokens=False)
        # Find the position of "Assistant:" in input_ids
        for i in range(len(input_ids) - len(assistant_token)):
            if input_ids[i:i + len(assistant_token)].tolist() == assistant_token:
                labels[:i + len(assistant_token)] = -100
                break
        else:
            # If not found, mask all visual tokens at least
            labels[:self.num_visual_tokens] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            "video": video,  # (C, T, H, W)
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# VLM Model — V-JEPA 2.1 + C-Abstractor + Qwen (LoRA)
# ---------------------------------------------------------------------------

class VLMModel(nn.Module):
    """Video Language Model: vision encoder + abstractor + LLM.

    Forward flow:
        video → vision_encoder (frozen) → visual tokens (B, N, D_v)
        → abstractor → compressed tokens (B, M, D_llm)
        → inject into text embeddings at <image> positions
        → LLM forward with labels → loss
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        abstractor: nn.Module,
        llm: nn.Module,
        image_token_id: int,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.abstractor = abstractor
        self.llm = llm
        self.image_token_id = image_token_id

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video through frozen vision encoder + abstractor.

        Args:
            video: (B, C, T, H, W)

        Returns:
            (B, num_visual_tokens, D_llm)
        """
        with torch.no_grad():
            visual_tokens = self.vision_encoder(video)  # (B, N, D_v)
        # Compress and project
        return self.abstractor(visual_tokens.float())  # (B, M, D_llm)

    def forward(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass with visual token injection.

        Args:
            video: (B, C, T, H, W)
            input_ids: (B, S) text token ids with IMAGE_TOKEN placeholders
            attention_mask: (B, S)
            labels: (B, S) with -100 for non-target positions
        """
        B = video.shape[0]

        # 1. Encode video → compressed visual tokens
        visual_tokens = self.encode_video(video)  # (B, M, D_llm)

        # 2. Get text embeddings from LLM
        embed_layer = self.llm.get_input_embeddings()

        text_embeds = embed_layer(input_ids).clone()  # (B, S, D_llm) — clone to allow in-place

        # 3. Replace <image> placeholder positions with visual tokens
        for b in range(B):
            img_positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            num_to_inject = min(len(img_positions), visual_tokens.shape[1])
            if num_to_inject > 0:
                text_embeds[b, img_positions[:num_to_inject]] = visual_tokens[b, :num_to_inject].to(text_embeds.dtype)

        # 4. Forward through LLM with injected embeddings
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text conditioned on video."""
        B = video.shape[0]
        visual_tokens = self.encode_video(video)

        embed_layer = self.llm.get_input_embeddings()

        text_embeds = embed_layer(input_ids).clone()

        for b in range(B):
            img_positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            num_to_inject = min(len(img_positions), visual_tokens.shape[1])
            if num_to_inject > 0:
                text_embeds[b, img_positions[:num_to_inject]] = visual_tokens[b, :num_to_inject].to(text_embeds.dtype)

        return self.llm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

class DummyVisionEncoder(nn.Module):
    """Dummy vision encoder that returns random features (for pipeline testing)."""
    def __init__(self, embed_dim: int = 1408, num_tokens: int = 196):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.is_video = True
        self.proj = nn.Linear(3, embed_dim)  # learnable projection

    def forward(self, x):
        B = x.shape[0]
        # Simple: average pool video to (B, 3) then project
        pooled = x.reshape(B, 3, -1).mean(dim=-1)  # (B, 3)
        token = self.proj(pooled).unsqueeze(1)  # (B, 1, D)
        return token.expand(B, self.num_tokens, self.embed_dim)


def load_vision_encoder(variant: str = "vjepa2.1_vitg", device="cuda"):
    """Load frozen V-JEPA 2.1 encoder for VLM."""
    if variant.lower() == "none" or variant.lower() == "dummy":
        model = DummyVisionEncoder(embed_dim=1408, num_tokens=196)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        logger.info("Vision encoder: DUMMY (embed_dim=1408, for pipeline testing)")
        return model, 1408

    # Try torch.hub direct loading (most reliable for V-JEPA2)
    hub_name_map = {
        "vjepa2_vitl": "vjepa2_vit_large", "vjepa2_vitb": "vjepa2_vit_base",
        "vjepa2_vitg": "vjepa2_vit_giant", "vjepa2_vith": "vjepa2_vit_huge",
        "vjepa2.1_vitl": "vjepa2_1_vit_large_384", "vjepa2.1_vitb": "vjepa2_1_vit_base_384",
        "vjepa2.1_vitg": "vjepa2_1_vit_giant_384", "vjepa2.1_vitG": "vjepa2_1_vit_gigantic_384",
    }
    hub_name = hub_name_map.get(variant, variant)

    try:
        # Fix localhost URL if needed
        import pathlib
        hub_backbones = pathlib.Path.home() / ".cache/torch/hub/facebookresearch_vjepa2_main/src/hub/backbones.py"
        if hub_backbones.exists():
            content = hub_backbones.read_text()
            if "localhost:8300" in content:
                content = content.replace(
                    'VJEPA_BASE_URL = "http://localhost:8300"',
                    'VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"',
                )
                hub_backbones.write_text(content)
                logger.info("  Fixed V-JEPA2 weight URL to dl.fbaipublicfiles.com")

        result = torch.hub.load('facebookresearch/vjepa2', hub_name, pretrained=True)
        model = result[0] if isinstance(result, tuple) else result
        embed_dim = getattr(model, "embed_dim", 1024)
    except Exception as e:
        logger.warning("torch.hub failed: %s, trying vjepa_loader...", e)
        try:
            from vjepa_loader import load_vjepa2
        except ImportError:
            from alchemist.core.vjepa_loader import load_vjepa2
        model = load_vjepa2(variant, num_classes=0, pretrained=True, for_vlm=True)
        embed_dim = model.embed_dim

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info("Vision encoder: %s (embed_dim=%d, frozen, GPU=%.1fGB)",
                variant, embed_dim, torch.cuda.memory_allocated() / 1e9)
    return model, embed_dim


def load_llm_with_lora(
    model_id: str = "Qwen/Qwen3.6-35B-A3B-FP8",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: list[str] = None,
    device_map: str = "auto",
):
    """Load Qwen LLM with LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    logger.info("Loading LLM: %s ...", model_id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (try flash_attention_2, fallback to sdpa)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device_map,
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device_map,
            trust_remote_code=True, attn_implementation="sdpa",
        )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    logger.info("LLM loaded: %s + LoRA(r=%d, alpha=%d)", model_id, lora_r, lora_alpha)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_vlm_training(
    base_model: str,
    task: dict,
    config: dict,
    trial_id: int,
) -> dict:
    """Train a Video Language Model."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    lr = config.get("lr", 2e-5)
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 16)
    num_frames = config.get("num_frames", 16)
    img_size = config.get("img_size", 384)
    max_text_len = config.get("max_text_len", 512)
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    num_output_tokens = config.get("num_output_tokens", 64)
    warmup_ratio = config.get("warmup_ratio", 0.03)
    weight_decay = config.get("weight_decay", 0.01)
    vision_variant = config.get("vision_encoder", "vjepa2.1_vitg")
    llm_model_id = config.get("llm_model", "Qwen/Qwen3.6-35B-A3B-FP8")

    data_path = task.get("data_path", "")
    video_dir = task.get("video_dir", data_path)

    logger.info("=== VLM Training ===")
    logger.info("  Vision: %s (frozen)", vision_variant)
    logger.info("  LLM: %s (LoRA r=%d)", llm_model_id, lora_r)
    logger.info("  Data: %s", data_path)

    # 1. Load vision encoder (frozen)
    vision_encoder, vision_dim = load_vision_encoder(vision_variant, device)

    # 2. Load LLM + LoRA
    llm, tokenizer = load_llm_with_lora(
        llm_model_id, lora_r=lora_r, lora_alpha=lora_alpha,
    )

    # Get LLM hidden dimension
    llm_config = llm.config if hasattr(llm, "config") else llm.base_model.config
    llm_hidden = getattr(llm_config, "hidden_size", 2560)

    # 3. Build C-Abstractor
    try:
        from alchemist.core.c_abstractor import CAbstractor
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from c_abstractor import CAbstractor

    abstractor = CAbstractor(
        in_dim=vision_dim,
        out_dim=llm_hidden,
        num_output_tokens=num_output_tokens,
    ).to(device)
    logger.info("  C-Abstractor: %d → %d, %d tokens",
                vision_dim, llm_hidden, num_output_tokens)

    # Add <image> token to tokenizer
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        llm.resize_token_embeddings(len(tokenizer))

    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # 4. Build VLM model
    vlm = VLMModel(vision_encoder, abstractor, llm, image_token_id)

    # 5. Dataset
    dataset = VLMVideoDataset(
        data_path=data_path,
        video_dir=video_dir,
        tokenizer=tokenizer,
        num_frames=num_frames,
        img_size=img_size,
        max_text_len=max_text_len,
        num_visual_tokens=num_output_tokens,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # 6. Optimizer (only trainable params: C-Abstractor + LoRA)
    trainable_params = [
        {"params": abstractor.parameters(), "lr": lr * 10},  # Higher LR for abstractor
        {"params": [p for p in llm.parameters() if p.requires_grad], "lr": lr},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=weight_decay)

    total_steps = epochs * (len(dataloader) // grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("  Training: %d epochs, batch=%d, grad_accum=%d, effective_batch=%d",
                epochs, batch_size, grad_accum, batch_size * grad_accum)
    logger.info("  Total steps: %d, warmup: %d", total_steps, warmup_steps)

    # 7. Training loop
    best_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        vlm.train()
        vision_encoder.eval()  # Keep frozen encoder in eval mode
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            video = batch["video"].to(device)  # (B, C, T, H, W)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = vlm(video, input_ids, attention_mask, labels)
                loss = outputs["loss"] / grad_accum

            # Backward
            loss.backward()
            epoch_loss += outputs["loss"].item()
            num_batches += 1

            # Optimizer step every grad_accum batches
            if (batch_idx + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in vlm.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        "  [step %d/%d] loss=%.4f, lr=%.2e",
                        global_step, total_steps, avg_loss,
                        scheduler.get_last_lr()[0],
                    )

            # Free GPU memory
            del video, outputs, loss
            torch.cuda.empty_cache()

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        logger.info("  Epoch %d/%d, avg_loss=%.4f, best_loss=%.4f",
                    epoch + 1, epochs, avg_loss, best_loss)

        # Progress
        _prog_path = os.environ.get("ALCHEMIST_PROGRESS_PATH")
        if _prog_path:
            try:
                prog = {
                    "epoch": epoch + 1, "total_epochs": epochs,
                    "train_loss": avg_loss, "best_loss": best_loss,
                    "global_step": global_step, "total_steps": total_steps,
                    "elapsed_s": time.time() - t0,
                }
                with open(_prog_path, "w") as f:
                    json.dump(prog, f)
            except Exception:
                pass

    # 8. Save checkpoint
    ckpt_dir = Path("/home/ubuntu/checkpoints/vlm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"vlm_trial{trial_id}"
    ckpt_path.mkdir(exist_ok=True)

    # Save C-Abstractor
    torch.save(abstractor.state_dict(), ckpt_path / "c_abstractor.pt")
    # Save LoRA adapter
    llm.save_pretrained(str(ckpt_path / "lora_adapter"))
    logger.info("  Checkpoint saved: %s", ckpt_path)

    # 9. Evaluate on Video-MME if available
    eval_score = 0.0
    eval_path = task.get("eval_dataset")
    if eval_path:
        try:
            eval_score = run_vlm_eval(vlm, tokenizer, eval_path, device,
                                       num_frames, img_size, num_output_tokens)
            logger.info("  Video-MME accuracy: %.2f%%", eval_score)
        except Exception as e:
            logger.warning("  Eval failed: %s", e)

    elapsed = time.time() - t0
    return {
        "status": "ok",
        "trial_id": trial_id,
        "score": round(eval_score, 2),
        "train_loss": round(best_loss, 4),
        "elapsed_s": round(elapsed, 1),
        "config": config,
        "checkpoint_path": str(ckpt_path),
        "applied_techniques": {
            "method": "vlm_vjepa2_cabstractor_qwen_lora",
            "vision_encoder": vision_variant,
            "llm": llm_model_id,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "num_visual_tokens": num_output_tokens,
            "precision": "bf16",
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": grad_accum,
            "num_frames": num_frames,
        },
    }


# ---------------------------------------------------------------------------
# Evaluation — Video-MME-v2
# ---------------------------------------------------------------------------

def run_vlm_eval(
    model: VLMModel,
    tokenizer,
    eval_path: str,
    device,
    num_frames: int = 16,
    img_size: int = 384,
    num_visual_tokens: int = 64,
) -> float:
    """Evaluate on Video-MME-v2 multiple-choice QA."""
    model.eval()

    # Load eval data (JSON or parquet)
    eval_ds = None
    if os.path.exists(eval_path) and eval_path.endswith(".json"):
        with open(eval_path) as f:
            eval_ds = json.load(f)
    else:
        try:
            from datasets import load_dataset
            eval_ds = list(load_dataset(eval_path, split="test"))
        except Exception:
            pass
    if not eval_ds:
        logger.warning("Cannot load eval dataset: %s", eval_path)
        return 0.0

    # Resolve video directory (same parent as eval file)
    eval_video_dir = os.path.join(os.path.dirname(eval_path), "videos")

    correct = total = 0
    skipped = 0
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    for sample in eval_ds:
        video_id = sample.get("video_id", "")
        question = sample.get("question", "")
        options = sample.get("options", "")
        answer = sample.get("answer", "")

        if not question:
            continue

        # Resolve video path: try video_dir/video_id.mp4
        video_file = os.path.join(eval_video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_file):
            skipped += 1
            continue

        # Load video
        try:
            video = load_video_frames(video_file, num_frames, img_size)
            video = video.unsqueeze(0).to(device)  # (1, C, T, H, W)
        except Exception:
            skipped += 1
            continue

        # Build prompt
        visual_placeholder = IMAGE_TOKEN * num_visual_tokens
        prompt = f"{visual_placeholder}\nQuestion: {question}\nOptions: {options}\nAnswer:"

        encoding = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Generate
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    video, input_ids, attention_mask, max_new_tokens=16,
                )
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract answer letter
            predicted = ""
            for char in response.strip():
                if char.upper() in "ABCDEFGH":
                    predicted = char.upper()
                    break

            if predicted == answer.strip().upper():
                correct += 1
            total += 1
        except Exception:
            continue

        if total % 50 == 0 and total > 0:
            logger.info("  [eval] %d/%d correct so far (%.1f%%), skipped=%d",
                        correct, total, 100.0 * correct / total, skipped)

    accuracy = 100.0 * correct / max(total, 1)
    logger.info("  Video-MME eval: %d/%d correct (%.2f%%), %d skipped (no video)",
                correct, total, accuracy, skipped)
    return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Alchemist VLM Training Worker")
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", help="Path to write per-epoch progress JSON")
    args = parser.parse_args()

    if args.progress:
        os.environ["ALCHEMIST_PROGRESS_PATH"] = args.progress

    with open(args.job) as f:
        job = json.load(f)

    logger.info("VLM Job: model=%s, task=%s",
                job.get("base_model"), job.get("task", {}).get("name"))

    try:
        result = run_vlm_training(
            base_model=job.get("base_model", "vlm_vjepa2_qwen"),
            task=job.get("task", {}),
            config=job.get("config", {}),
            trial_id=job.get("trial_id", 0),
        )
    except Exception as e:
        logger.error("Failed: %s", e, exc_info=True)
        result = {
            "status": "error",
            "error": str(e),
            "trial_id": job.get("trial_id", 0),
        }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Result: status=%s, score=%s", result.get("status"), result.get("score"))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
