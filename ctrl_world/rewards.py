#!/usr/bin/env python3
"""
Qwen3-VL reward for Ctrl-World noise optimization.

This is the Qwen-only subset of the upstream rewards.py (ReWiND, DINO,
RoboMeter, TOPReward dependencies have been stripped).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════
#  Single-view Qwen reward
# ══════════════════════════════════════════════════════════════════════════

QWEN_ZERO_SHOT_PROMPT = (
    "You are evaluating a robotic manipulation task. "
    "The following images are consecutive frames from a single camera "
    "showing a robot gripper attempting to pick up an object from a table.\n\n"
    "A SUCCESSFUL grasp means: the gripper closes around the object, lifts it "
    "clearly off the table surface, and holds it stably in the air.\n"
    "A FAILED grasp means any of: the gripper misses the object entirely, the "
    "gripper closes on empty space, the object slides out during lifting, the "
    "object never leaves the table, or the object is knocked away.\n\n"
    "Based on these frames, did the robot FAIL to grasp and lift the object?\n"
    "Answer with a single word: Yes or No."
)


def _image_entry(img: Image.Image) -> dict:
    return {"type": "image", "image": img}


def _format_qwen_messages(images: List[Image.Image], prompt: str) -> list:
    content = [_image_entry(img) for img in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _preprocess_qwen_messages(messages, processor, device):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs.pop("token_type_ids", None)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def _get_single_token_id(processor, text: str) -> int:
    ids = processor.tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"'{text}' maps to {len(ids)} tokens: {ids}")
    return ids[0]


def _create_diff_pixel_values_qwen(
    frames_tchw: torch.Tensor,
    img_proc,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Differentiable Qwen2/3-VL image preprocessing."""
    N, C, H, W = frames_tchw.shape
    patch_size = img_proc.patch_size
    temporal_patch_size = img_proc.temporal_patch_size
    merge_size = img_proc.merge_size

    x = (frames_tchw + 1.0) * 0.5
    x = x.clamp(0, 1)

    if H != target_h or W != target_w:
        x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)

    mean = torch.tensor(img_proc.image_mean, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(img_proc.image_std, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std

    x = x.unsqueeze(1).repeat(1, temporal_patch_size, 1, 1, 1)

    grid_h = target_h // patch_size
    grid_w = target_w // patch_size

    x = x.reshape(
        N, temporal_patch_size, C,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    x = x.reshape(-1, C * temporal_patch_size * patch_size * patch_size)
    return x


def compute_qwen_reward(
    pixels_tchw: torch.Tensor,
    qwen_model,
    qwen_processor,
    num_sample_frames: int = 5,
    device: torch.device = torch.device("cuda"),
    target_success: bool = True,
    prompt: str = QWEN_ZERO_SHOT_PROMPT,
    yes_token_id: Optional[int] = None,
    no_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Differentiable single-view Qwen3-VL reward (5 frames per inference)."""
    T = pixels_tchw.shape[0]
    if T <= num_sample_frames:
        indices = list(range(T))
        while len(indices) < num_sample_frames:
            indices.append(T - 1)
    else:
        indices = np.linspace(0, T - 1, num=num_sample_frames, dtype=int).tolist()

    idx_tensor = torch.tensor(indices, device=pixels_tchw.device, dtype=torch.long)
    sampled = pixels_tchw.index_select(0, idx_tensor)

    pil_frames: List[Image.Image] = []
    with torch.no_grad():
        for i in range(sampled.shape[0]):
            arr = ((sampled[i].detach().cpu().float() + 1) * 0.5 * 255
                   ).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            pil_frames.append(Image.fromarray(arr))

    messages = _format_qwen_messages(pil_frames, prompt)
    inputs = _preprocess_qwen_messages(messages, qwen_processor, device)

    img_proc = qwen_processor.image_processor
    patch_size = img_proc.patch_size
    image_grid_thw = inputs.get("image_grid_thw") or inputs.get("video_grid_thw")
    if image_grid_thw is None:
        raise RuntimeError("Processor returned neither image_grid_thw nor video_grid_thw")

    grid_h = int(image_grid_thw[0, 1])
    grid_w = int(image_grid_thw[0, 2])
    target_h = grid_h * patch_size
    target_w = grid_w * patch_size

    diff_pv = _create_diff_pixel_values_qwen(sampled, img_proc, target_h, target_w, device)
    pv_key = "pixel_values" if "pixel_values" in inputs else "pixel_values_videos"
    proc_pv = inputs[pv_key]
    assert diff_pv.shape == proc_pv.shape
    inputs[pv_key] = diff_pv.to(dtype=proc_pv.dtype)

    outputs = qwen_model(**inputs, return_dict=True)
    logits = outputs.logits[0, -1, :].float()

    if yes_token_id is None:
        yes_token_id = _get_single_token_id(qwen_processor, "Yes")
    if no_token_id is None:
        no_token_id = _get_single_token_id(qwen_processor, "No")

    log_probs = F.log_softmax(logits, dim=-1)
    probs = logits.softmax(dim=-1)
    p_yes = probs[yes_token_id]
    p_no = probs[no_token_id]

    if target_success:
        reward = log_probs[no_token_id] - log_probs[yes_token_id]
    else:
        reward = log_probs[yes_token_id] - log_probs[no_token_id]

    info = {
        "p_yes": p_yes.item(),
        "p_no": p_no.item(),
        "margin": (log_probs[no_token_id] - log_probs[yes_token_id]).item(),
    }
    return reward, info


# ══════════════════════════════════════════════════════════════════════════
#  Multi-view Qwen reward
# ══════════════════════════════════════════════════════════════════════════

QWEN_ZERO_SHOT_PROMPT_MULTI_VIEW = (
    "The above video shows a robot manipulation trajectory that "
    "completes the following task: pick up an object.\n"
    "Is the robot successfully completing the task? "
    "Answer with a single word: Yes or No."
)

VIEW_LABELS = ["Left camera view:", "Right camera view:", "Wrist camera view:"]


def _format_qwen_messages_multi_view(
    view_images: List[List[Image.Image]],
    prompt: str,
) -> list:
    content = []
    for label, images in zip(VIEW_LABELS, view_images):
        content.append({"type": "text", "text": label})
        content.extend(_image_entry(img) for img in images)
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def compute_qwen_reward_multi_view(
    pixels_per_view: List[torch.Tensor],
    qwen_model,
    qwen_processor,
    num_sample_frames: int = 5,
    device: torch.device = torch.device("cuda"),
    target_success: bool = True,
    prompt: str = QWEN_ZERO_SHOT_PROMPT_MULTI_VIEW,
    yes_token_id: Optional[int] = None,
    no_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Differentiable multi-view Qwen3-VL reward."""
    num_views = len(pixels_per_view)

    all_sampled: List[torch.Tensor] = []
    all_pil_per_view: List[List[Image.Image]] = []
    for vi in range(num_views):
        T = pixels_per_view[vi].shape[0]
        if T <= num_sample_frames:
            indices = list(range(T))
            while len(indices) < num_sample_frames:
                indices.append(T - 1)
        else:
            indices = np.linspace(0, T - 1, num=num_sample_frames, dtype=int).tolist()

        idx_tensor = torch.tensor(indices, device=pixels_per_view[vi].device, dtype=torch.long)
        sampled = pixels_per_view[vi].index_select(0, idx_tensor)
        all_sampled.append(sampled)

        pil_frames: List[Image.Image] = []
        with torch.no_grad():
            for i in range(sampled.shape[0]):
                arr = ((sampled[i].detach().cpu().float() + 1) * 0.5 * 255
                       ).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                pil_frames.append(Image.fromarray(arr))
        all_pil_per_view.append(pil_frames)

    messages = _format_qwen_messages_multi_view(all_pil_per_view, prompt)
    inputs = _preprocess_qwen_messages(messages, qwen_processor, device)

    img_proc = qwen_processor.image_processor
    patch_size = img_proc.patch_size
    image_grid_thw = inputs.get("image_grid_thw") or inputs.get("video_grid_thw")
    if image_grid_thw is None:
        raise RuntimeError("Processor returned neither image_grid_thw nor video_grid_thw")

    grid_h = int(image_grid_thw[0, 1])
    grid_w = int(image_grid_thw[0, 2])
    target_h = grid_h * patch_size
    target_w = grid_w * patch_size

    all_frames = torch.cat(all_sampled, dim=0)
    diff_pv = _create_diff_pixel_values_qwen(all_frames, img_proc, target_h, target_w, device)
    pv_key = "pixel_values" if "pixel_values" in inputs else "pixel_values_videos"
    proc_pv = inputs[pv_key]
    assert diff_pv.shape == proc_pv.shape
    inputs[pv_key] = diff_pv.to(dtype=proc_pv.dtype)

    outputs = qwen_model(**inputs, return_dict=True)
    logits = outputs.logits[0, -1, :].float()

    if yes_token_id is None:
        yes_token_id = _get_single_token_id(qwen_processor, "Yes")
    if no_token_id is None:
        no_token_id = _get_single_token_id(qwen_processor, "No")

    log_probs = F.log_softmax(logits, dim=-1)
    probs = logits.softmax(dim=-1)
    p_yes = probs[yes_token_id]
    p_no = probs[no_token_id]

    if target_success:
        reward = log_probs[yes_token_id] - log_probs[no_token_id]
    else:
        reward = log_probs[no_token_id] - log_probs[yes_token_id]

    info = {
        "p_yes": p_yes.item(),
        "p_no": p_no.item(),
        "margin": (log_probs[yes_token_id] - log_probs[no_token_id]).item(),
    }
    return reward, info
