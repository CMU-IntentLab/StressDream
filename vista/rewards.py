"""Vista reward functions for noise optimization.

X-CLIP (default) and optional Qwen2.5-VL reward.

Qwen preprocessing follows vista_robust/dsrl/qwen_test_qa.py:
- pixels [T, C, H, W] in [-1, 1]  →  convert to [0, 255] THWC
- bilinear resize → differentiable_preprocess → patch tokens
Gradients flow back to pixels via differentiable_preprocess.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Default X-CLIP prompts (driving distance to lead vehicle)
DEFAULT_XCLIP_PROMPTS = [
    "Distance to the front vehicle is increasing",
    "Distance to the front vehicle is decreasing",
    "Distance to the front vehicle is similar",
]
DEFAULT_XCLIP_TARGET_IDX = 0  # maximise "increasing"

# Default Qwen2.5-VL prompt for Vista
DEFAULT_QWEN_PROMPT = "The lead vehicle getting further from the view of the front camera.\n"
DEFAULT_QWEN_MODEL = "WoWolf/Qwen2_5vl-7b-fm-tuned"


# ── X-CLIP ────────────────────────────────────────────────────────────────────

def _sample_frame_indices_random(clip_len, frame_sample_rate, seg_len, start_offset=0):
    converted_len = int(clip_len * frame_sample_rate)
    max_start = max(seg_len - converted_len, 0)
    start_idx = start_offset % (max_start + 1)
    end_idx = start_idx + converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, min(end_idx - 1, seg_len - 1)).astype(np.int64)
    return indices


def _preprocess_frames_torch(frames_tchw, processor, device):
    x = (frames_tchw + 1.0) * 0.5
    x = x.clamp(0, 1)
    img_proc = getattr(processor, "image_processor", None) or getattr(processor, "feature_extractor", None)
    if img_proc is None:
        raise AttributeError("Processor has neither image_processor nor feature_extractor.")

    size_cfg = getattr(img_proc, "size", 224)
    if hasattr(size_cfg, "get"):   # dict or transformers SizeDict
        shortest_edge = size_cfg.get("shortest_edge", size_cfg.get("height", 224))
    else:
        shortest_edge = int(size_cfg)

    crop_cfg = getattr(img_proc, "crop_size", None)
    if crop_cfg is None:
        crop_h = crop_w = shortest_edge
    elif hasattr(crop_cfg, "get"):   # dict or transformers SizeDict
        crop_h = crop_cfg.get("height", shortest_edge)
        crop_w = crop_cfg.get("width", shortest_edge)
    else:
        crop_h = crop_w = int(crop_cfg)

    x = F.interpolate(x, size=(crop_h, crop_w), mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor(getattr(img_proc, "image_mean"), device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(getattr(img_proc, "image_std"), device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def compute_xclip_reward(
    pixels: torch.Tensor,
    vlm_model,
    vlm_processor,
    text_prompts: List[str],
    target_text_idx: int,
    frame_sampling_mode: str,
    num_random_starts: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """X-CLIP reward: average probability of target_text_idx over sampled windows.

    Returns (avg_reward, avg_probs_over_all_prompts).
    """
    T = pixels.shape[0]
    rewards, probs_list = [], []
    text_inputs = vlm_processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    def _score(idx_tensor):
        window = pixels.index_select(0, idx_tensor)
        pv = _preprocess_frames_torch(window, vlm_processor, device)
        out = vlm_model(pixel_values=pv, **text_inputs, return_dict=True)
        probs = out.logits_per_video.softmax(dim=1)
        rewards.append(probs[0, target_text_idx])
        probs_list.append(probs[0])

    if frame_sampling_mode == "overlapping":
        for start in range(T - 7):
            idx = torch.arange(start, start + 8, device=device, dtype=torch.long)
            if idx[-1] < T:
                _score(idx)
    elif frame_sampling_mode == "random":
        for i in range(num_random_starts):
            start_offset = int((T - 8) * i / max(1, num_random_starts - 1))
            indices = _sample_frame_indices_random(8, 1, T, start_offset)
            _score(torch.tensor(indices, device=device, dtype=torch.long))
    elif frame_sampling_mode == "stride3":
        stride, clip_len = 3, 8
        for start in range(4):
            all_idx = torch.arange(start, T, stride, device=device)
            if all_idx.numel() >= clip_len:
                _score(all_idx[:clip_len].long())
    else:
        raise ValueError(f"Unknown frame_sampling_mode: {frame_sampling_mode}")

    avg_reward = torch.stack(rewards).mean()
    avg_probs = torch.stack(probs_list).mean(dim=0)
    return avg_reward, avg_probs


# ── Qwen2.5-VL (Vista single-view) ───────────────────────────────────────────
# Adapted from vista_robust/dsrl/qwen_test_qa.py.
# Input format: pixels [T, C, H, W] in [-1, 1].
# Converted to [0, 255] THWC then processed by differentiable_preprocess (bilinear).

def _get_processor_vision_config(processor) -> dict:
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return {
            "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "rescale_factor": 1.0 / 255.0,
        }
    return {
        "patch_size": getattr(ip, "patch_size", 14),
        "temporal_patch_size": getattr(ip, "temporal_patch_size", 2),
        "merge_size": getattr(ip, "merge_size", 2),
        "image_mean": getattr(ip, "image_mean", [0.48145466, 0.4578275, 0.40821073]),
        "image_std": getattr(ip, "image_std", [0.26862954, 0.26130258, 0.27577711]),
        "rescale_factor": getattr(ip, "rescale_factor", 1.0 / 255.0),
    }


def differentiable_preprocess(
    frames: torch.Tensor,
    target_height: int,
    target_width: int,
    processor,
    device,
) -> torch.Tensor:
    """Map (T, H, W, 3) frames in [0, 255] → Qwen patch tokens (num_patches, patch_dim).

    Differentiable: gradients flow back to frames.
    Bilinear resize matches the upstream Vista qwen_test_qa.py implementation.
    """
    cfg = _get_processor_vision_config(processor)
    patch_size = cfg["patch_size"]
    temporal_patch_size = cfg["temporal_patch_size"]
    merge_size = cfg["merge_size"]
    mean = torch.tensor(cfg["image_mean"], device=device, dtype=frames.dtype).view(1, 1, 1, 3)
    std = torch.tensor(cfg["image_std"], device=device, dtype=frames.dtype).view(1, 1, 1, 3)
    scale = cfg["rescale_factor"]

    x = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
    x = F.interpolate(x, size=(target_height, target_width), mode="bilinear", align_corners=False)
    x = x * scale
    x = (x - mean.permute(0, 3, 1, 2)) / std.permute(0, 3, 1, 2)

    T, C, H, W = x.shape
    factor = patch_size * merge_size
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"After resize {target_height}x{target_width} must be divisible by "
            f"patch_size*merge_size={factor}; got H={H} W={W}"
        )

    remainder = T % temporal_patch_size
    if remainder != 0:
        x = torch.cat([x, x[-1:].repeat(temporal_patch_size - remainder, 1, 1, 1)], dim=0)

    T_pad = x.shape[0]
    grid_t = T_pad // temporal_patch_size
    grid_h = H // patch_size
    grid_w = W // patch_size

    patches = x.reshape(
        grid_t, temporal_patch_size, C,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(
        grid_t * grid_h * grid_w,
        C * temporal_patch_size * patch_size * patch_size,
    )


def _single_token_id(processor, text: str) -> int:
    ids = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"'{text}' must be a single token, got {len(ids)}: {ids}")
    return ids[0]


def _format_messages_qwen(
    images: List[Image.Image], question: str, height: int = 224, width: int = 224,
) -> list:
    entries = [
        {"type": "image", "image": img.resize((width, height)),
         "min_pixels": height * width, "max_pixels": height * width}
        for img in images
    ]
    return [{"role": "user", "content": entries + [{"type": "text", "text": question}]}]


def _preprocess_qwen(messages, processor, device) -> dict:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    all_images = [
        item["image"]
        for msg in messages
        for item in msg.get("content", [])
        if isinstance(item, dict) and item.get("type") == "image"
    ]
    batch = processor(text=[text], images=all_images, padding=True, return_tensors="pt")
    return batch.to(device)


def _compute_qwen_single_window(
    model, processor, pixels_frames, device,
    target_height, target_width, inputs_template,
    target_answer, anchor_answer, use_log_prob, use_margin, N, query_start,
) -> torch.Tensor:
    """One forward pass for N frames; returns differentiable scalar score."""
    resized = F.interpolate(
        pixels_frames, size=(target_height, target_width), mode="bilinear", align_corners=False,
    )
    frames_thwc = (resized.permute(0, 2, 3, 1) + 1.0) * 0.5 * 255.0
    frames_thwc = frames_thwc.clamp(0.0, 255.0)

    pv_list = [
        differentiable_preprocess(frames_thwc[t:t+1], target_height, target_width, processor, device)
        for t in range(N)
    ]
    query_pv = torch.cat(pv_list, dim=0).to(inputs_template["pixel_values"].dtype)

    pv_full = (
        torch.cat([inputs_template["pixel_values"][:query_start].detach(), query_pv], dim=0)
        if query_start > 0 else query_pv
    )
    logits = model(**{**inputs_template, "pixel_values": pv_full}, return_dict=True).logits
    last = logits[0, -1, :].float()

    dist = F.log_softmax(last, dim=-1) if use_log_prob else F.softmax(last, dim=-1)
    tid = _single_token_id(processor, target_answer)
    score = dist[tid]
    if use_margin and anchor_answer:
        score = score - dist[_single_token_id(processor, anchor_answer)]
    return score


def compute_qwen_reward_vista(
    pixels: torch.Tensor,
    model,
    processor,
    question: str,
    device,
    num_frames: int = 8,
    target_height: int = 224,
    target_width: int = 224,
    target_answer: str = "Yes",
    anchor_answer: str = "No",
    use_log_prob: bool = True,
    use_margin: bool = True,
    frame_sampling_mode: str = "stride3",
    stride3_num_starts: int = 4,
) -> Tuple[torch.Tensor, Dict]:
    """Qwen2.5-VL Yes/No margin reward for Vista (single view).

    pixels: [T, C, H, W] in [-1, 1]. Differentiable.
    """
    T = pixels.shape[0]

    if frame_sampling_mode in ("stride3", "stride3_sample"):
        stride, clip_len = 3, num_frames
        valid_starts = [
            s for s in range(stride3_num_starts)
            if len(torch.arange(s, T, stride)) >= clip_len
        ]
        if not valid_starts:
            raise ValueError(f"stride3: no valid start (T={T}, stride={stride}, clip_len={clip_len})")

        # Build template inputs once (shared structure for all windows)
        dummy_np = np.zeros((clip_len, target_height, target_width, 3), dtype=np.uint8)
        dummy_pil = [Image.fromarray(dummy_np[i]) for i in range(clip_len)]
        messages = _format_messages_qwen(dummy_pil, question, target_height, target_width)
        inputs_template = _preprocess_qwen(messages, processor, device)
        grid_thw = inputs_template["image_grid_thw"]
        if torch.is_tensor(grid_thw):
            grid_thw = grid_thw.cpu().numpy()
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.reshape(1, -1)
        patches_per_img = [int(np.prod(grid_thw[i])) for i in range(len(grid_thw))]
        query_start = sum(patches_per_img) - sum(patches_per_img[-clip_len:])

        starts_to_run = (
            [random.choice(valid_starts)] if frame_sampling_mode == "stride3_sample"
            else valid_starts
        )
        scores = []
        for start in starts_to_run:
            all_idx = torch.arange(start, T, stride, device=pixels.device)
            pixels_frames = pixels.index_select(0, all_idx[:clip_len].long())
            scores.append(_compute_qwen_single_window(
                model, processor, pixels_frames, device,
                target_height, target_width, inputs_template,
                target_answer, anchor_answer, use_log_prob, use_margin,
                clip_len, query_start,
            ))
        avg_score = torch.stack(scores).mean() if len(scores) > 1 else scores[0]

        # Diagnostics (no grad)
        with torch.no_grad():
            s0 = starts_to_run[-1]
            last_idx = torch.arange(s0, s0 + clip_len * stride, stride, device=pixels.device)[:clip_len]
            resized = F.interpolate(
                pixels.index_select(0, last_idx.long()).detach(),
                size=(target_height, target_width), mode="bilinear", align_corners=False,
            )
            frames_thwc = (resized.permute(0, 2, 3, 1) + 1.0) * 0.5 * 255.0
            pv_list = [
                differentiable_preprocess(frames_thwc[t:t+1], target_height, target_width, processor, device)
                for t in range(clip_len)
            ]
            qpv = torch.cat(pv_list, dim=0).to(inputs_template["pixel_values"].dtype)
            pv_full = (
                torch.cat([inputs_template["pixel_values"][:query_start].detach(), qpv], dim=0)
                if query_start > 0 else qpv
            )
            logits = model(**{**inputs_template, "pixel_values": pv_full}, return_dict=True).logits
            probs = F.softmax(logits[0, -1, :].float(), dim=-1)
            tid = _single_token_id(processor, target_answer)
            aid = _single_token_id(processor, anchor_answer)

        info = {
            "p_yes": float(probs[tid]), "p_no": float(probs[aid]),
            "margin": float(probs[tid]) - float(probs[aid]),
            "num_windows": len(scores), "starts_used": starts_to_run,
        }
        return avg_score, info

    # uniform mode
    if T < num_frames:
        indices = list(range(T))
    else:
        indices = np.linspace(0, T - 1, num=num_frames, dtype=np.int64).tolist()
    N = len(indices)
    pixels_frames = pixels.index_select(0, torch.tensor(indices, device=pixels.device, dtype=torch.long))

    dummy_np = np.zeros((N, target_height, target_width, 3), dtype=np.uint8)
    dummy_pil = [Image.fromarray(dummy_np[i]) for i in range(N)]
    messages = _format_messages_qwen(dummy_pil, question, target_height, target_width)
    inputs = _preprocess_qwen(messages, processor, device)
    grid_thw = inputs["image_grid_thw"]
    if torch.is_tensor(grid_thw):
        grid_thw = grid_thw.cpu().numpy()
    if grid_thw.ndim == 1:
        grid_thw = grid_thw.reshape(1, -1)
    patches_per_img = [int(np.prod(grid_thw[i])) for i in range(len(grid_thw))]
    query_start = sum(patches_per_img) - sum(patches_per_img[-N:])

    resized = F.interpolate(
        pixels_frames, size=(target_height, target_width), mode="bilinear", align_corners=False,
    )
    frames_thwc = (resized.permute(0, 2, 3, 1) + 1.0) * 0.5 * 255.0
    frames_thwc = frames_thwc.clamp(0.0, 255.0)
    pv_list = [
        differentiable_preprocess(frames_thwc[t:t+1], target_height, target_width, processor, device)
        for t in range(N)
    ]
    query_pv = torch.cat(pv_list, dim=0).to(inputs["pixel_values"].dtype)
    pv_full = (
        torch.cat([inputs["pixel_values"][:query_start].detach(), query_pv], dim=0)
        if query_start > 0 else query_pv
    )
    logits = model(**{**inputs, "pixel_values": pv_full}, return_dict=True).logits
    last = logits[0, -1, :].float()
    dist = F.log_softmax(last, dim=-1) if use_log_prob else F.softmax(last, dim=-1)
    tid = _single_token_id(processor, target_answer)
    score = dist[tid]
    if use_margin and anchor_answer:
        score = score - dist[_single_token_id(processor, anchor_answer)]

    with torch.no_grad():
        probs = F.softmax(last, dim=-1)
        tid_ = _single_token_id(processor, target_answer)
        aid_ = _single_token_id(processor, anchor_answer)
    info = {
        "p_yes": float(probs[tid_]), "p_no": float(probs[aid_]),
        "margin": float(probs[tid_]) - float(probs[aid_]),
        "num_windows": 1,
    }
    return score, info
