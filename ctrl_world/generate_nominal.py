#!/usr/bin/env python3
"""
Generate nominal (no optimization) Ctrl-World rollouts and evaluate Qwen reward.

Runs interact_num forward passes with random noise (no gradient / no SGD), evaluates
Qwen at each step, updates conditioning from the last predicted frame, and saves:
  - run_dir/nominal.mp4          — all steps concatenated, 3 views side-by-side
  - run_dir/nominal_reward.mp4   — same video with reward overlay sparklines
  - run_dir/history.json         — per-step Qwen rewards

Usage:
    python ctrl_world/generate_nominal.py \
        --hdf5_path ctrl_world/example_data/traj_0001.hdf5
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from config import wm_args
from models.ctrl_world import CrtlWorld
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from wm_helpers import (
    normalize_bound,
    build_cond_latent_3views,
    forward_wm_with_noise,
    decode_latents_with_grad,
    load_hdf5_views,
)
from rewards import (
    compute_qwen_reward_multi_view,
    _get_single_token_id,
    QWEN_ZERO_SHOT_PROMPT_MULTI_VIEW,
)
from visualize import pixels_to_uint8_views, save_overlay_video


# ---------------------------------------------------------------------------
# Image → 3-view conditioning latent (same helper as run_steering.py)
# ---------------------------------------------------------------------------

def encode_single_image_3views(image_path, vae, target_h, target_w, device, dtype):
    """Load an image, copy it into all three camera slots, VAE-encode each."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img_rgb).to(dtype).to(device)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1   # (1, 3, H, W) in [-1, 1]

    with torch.no_grad():
        latent_dist = vae.encode(x).latent_dist
        latent = latent_dist.sample().mul_(vae.config.scaling_factor)
    latent = latent.to(device=device, dtype=dtype)
    return [latent[0]] * 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Nominal Ctrl-World rollouts with Qwen reward evaluation")

    p.add_argument("--hdf5_path", default=None,
                   help="HDF5 trajectory file; camera_0/1/2 used as 3-view conditioning")
    p.add_argument("--image_path", default=None,
                   help="Single conditioning image (copied into all 3 camera views)")
    p.add_argument("--hdf5_frame_idx", type=int, default=0,
                   help="Frame index to read from the HDF5 trajectory (default: 0)")

    p.add_argument("--instruction", default=None,
                   help="Task description (default: from steering_config.yaml task.instruction)")
    p.add_argument("--qwen_prompt", default=None,
                   help="Qwen Yes/No prompt (default: from steering_config.yaml task.qwen_prompt)")

    p.add_argument("--steering_config", default=os.path.join(SCRIPT_DIR, "steering_config.yaml"))
    p.add_argument("--interact_num", type=int, default=None,
                   help="Number of rollout steps (overrides cfg.optim.interact_num)")
    p.add_argument("--save_dir", default=None,
                   help="Output directory (default: outputs/ctrl_world_nominal)")
    return p.parse_args()


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.steering_config)

    if args.hdf5_path is None and args.image_path is None:
        raise ValueError("Provide exactly one of --hdf5_path or --image_path")
    if args.hdf5_path is not None and args.image_path is not None:
        raise ValueError("Provide exactly one of --hdf5_path or --image_path, not both")

    # interact_num: CLI > config (default 15)
    interact_num = args.interact_num if args.interact_num is not None else int(cfg.optim.get("interact_num", 15))

    # save_dir: CLI > config.output.nominal_save_dir > hardcoded default
    if args.save_dir is not None:
        save_dir_raw = args.save_dir
    else:
        save_dir_raw = cfg.output.get("nominal_save_dir", "outputs/ctrl_world_nominal")

    # Resolve instruction and qwen_prompt: CLI > config defaults
    task_cfg = cfg.get("task", {})
    instruction = args.instruction or task_cfg.get("instruction", "")
    if not instruction:
        raise ValueError("--instruction or task.instruction in steering_config.yaml must be set")
    qwen_prompt_arg = args.qwen_prompt or task_cfg.get("qwen_prompt", QWEN_ZERO_SHOT_PROMPT_MULTI_VIEW)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(cfg.optim.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    save_dir = _resolve(save_dir_raw, SCRIPT_DIR)
    run_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    logging.info("Run dir: %s", run_dir)

    # ── Load Ctrl-World ────────────────────────────────────────────────
    logging.info("Loading Ctrl-World ...")
    wm_config = wm_args(task_type=cfg.model.task_type)
    wm_config.num_frames = cfg.model.num_frames

    model = CrtlWorld(wm_config)
    model.load_state_dict(torch.load(wm_config.ckpt_path, map_location="cpu"))
    model.to(device).to(wm_config.dtype).eval()
    for p in model.parameters():
        p.requires_grad = False
    logging.info("Ctrl-World loaded.")

    pipeline = model.pipeline
    vae = pipeline.vae
    dtype = wm_config.dtype
    num_frames = cfg.model.num_frames
    num_history = wm_config.num_history

    with open(wm_config.data_stat_path, "r") as f:
        data_stat = json.load(f)
        state_p01 = np.array(data_stat["state_01"])[None, :]
        state_p99 = np.array(data_stat["state_99"])[None, :]

    # ── Load Qwen ──────────────────────────────────────────────────────
    logging.info("Loading Qwen: %s", cfg.qwen.model_name)
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.qwen.model_name, torch_dtype=torch.float16, device_map="auto",
    ).eval()
    for p in qwen_model.parameters():
        p.requires_grad_(False)
    qwen_processor = AutoProcessor.from_pretrained(cfg.qwen.model_name)
    yes_token_id = _get_single_token_id(qwen_processor, "Yes")
    no_token_id = _get_single_token_id(qwen_processor, "No")
    logging.info("Qwen ready. Yes=%d No=%d", yes_token_id, no_token_id)
    logging.info("Qwen prompt: %r", qwen_prompt_arg)

    # ── Conditioning latent (HDF5 or single image) ────────────────────
    if args.hdf5_path is not None:
        logging.info("Encoding 3-view conditioning from HDF5: %s (frame %d)",
                     args.hdf5_path, args.hdf5_frame_idx)
        view_latents = load_hdf5_views(
            args.hdf5_path, frame_idx=args.hdf5_frame_idx,
            device=device, dtype=dtype, vae=vae,
            scaling_factor=vae.config.scaling_factor,
        )
    else:
        logging.info("Encoding single image into 3-view conditioning latent")
        view_latents = encode_single_image_3views(
            args.image_path, vae, wm_config.height, wm_config.width, device, dtype,
        )
    first_latent = build_cond_latent_3views(
        view_latents, target_height_total=72, target_w=40, device=device,
    )

    # History: repeat initial frame's latent
    his_cond = [first_latent for _ in range(num_history * 4)]
    zero_eef = np.zeros((1, wm_config.action_dim), dtype=np.float32)
    his_eef = [zero_eef for _ in range(num_history * 4)]

    cartesian_pose = np.zeros((wm_config.pred_step, wm_config.action_dim), dtype=np.float32)
    history_idx = [-1, -1, -1, -1, -1, -1]
    his_pose = np.concatenate([his_eef[i] for i in history_idx], axis=0)
    action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
    action_norm = normalize_bound(action_cond, state_p01, state_p99)
    action_tensor = torch.tensor(action_norm).unsqueeze(0).to(device, dtype)

    his_cond_input = torch.cat([his_cond[i] for i in history_idx], dim=0).unsqueeze(0)
    current_latent = his_cond[-1]

    if wm_config.text_cond:
        text_token = model.action_encoder(
            action_tensor, instruction, model.tokenizer, model.text_encoder,
        )
    else:
        text_token = model.action_encoder(action_tensor)

    # ── Nominal rollout loop ───────────────────────────────────────────
    history = []
    all_view_frames = [[] for _ in range(3)]   # per-view list of (num_frames, H, W, 3)
    step_p_yes_values = []

    for step in range(1, interact_num + 1):
        logging.info("=== Nominal Step %d/%d ===", step, interact_num)

        with torch.no_grad():
            noise = torch.randn(1, num_frames, 4, 72, 40, device=device, dtype=dtype)
            latents = forward_wm_with_noise(
                model, pipeline, action_tensor, current_latent,
                his_cond_input, text_token, wm_config,
                custom_noise=noise, device=device, dtype=dtype,
            )
            pixels = decode_latents_with_grad(
                vae, latents, decode_chunk_size=wm_config.decode_chunk_size,
            )

        num_views = pixels.shape[0]

        with torch.no_grad():
            qr, qwen_info = compute_qwen_reward_multi_view(
                [pixels[vi] for vi in range(num_views)],
                qwen_model, qwen_processor,
                num_sample_frames=cfg.qwen.num_frames,
                device=device,
                target_success=task_cfg.get("target_success", True),
                prompt=qwen_prompt_arg,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )

        reward_val = qr.item()
        p_yes_val = qwen_info["p_yes"]
        logging.info(
            "  Reward=%.4f  P(Yes)=%.3f  P(No)=%.3f",
            reward_val, p_yes_val, qwen_info["p_no"],
        )

        pixels_cpu = pixels.detach().cpu()
        view_arrays = pixels_to_uint8_views(pixels_cpu.numpy(), num_views=3)
        for vi in range(3):
            all_view_frames[vi].append(view_arrays[vi])

        step_p_yes_values.append(p_yes_val)
        history.append({
            "step": step,
            "qwen_reward": reward_val,
            "p_yes": p_yes_val,
            "p_no": qwen_info["p_no"],
            "margin": qwen_info["margin"],
        })

        # Update conditioning from last predicted frame
        last_frame_latents = [latents[v, -1, :, :, :] for v in range(3)]  # 3x (4,24,40)
        current_latent = build_cond_latent_3views(
            last_frame_latents, target_height_total=72, target_w=40,
            device=device, dtype=dtype,
        )
        his_cond[-1] = current_latent
        his_cond_input = torch.cat([his_cond[i] for i in history_idx], dim=0).unsqueeze(0)

        del pixels, pixels_cpu, latents, noise
        torch.cuda.empty_cache()

    # ── Save nominal video (no overlay) ───────────────────────────────
    concat_view_frames = []
    for vi in range(3):
        if all_view_frames[vi]:
            concat_view_frames.append(np.concatenate(all_view_frames[vi], axis=0))

    if concat_view_frames:
        nominal_cat = np.concatenate(concat_view_frames, axis=-2)  # (T_total, H, 3*W, 3)
        nominal_path = os.path.join(run_dir, "nominal.mp4")
        imageio.mimwrite(nominal_path, nominal_cat, fps=cfg.output.fps, codec="libx264")
        logging.info("Nominal video saved to %s", nominal_path)

        # ── Overlay reward video ───────────────────────────────────────
        total_T = concat_view_frames[0].shape[0]
        p_yes_curve = []
        for p_yes in step_p_yes_values:
            p_yes_curve.extend([p_yes] * num_frames)
        p_yes_curve = p_yes_curve[:total_T]

        reward_curves = {"Qwen p(Yes)": p_yes_curve}
        overlay_path = os.path.join(run_dir, "nominal_reward.mp4")
        save_overlay_video(
            concat_view_frames,
            [reward_curves] * 3,
            instruction,
            overlay_path,
            fps=cfg.output.fps,
        )

    # ── Save history ───────────────────────────────────────────────────
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logging.info("Done. Results in %s", run_dir)


if __name__ == "__main__":
    main()
