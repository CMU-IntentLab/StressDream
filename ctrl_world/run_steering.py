#!/usr/bin/env python3
"""
Qwen noise steering for Ctrl-World.

Usage (HDF5 trajectory, uses instruction/qwen_prompt from steering_config.yaml):
    python ctrl_world/run_steering.py \
        --hdf5_path ctrl_world/example_data/traj_0001.hdf5

Usage (single image fallback, override instruction):
    python ctrl_world/run_steering.py \
        --image_path ctrl_world/example_images/scene.jpg \
        --instruction "pick up the object"

HDF5 path: full trajectory (eef_states + 3 VAE-encoded views) is loaded
once and the steering loop runs `interact_num` autoregressive rollout
windows driven by **real GT end-effector chunks** — matching
ctrl-world/scripts/oolong/iterative_noise_optim_oolong.py. After each step
the optimised last latent + ground-truth eef pose are appended to the
running history; the next window's action chunk is the next `pred_step`
GT eef states.

Single-image path: action chunk and eef history are zeroed and the
conditioning image is repeated across views and frames. Use this only
when no GT actions exist.

Qwen3-VL provides the optimisation reward (Yes/No log-prob margin on the
multi-view clip).
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from typing import List

import cv2
import einops
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
    load_hdf5_trajectory,
)
from rewards import (
    compute_qwen_reward_multi_view,
    _get_single_token_id,
    QWEN_ZERO_SHOT_PROMPT_MULTI_VIEW,
)
from wm_steer.regularizer import compute_regularizer_video
from visualize import pixels_to_uint8_views


# ---------------------------------------------------------------------------
# Image → 3-view conditioning latent
# ---------------------------------------------------------------------------

def encode_single_image_3views(
    image_path: str,
    vae,
    target_h: int,
    target_w: int,
    device,
    dtype,
):
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
        latent = latent_dist.sample().mul_(vae.config.scaling_factor)   # (1, 4, h, w)
    latent = latent.to(device=device, dtype=dtype)
    # 3 views = same image latent thrice
    return [latent[0]] * 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Qwen noise steering for Ctrl-World")
    # Conditioning input: HDF5 trajectory (3 distinct views) or single image (same image × 3).
    # Exactly one of --hdf5_path / --image_path must be provided.
    p.add_argument("--hdf5_path", default=None,
                   help="HDF5 trajectory file; camera_0/1/2 + ee_states/gripper_states "
                        "drive autoregressive rollout windows.")
    p.add_argument("--image_path", default=None,
                   help="Single conditioning image (copied into all 3 camera views, "
                        "zero actions/history — for use when no GT trajectory exists).")
    p.add_argument("--start_window", type=int, default=None,
                   help="Window offset into the HDF5 trajectory. "
                        "Default: max(0, total_windows - interact_num) (matches oolong).")

    p.add_argument("--instruction", default=None,
                   help="Task description (default: from steering_config.yaml task.instruction)")
    p.add_argument("--qwen_prompt", default=None,
                   help="Qwen Yes/No prompt (default: from steering_config.yaml task.qwen_prompt)")

    p.add_argument("--steering_config", default=os.path.join(SCRIPT_DIR, "steering_config.yaml"))
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--interact_num", type=int, default=None,
                   help="Number of rollout/interact steps (overrides cfg.optim.interact_num)")
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--seed", type=int, default=None)
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

    if args.iters is not None:
        cfg.optim.iters = args.iters
    if args.num_frames is not None:
        cfg.model.num_frames = args.num_frames
    if args.save_dir is not None:
        cfg.output.save_dir = args.save_dir
    if args.seed is not None:
        cfg.optim.seed = args.seed

    # interact_num: CLI > config (default 15)
    interact_num = args.interact_num if args.interact_num is not None else int(cfg.optim.get("interact_num", 15))

    # Resolve instruction and qwen_prompt: CLI > config defaults
    task_cfg = cfg.get("task", {})
    instruction = args.instruction or task_cfg.get("instruction", "")
    if not instruction:
        raise ValueError("--instruction or task.instruction in steering_config.yaml must be set")
    qwen_prompt_arg = args.qwen_prompt or task_cfg.get("qwen_prompt", QWEN_ZERO_SHOT_PROMPT_MULTI_VIEW)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.optim.seed)
    np.random.seed(cfg.optim.seed)
    random.seed(cfg.optim.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.optim.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    save_dir = _resolve(cfg.output.save_dir, SCRIPT_DIR)
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
    if hasattr(model.pipeline.vae, "enable_gradient_checkpointing"):
        model.pipeline.vae.enable_gradient_checkpointing()
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
    if hasattr(qwen_model, "gradient_checkpointing_enable"):
        qwen_model.gradient_checkpointing_enable()
    qwen_processor = AutoProcessor.from_pretrained(cfg.qwen.model_name)
    yes_token_id = _get_single_token_id(qwen_processor, "Yes")
    no_token_id = _get_single_token_id(qwen_processor, "No")
    logging.info("Qwen ready. Yes=%d No=%d", yes_token_id, no_token_id)
    logging.info("Qwen prompt: %r", qwen_prompt_arg)

    # ── Conditioning: HDF5 trajectory (GT eef + per-view latents) or single image ──
    history_idx = [-1, -1, -1, -1, -1, -1]
    pred_step = wm_config.pred_step
    action_dim = wm_config.action_dim

    if args.hdf5_path is not None:
        hdf5_cfg = cfg.get("hdf5", {})
        camera_order = list(hdf5_cfg.get("camera_order", [0, 1, 2]))
        rgb_skip = int(hdf5_cfg.get("rgb_skip", 3))
        center_crop_half = bool(hdf5_cfg.get("center_crop_half", True))

        logging.info("Loading full HDF5 trajectory: %s (cams=%s, rgb_skip=%d, crop_half=%s)",
                     args.hdf5_path, camera_order, rgb_skip, center_crop_half)
        eef_gt, video_latents, video_rgb = load_hdf5_trajectory(
            args.hdf5_path, vae=vae, device=device, dtype=dtype,
            camera_order=camera_order, rgb_skip=rgb_skip,
            center_crop_half=center_crop_half,
        )
        T_sub = video_latents[0].shape[0]
        total_windows = max(1, math.ceil((T_sub - 1) / (pred_step - 1)))

        # Match oolong: default start_window puts us on the LAST interact_num windows.
        start_window_cfg = args.start_window if args.start_window is not None else hdf5_cfg.get("start_window", None)
        if start_window_cfg is None:
            start_window = max(0, total_windows - interact_num)
        else:
            start_window = int(start_window_cfg)
        interact_num = min(interact_num, total_windows - start_window)
        logging.info("HDF5 frames=%d, total_windows=%d, start_window=%d, interact_num=%d",
                     T_sub, total_windows, start_window, interact_num)

        init_frame = int(start_window * (pred_step - 1))
        first_latent = build_cond_latent_3views(
            [v[init_frame] for v in video_latents],
            target_height_total=72, target_w=40, device=device,
        )
        initial_eef = eef_gt[init_frame:init_frame + 1]
    else:
        logging.info("Encoding single image into 3-view conditioning latent")
        view_latents = encode_single_image_3views(
            args.image_path, vae, wm_config.height, wm_config.width, device, dtype,
        )
        first_latent = build_cond_latent_3views(
            view_latents, target_height_total=72, target_w=40, device=device,
        )
        eef_gt = None
        video_rgb = None
        start_window = 0
        initial_eef = np.zeros((1, action_dim), dtype=np.float32)

    # History buffers (need num_history*4 entries so history_idx=-1 just reads the most recent).
    his_cond = [first_latent for _ in range(num_history * 4)]
    his_eef = [initial_eef.astype(np.float32) for _ in range(num_history * 4)]

    # ── Outer interact loop ────────────────────────────────────────────
    reg_cfg = cfg.regularizer
    history = []
    all_view_frames = [[] for _ in range(3)]
    last_noise = None  # will hold noise from the last step for saving

    for step in range(1, interact_num + 1):
        logging.info("=== Interact Step %d/%d ===", step, interact_num)

        # ── Per-step action chunk ────────────────────────────────────
        if eef_gt is not None:
            sid = int((start_window + step - 1) * (pred_step - 1))
            cartesian_pose = eef_gt[sid:sid + pred_step]
            if len(cartesian_pose) < pred_step:
                pad = np.repeat(cartesian_pose[-1:], pred_step - len(cartesian_pose), axis=0)
                cartesian_pose = np.concatenate([cartesian_pose, pad], axis=0)
        else:
            cartesian_pose = np.zeros((pred_step, action_dim), dtype=np.float32)

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

        # Fresh noise for this step (re-optimise from scratch)
        noise = torch.randn(
            1, num_frames, 4, 72, 40, device=device,
            dtype=torch.float32, requires_grad=True,
        )
        initial_noise = noise.detach().clone()
        optimizer = torch.optim.SGD([noise], lr=cfg.optim.lr, momentum=0.0)

        best_score = -float("inf")
        best_latents = None
        best_step_reward = -float("inf")
        best_step_p_yes = 0.0

        iter_history = []

        for iteration in range(1, cfg.optim.iters + 1):
            logging.info("  -- Step %d Iter %d/%d --", step, iteration, cfg.optim.iters)
            optimizer.zero_grad()

            if cfg.optim.detach_latents:
                with torch.no_grad():
                    latents = forward_wm_with_noise(
                        model, pipeline, action_tensor, current_latent,
                        his_cond_input, text_token, wm_config,
                        custom_noise=noise.detach().to(dtype),
                        device=device, dtype=dtype,
                    )
                latents = latents.detach().requires_grad_(True)
            else:
                latents = forward_wm_with_noise(
                    model, pipeline, action_tensor, current_latent,
                    his_cond_input, text_token, wm_config,
                    custom_noise=noise.to(dtype),
                    device=device, dtype=dtype,
                )

            pixels = decode_latents_with_grad(vae, latents, decode_chunk_size=wm_config.decode_chunk_size)
            pixels_cpu = pixels.detach().cpu()
            num_views = pixels.shape[0]

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
            (-cfg.optim.grad_scale * qr).backward(retain_graph=False)

            # Transfer latents.grad → noise via surrogate when detach_latents=True
            if cfg.optim.detach_latents and latents.grad is not None:
                g = latents.grad.detach().clone()
                latents.grad = None
                b = noise.shape[0]
                g = einops.rearrange(g, "(b m) f c h w -> b f c (m h) w", b=b, m=num_views).float()
                surrogate = (noise * g).sum()
                surrogate.backward()
                if noise.grad is not None:
                    torch.nn.utils.clip_grad_norm_([noise], cfg.optim.grad_clip_per_loss)
                    noise.grad.clamp_(-cfg.optim.grad_clamp, cfg.optim.grad_clamp)

            # Regulariser
            reg_loss, shrink_r, kl_r, std_r, std_perm_r, _, spec_loss = compute_regularizer_video(
                noise,
                kl_coeff=reg_cfg.kl_coeff,
                kl_coeff_spherical=reg_cfg.kl_coeff_spherical,
                std_coeff=reg_cfg.std_coeff,
                spectral_coeff=reg_cfg.spectral_coeff,
                std_permutation_coeff=reg_cfg.std_permutation_coeff,
                gram_normalize=reg_cfg.gram_normalize,
                spectral_threshold=reg_cfg.spectral_threshold,
                std_perm_activation=reg_cfg.std_perm_activation,
                std_perm_threshold=reg_cfg.std_perm_threshold,
                std_perm_include_patched=reg_cfg.std_perm_include_patched,
                num_gram_perms=reg_cfg.num_gram_perms,
            )
            reg_loss.backward()

            grad_norm = noise.grad.norm().item() if noise.grad is not None else 0.0
            norm_before = noise.norm().item()
            if noise.grad is not None:
                torch.nn.utils.clip_grad_norm_([noise], 20.0)
                noise.grad.clamp_(-0.5, 0.5)
            optimizer.step()

            # Hard renormalisation
            n_norm = noise.norm().item()
            tgt_norm = math.sqrt(noise.numel())
            if n_norm > 0 and abs(n_norm - tgt_norm) > cfg.optim.noise_norm_threshold:
                noise.data.copy_((noise.data - noise.mean()) / (noise.std() + 1e-10))

            norm_after = noise.norm().item()
            reward_val = qr.item()
            p_yes_val = qwen_info["p_yes"]

            if reward_val > best_score:
                best_score = reward_val
                best_latents = latents.detach().clone()
                best_step_reward = reward_val
                best_step_p_yes = p_yes_val

            logging.info(
                "  Reward=%.4f  P(Yes)=%.3f P(No)=%.3f  grad_norm=%.3f  noise %.2f→%.2f",
                reward_val, p_yes_val, qwen_info["p_no"],
                grad_norm, norm_before, norm_after,
            )

            iter_history.append({
                "iter": iteration,
                "qwen_reward": reward_val,
                "p_yes": p_yes_val,
                "p_no": qwen_info["p_no"],
                "margin": qwen_info["margin"],
                "noise_norm": norm_after,
                "noise_drift": (noise - initial_noise).norm().item(),
                "grad_norm": grad_norm,
                "spectral_loss": spec_loss.item(),
                "std_perm": std_perm_r.item(),
                "reg_loss": reg_loss.item(),
            })

            optimizer.zero_grad()
            del pixels, pixels_cpu
            torch.cuda.empty_cache()

        # ── End of inner loop: save best video for this step ──────────
        step_best_pixels_cpu = None
        if best_latents is not None:
            with torch.no_grad():
                bp = decode_latents_with_grad(
                    vae, best_latents, decode_chunk_size=wm_config.decode_chunk_size,
                )
                step_best_pixels_cpu = bp.cpu()

            view_arrays = pixels_to_uint8_views(step_best_pixels_cpu.float().numpy(), num_views=3)
            step_cat = np.concatenate(view_arrays, axis=-2)  # (T, H, 3*W, 3)
            # Prepend GT frames [init_frame, sid) so each video shows the full trajectory so far
            if video_rgb is not None:
                step_sid = int((start_window + step - 1) * (pred_step - 1))
                if step_sid > init_frame:
                    gt_prefix = np.concatenate(
                        [video_rgb[vi][init_frame:step_sid] for vi in range(3)], axis=-2
                    )  # (prefix_T, H, 3*W, 3)
                    step_cat = np.concatenate([gt_prefix, step_cat], axis=0)
            step_vpath = os.path.join(run_dir, f"step_{step:04d}_best.mp4")
            imageio.mimwrite(step_vpath, step_cat, fps=cfg.output.fps, codec="libx264")
            logging.info("Step %d best video → %s (reward=%.4f)", step, step_vpath, best_step_reward)

            for vi in range(3):
                all_view_frames[vi].append(view_arrays[vi])

        history.append({
            "step": step,
            "best_qwen_reward": best_step_reward,
            "best_p_yes": best_step_p_yes,
            "iters": iter_history,
        })

        # ── Update conditioning for next step (matches oolong) ────────
        if best_latents is not None:
            # best_latents shape: (3, num_frames, 4, 24, 40). Take the last
            # generated frame of each view as the next step's image cond.
            last_frame_latents = [best_latents[v, -1, :, :, :] for v in range(3)]
            new_latent = build_cond_latent_3views(
                last_frame_latents, target_height_total=72, target_w=40,
                device=device, dtype=dtype,
            )
            his_cond.append(new_latent)

        # Append the GT (or zero) end-of-window eef pose to history so the
        # next window's `his_pose` reflects what just happened.
        his_eef.append(cartesian_pose[pred_step - 1:pred_step])

        last_noise = noise.detach().cpu()
        del best_latents
        torch.cuda.empty_cache()

    # ── Full concatenated video (all steps, no graph) ─────────────────
    if any(len(f) > 0 for f in all_view_frames):
        all_frames = np.concatenate(
            [np.concatenate(all_view_frames[vi], axis=0) for vi in range(3)],
            axis=-2,
        )  # (total_T, H, 3*W, 3)
        full_vpath = os.path.join(run_dir, "full_steered.mp4")
        imageio.mimwrite(full_vpath, all_frames, fps=cfg.output.fps, codec="libx264")
        logging.info("Full video → %s", full_vpath)

    # ── Save outputs ───────────────────────────────────────────────────
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if last_noise is not None:
        torch.save(last_noise, os.path.join(run_dir, "optimized_noise.pt"))

    logging.info("Done. Results in %s", run_dir)


if __name__ == "__main__":
    main()
