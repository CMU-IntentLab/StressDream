#!/usr/bin/env python3
"""
Single-image X-CLIP (+ optional Qwen2.5-VL) noise steering for Vista.

Usage:
    # X-CLIP only (default prompts from steering_config.yaml)
    python vista/run_steering.py \
        --image_path vista/example_images/truck.jpg

    # Override prompts and target index
    python vista/run_steering.py \
        --image_path vista/example_images/truck.jpg \
        --prompts "a truck blocks the road,a clear road ahead" \
        --target_idx 0

    # Add optional Qwen2.5-VL reward on top of X-CLIP
    python vista/run_steering.py \
        --image_path vista/example_images/truck.jpg \
        --use_qwen

Outputs (under steering_config.yaml `output.save_dir`):
    iter_<n>.mp4              # rolled-out video per iteration
    history.json              # per-iteration rewards / norms
    optimized_noise.pt        # final optimised noise tensor

Optional driving-scenario conditioning:
    --trajectory_json path/to/traj.json   # JSON list of {"x": float, "y": float}
    --command 3                           # int command index
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from wm_helpers import (
    init_vista_model,
    get_latent_shape_vista,
    load_img,
    save_video,
    init_embedder_options,
    get_condition,
    load_model,
    unload_model,
    init_sampling,
    set_lowvram_mode,
)
from rewards import (
    compute_xclip_reward,
    compute_qwen_reward_vista,
    DEFAULT_XCLIP_PROMPTS,
    DEFAULT_XCLIP_TARGET_IDX,
    DEFAULT_QWEN_PROMPT,
    DEFAULT_QWEN_MODEL,
)
from wm_steer.regularizer import compute_regularizer_video


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "example_images", "truck.jpg")


def parse_args():
    p = argparse.ArgumentParser(description="Single-image X-CLIP noise steering for Vista")
    p.add_argument("--image_path", default=DEFAULT_IMAGE_PATH,
                   help="Path to a single conditioning image. "
                        "Default: vista/example_images/truck.jpg")
    p.add_argument("--prompts", default=None,
                   help="Comma-separated prompt list or JSON file. "
                        "Default: prompts from steering_config.yaml (vlm.default_prompts)")
    p.add_argument("--target_idx", type=int, default=None,
                   help="Index in the prompt list to maximise. "
                        "Default: vlm.default_target_idx in steering_config.yaml")

    # Optional driving-scenario conditioning
    p.add_argument("--trajectory_json", default=None,
                   help="Optional JSON file: list of {x, y} waypoints")
    p.add_argument("--command", type=int, default=None,
                   help="Optional command index for Vista's command embedder")

    # Optional Qwen reward
    p.add_argument("--use_qwen", action="store_true", default=False,
                   help="Add Qwen2.5-VL reward on top of X-CLIP (default: off)")
    p.add_argument("--qwen_model", default=None,
                   help=f"Qwen model name (default: {DEFAULT_QWEN_MODEL})")
    p.add_argument("--qwen_prompt", default=None,
                   help="Qwen prompt (default: from steering_config.yaml)")
    p.add_argument("--qwen_coeff", type=float, default=None,
                   help="Qwen reward weight (default: from steering_config.yaml)")
    p.add_argument("--qwen_num_frames", type=int, default=None,
                   help="Frames sampled per Qwen window (default: from steering_config.yaml)")
    p.add_argument("--frame_sampling_mode", default=None,
                   help="stride3 | overlapping | random (default: from steering_config.yaml)")

    # Config + overrides
    p.add_argument("--steering_config", default=os.path.join(SCRIPT_DIR, "steering_config.yaml"))
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--low_vram", action="store_true", default=None)
    p.add_argument("--no_regularizer", action="store_true", default=False,
                   help="Zero out all regularizer coefficients (ablation: pure reward gradient only).")
    return p.parse_args()


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def _load_prompts(arg: str) -> List[str]:
    if os.path.isfile(arg):
        with open(arg) as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
            raise ValueError(f"{arg} must contain a flat JSON list of strings")
        return data
    return [s.strip() for s in arg.split(",") if s.strip()]


def _load_trajectory(arg: str, device) -> torch.Tensor:
    with open(arg) as f:
        waypoints = json.load(f)
    if len(waypoints) <= 2:
        raise ValueError("trajectory needs at least 3 waypoints (origin + 2 targets)")
    coords = []
    for wp in waypoints[2:]:
        coords.append(-wp["y"])
        coords.append(wp["x"])
    return torch.tensor(coords, device=device, dtype=torch.float32)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.steering_config)

    if args.iters is not None:
        cfg.optim.iters = args.iters
    if args.save_dir is not None:
        cfg.output.save_dir = args.save_dir
    if args.seed is not None:
        cfg.optim.seed = args.seed
    if args.low_vram is not None:
        cfg.model.low_vram = bool(args.low_vram)

    # Resolve prompts and target index from CLI or config defaults
    if args.prompts is not None:
        prompts = _load_prompts(args.prompts)
    else:
        prompts = list(cfg.vlm.default_prompts)
    target_idx = args.target_idx if args.target_idx is not None else cfg.vlm.default_target_idx
    if not (0 <= target_idx < len(prompts)):
        raise ValueError(f"target_idx={target_idx} out of range for {len(prompts)} prompts")

    # Qwen config (merge CLI > YAML defaults)
    qwen_cfg = cfg.get("qwen", {})
    qwen_model_name = args.qwen_model or qwen_cfg.get("model_name", DEFAULT_QWEN_MODEL)
    qwen_prompt = args.qwen_prompt or qwen_cfg.get("prompt", DEFAULT_QWEN_PROMPT)
    qwen_coeff = args.qwen_coeff if args.qwen_coeff is not None else float(qwen_cfg.get("coeff", 10.0))
    qwen_num_frames = args.qwen_num_frames if args.qwen_num_frames is not None else int(qwen_cfg.get("num_frames", 8))
    frame_sampling_mode = args.frame_sampling_mode or cfg.vlm.get("frame_sampling_mode", "stride3")
    num_random_starts = int(cfg.vlm.get("num_random_starts", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.optim.seed)
    set_lowvram_mode(cfg.model.low_vram)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Prompts (%d): target=%d → %r", len(prompts), target_idx, prompts[target_idx])

    save_dir = _resolve(cfg.output.save_dir, SCRIPT_DIR)
    run_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    logging.info("Run dir: %s", run_dir)

    config_path = _resolve(cfg.model.config, SCRIPT_DIR)
    ckpt_path = _resolve(cfg.model.base_ckpt, SCRIPT_DIR)
    logging.info("Loading Vista: cfg=%s ckpt=%s", config_path, ckpt_path)
    model = init_vista_model(config_path=config_path, base_ckpt_path=ckpt_path, device=device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    latent_shape = get_latent_shape_vista(model, cfg.model.n_frames, cfg.model.height, cfg.model.width)
    logging.info("Latent shape: %s", latent_shape)

    sampler = init_sampling(
        sampler="EulerEDMSampler", guider="VanillaCFG",
        steps=cfg.model.n_steps, cfg_scale=cfg.model.cfg_scale, num_frames=cfg.model.n_frames,
    )

    # ── Load X-CLIP ────────────────────────────────────────────────────────
    from transformers import AutoProcessor, AutoModel
    logging.info("Loading X-CLIP: %s", cfg.vlm.model_id)
    vlm_processor = AutoProcessor.from_pretrained(cfg.vlm.model_id)
    vlm_model = AutoModel.from_pretrained(cfg.vlm.model_id).to(device).eval()
    for p in vlm_model.parameters():
        p.requires_grad = False

    # ── Load Qwen (optional) ───────────────────────────────────────────────
    qwen_model = None
    qwen_processor = None
    if args.use_qwen:
        from transformers import AutoProcessor as _AP, Qwen2_5_VLForConditionalGeneration
        logging.info("Loading Qwen2.5-VL: %s", qwen_model_name)
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_name, torch_dtype=torch.float16, device_map="auto",
        ).eval()
        for p in qwen_model.parameters():
            p.requires_grad_(False)
        qwen_processor = _AP.from_pretrained(qwen_model_name)
        logging.info("Qwen2.5-VL ready. Prompt: %r", qwen_prompt)

    # ── Conditioning ──────────────────────────────────────────────────────
    img = load_img(args.image_path, cfg.model.height, cfg.model.width, device=device)
    unique_keys = set(x.input_key for x in model.conditioner.embedders)
    value_dict = init_embedder_options(unique_keys)
    value_dict["cond_frames_without_noise"] = img.unsqueeze(0)
    value_dict["cond_aug"] = cfg.model.cond_aug
    value_dict["cond_frames"] = img.unsqueeze(0) + cfg.model.cond_aug * torch.randn_like(img.unsqueeze(0))

    if args.trajectory_json is not None:
        traj = _load_trajectory(args.trajectory_json, device)
        value_dict["trajectory"] = traj
        logging.info("Trajectory loaded: shape=%s", traj.shape)
    if args.command is not None:
        value_dict["command"] = torch.tensor([args.command], device=device)

    force_uc_zero = ["cond_frames", "cond_frames_without_noise", "command",
                     "trajectory", "speed", "angle", "goal"]
    cond, uc = get_condition(model, value_dict, cfg.model.n_frames, force_uc_zero, device)
    cond_mask = torch.zeros(cfg.model.n_frames, device=device)
    cond_mask[0] = 1

    load_model(model.first_stage_model)
    cond_frame_latent = model.encode_first_stage(img.unsqueeze(0)).unsqueeze(1)
    unload_model(model.first_stage_model)
    cond_latents_full = torch.zeros(
        (1, cfg.model.n_frames, *cond_frame_latent.shape[2:]),
        device=device, dtype=cond_frame_latent.dtype,
    )
    cond_latents_full[:, 0] = cond_frame_latent[:, 0]
    cond_frame_no_batch = cond_latents_full.squeeze(0)

    # ── Noise + optimiser ─────────────────────────────────────────────────
    noise = torch.randn(
        [1, *latent_shape], device=device,
        dtype=next(model.parameters()).dtype, requires_grad=True,
    )
    initial_noise = noise.detach().clone()
    optimizer = torch.optim.SGD([noise], lr=cfg.optim.lr, momentum=0.0)

    def denoiser_fn(x, sigma, c, cmask):
        return model.denoiser(model.model, x, sigma, c, cmask)

    if args.no_regularizer:
        logging.info("--no_regularizer: all regularizer coefficients set to zero.")
        from omegaconf import OmegaConf as _OC
        cfg.regularizer = _OC.create({k: 0.0 for k in cfg.regularizer})

    history = []
    reg_cfg = cfg.regularizer
    for it in range(1, cfg.optim.iters + 1):
        logging.info("--- Iter %d/%d ---", it, cfg.optim.iters)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            load_model(model.denoiser)
            load_model(model.model)
            latents_nb = noise.detach().clone().squeeze(0)
            cond_s = {k: (v[0:1] if torch.is_tensor(v) and v.shape[0] == 1 else v) for k, v in cond.items()}
            uc_s = {k: (v[0:1] if torch.is_tensor(v) and v.shape[0] == 1 else v) for k, v in uc.items()}
            samples_z = sampler(
                denoiser_fn, latents_nb, cond=cond_s, uc=uc_s,
                cond_frame=cond_frame_no_batch, cond_mask=cond_mask,
            )
            unload_model(model.model)
            unload_model(model.denoiser)

        samples_z = samples_z.detach().requires_grad_(True)

        load_model(model.first_stage_model)
        pixels = model.decode_first_stage_grad(samples_z)
        pixels_cpu = pixels.detach().cpu()

        # X-CLIP reward
        reward, probs = compute_xclip_reward(
            pixels, vlm_model, vlm_processor, prompts, target_idx,
            frame_sampling_mode, num_random_starts, device,
        )
        logging.info("X-CLIP probs: %s", probs.detach().cpu().tolist())
        logging.info("Target prob (idx %d): %.4f", target_idx, reward.item())

        g = torch.autograd.grad(-reward, samples_z, retain_graph=False, only_inputs=True)[0].detach()
        g = g * cfg.optim.grad_scale
        gn = g.norm()
        if gn > cfg.optim.max_grad_norm:
            g = g * (cfg.optim.max_grad_norm / (gn + 1e-6))
        (noise * g).sum().backward()
        del g

        # Optional Qwen reward
        qwen_info = None
        if args.use_qwen and qwen_model is not None:
            qr, qwen_info = compute_qwen_reward_vista(
                pixels, qwen_model, qwen_processor, qwen_prompt, device,
                num_frames=qwen_num_frames, frame_sampling_mode=frame_sampling_mode,
            )
            logging.info("Qwen p_yes=%.3f p_no=%.3f margin=%.4f",
                         qwen_info["p_yes"], qwen_info["p_no"], qwen_info["margin"])
            # Detach pixels to avoid double-graph issues; use surrogate gradient.
            # Latent-gradient value clamp matches the upstream eval script
            # (vista_robust/dsrl/iterative_noise_optim_eval.py).
            qg = torch.autograd.grad(-qwen_coeff * qr, samples_z,
                                     retain_graph=False, only_inputs=True)[0].detach()
            qg = torch.clamp(qg, -cfg.optim.grad_clamp, cfg.optim.grad_clamp)
            (noise * qg).sum().backward()
            del qg, qr

        unload_model(model.first_stage_model)

        reg_loss, shrink, kl, std, std_perm, _, spec = compute_regularizer_video(
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

        if noise.grad is not None:
            torch.nn.utils.clip_grad_norm_([noise], cfg.optim.max_grad_norm)
            noise.grad.clamp_(-cfg.optim.grad_clamp, cfg.optim.grad_clamp)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        noise_norm = noise.norm().item()
        diff = (noise - initial_noise).norm().item()
        logging.info("Noise norm: %.4f | drift: %.4f | spec=%.4f", noise_norm, diff, spec.item())

        entry = {
            "iter": it,
            "target_prob": reward.item(),
            "probs": probs.detach().cpu().tolist(),
            "noise_norm": noise_norm,
            "noise_drift": diff,
            "spectral_loss": spec.item(),
            "std_perm": std_perm.item(),
            "reg_loss": reg_loss.item(),
        }
        if qwen_info is not None:
            entry["qwen_p_yes"] = qwen_info["p_yes"]
            entry["qwen_p_no"] = qwen_info["p_no"]
            entry["qwen_margin"] = qwen_info["margin"]
        history.append(entry)

        if it % cfg.output.save_freq == 0 or it == 1 or it == cfg.optim.iters:
            video_path = os.path.join(run_dir, f"iter_{it:04d}.mp4")
            save_video(pixels_cpu, video_path, fps=cfg.output.fps)
            logging.info("Saved video → %s", video_path)

        del pixels, pixels_cpu, samples_z
        torch.cuda.empty_cache()

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save(noise.detach().cpu(), os.path.join(run_dir, "optimized_noise.pt"))
    logging.info("Done. Results in %s", run_dir)


if __name__ == "__main__":
    main()
