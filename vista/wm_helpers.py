"""Vista world-model helpers used by run_steering.py.

Extracted from dsrl/model_utils.py, dsrl/datasets.py, dsrl/vista_utils.py,
and scripts/sample_utils.py — the functions actually called during single-image
X-CLIP / Qwen noise steering.
"""

import logging
import math
import os
import sys
from typing import List, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms

# Ensure vista/ is on path so vwm.* resolves correctly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from vwm.util import instantiate_from_config
from vwm.modules.diffusionmodules.sampling import EulerEDMSampler


# ── Vista model ───────────────────────────────────────────────────────────────

def init_vista_model(
    config_path: str,
    base_ckpt_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    device="cuda",
):
    """Initialize Vista model from config and checkpoint."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    if base_ckpt_path is not None:
        assert base_ckpt_path.endswith("safetensors"), "Base checkpoint must be a safetensors file"
        svd = load_safetensors(base_ckpt_path)
        missing, unexpected = model.load_state_dict(svd, strict=False)
        print(f"Loaded base model from {base_ckpt_path}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    if ckpt_path is not None:
        print(f"Loading model from {ckpt_path}")
        if ckpt_path.endswith("ckpt"):
            pl_svd = torch.load(ckpt_path, map_location="cpu")
            if "global_step" in pl_svd:
                print(f"Global step: {pl_svd['global_step']}")
            svd = pl_svd["state_dict"]
        elif ckpt_path.endswith("safetensors"):
            svd = load_safetensors(ckpt_path)
        elif ckpt_path.endswith("pt"):
            pl_svd = torch.load(ckpt_path, map_location="cpu")
            svd = {}
            for old_key, tensor in pl_svd.items():
                if old_key.startswith("diffusion_core."):
                    svd[old_key.replace("diffusion_core.", "", 1)] = tensor
                elif old_key.startswith("model_ema."):
                    continue
                else:
                    svd[old_key] = tensor
        else:
            raise NotImplementedError("Use .ckpt, .safetensors, or .pt checkpoint")

        missing, unexpected = model.load_state_dict(svd, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


def get_latent_shape_vista(
    model,
    num_frames: int = 25,
    height: int = 576,
    width: int = 1024,
) -> List[int]:
    """Return latent shape [T, C, H, W] for Vista (8× spatial downsampling)."""
    return [num_frames, 4, height // 8, width // 8]


# ── Image loading ─────────────────────────────────────────────────────────────

def load_img(file_path, target_height=576, target_width=1024, device="cuda"):
    """Load image, center-crop to Vista aspect ratio, resize, normalize to [-1, 1]."""
    if not os.path.exists(file_path):
        raise ValueError(f"Image file not found: {file_path}")

    image = Image.open(file_path).convert("RGB")
    ori_w, ori_h = image.size

    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)
        left = (ori_w - tmp_w) // 2
        image = image.crop((left, 0, left + tmp_w, ori_h))
    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)
        top = (ori_h - tmp_h) // 2
        image = image.crop((0, top, ori_w, top + tmp_h))

    image = image.resize((target_width, target_height), resample=Image.LANCZOS).convert("RGB")
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])(image)
    return image.to(device)


# ── Video saving ──────────────────────────────────────────────────────────────

def save_video(frames: torch.Tensor, path: str, fps: int = 12):
    """Save [T, C, H, W] frames in [-1, 1] to an MP4 file."""
    frames_01 = torch.clamp((frames + 1.0) / 2.0, 0.0, 1.0)
    frames_np = (frames_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    imageio.mimwrite(path, frames_np, fps=fps, codec="libx264", quality=8)


# ── Low-VRAM helpers ──────────────────────────────────────────────────────────

_lowvram_mode = False


def set_lowvram_mode(mode: bool):
    global _lowvram_mode
    _lowvram_mode = bool(mode)


def load_model(model):
    model.cuda()


def unload_model(model):
    global _lowvram_mode
    if _lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()


# ── Embedder / conditioning ───────────────────────────────────────────────────

def init_embedder_options(keys):
    value_dict = {}
    for key in keys:
        if key in ["fps_id", "fps"]:
            value_dict["fps"] = 10
            value_dict["fps_id"] = 9
        elif key == "motion_bucket_id":
            value_dict["motion_bucket_id"] = 127
    return value_dict


def _get_batch(keys, value_dict, N, device="cuda"):
    batch, batch_uc = {}, {}
    for key in keys:
        if key not in value_dict:
            continue
        if key in ["fps", "fps_id", "motion_bucket_id", "cond_aug"]:
            batch[key] = repeat(torch.tensor([value_dict[key]]).to(device), "1 -> b", b=math.prod(N))
        elif key in ["command", "trajectory", "speed", "angle", "goal"]:
            batch[key] = repeat(value_dict[key][None].to(device), "1 ... -> b ...", b=N[0])
        elif key in ["cond_frames", "cond_frames_without_noise"]:
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        else:
            raise NotImplementedError(f"Unhandled embedder key: {key}")
    for key in batch:
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_condition(model, value_dict, num_samples, force_uc_zero_embeddings, device):
    load_model(model.conditioner)
    batch, batch_uc = _get_batch(
        list({x.input_key for x in model.conditioner.embedders}),
        value_dict,
        [num_samples],
    )
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch, batch_uc=batch_uc, force_uc_zero_embeddings=force_uc_zero_embeddings,
    )
    unload_model(model.conditioner)

    for k in c:
        if isinstance(c[k], torch.Tensor):
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))
            if c[k].shape[0] < num_samples:
                c[k] = c[k][[0]]
            if uc[k].shape[0] < num_samples:
                uc[k] = uc[k][[0]]
    return c, uc


# ── Sampler setup ─────────────────────────────────────────────────────────────

def init_sampling(
    sampler="EulerEDMSampler",
    guider="VanillaCFG",
    discretization="EDMDiscretization",
    steps=50,
    cfg_scale=2.5,
    num_frames=25,
):
    disc = _get_discretization(discretization)
    guid = _get_guider(guider, cfg_scale, num_frames)
    return _get_sampler(sampler, steps, disc, guid)


def _get_discretization(name):
    if name == "EDMDiscretization":
        return {
            "target": "vwm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {"sigma_min": 0.002, "sigma_max": 700.0, "rho": 7.0},
        }
    if name == "LegacyDDPMDiscretization":
        return {"target": "vwm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"}
    raise NotImplementedError(name)


def _get_guider(name, cfg_scale, num_frames):
    if name == "VanillaCFG":
        return {"target": "vwm.modules.diffusionmodules.guiders.VanillaCFG",
                "params": {"scale": cfg_scale}}
    if name == "IdentityGuider":
        return {"target": "vwm.modules.diffusionmodules.guiders.IdentityGuider"}
    if name == "LinearPredictionGuider":
        return {"target": "vwm.modules.diffusionmodules.guiders.LinearPredictionGuider",
                "params": {"max_scale": cfg_scale, "min_scale": 1.0, "num_frames": num_frames}}
    raise NotImplementedError(name)


def _get_sampler(name, steps, disc, guid):
    if name == "EulerEDMSampler":
        return EulerEDMSampler(
            num_steps=steps, discretization_config=disc, guider_config=guid,
            s_churn=0.0, s_tmin=0.0, s_tmax=999.0, s_noise=1.0, verbose=False,
        )
    raise ValueError(f"Unknown sampler: {name}")
