"""Ctrl-World world-model helpers used by run_steering.py.

Forward / decode helpers are extracted from
ctrl-world/scripts/iterative_noise_optim_vlm_ctrlworld.py.
The HDF5 trajectory loader is extracted from
ctrl-world/scripts/oolong/iterative_noise_optim_oolong.py
so the HDF5 path runs full autoregressive rollouts from the real
GT end-effector history.
"""

import io

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline


# ── HDF5 → eef_states + per-view video latents ────────────────────────────
# Constants mirror ctrl-world/scripts/oolong/iterative_noise_optim_oolong.py.

TARGET_HEIGHT = 192
TARGET_WIDTH = 320
FRANKA_GRIPPER_MIN_WIDTH = 0.01310573
FRANKA_GRIPPER_MAX_WIDTH = 0.08013216


def _decode_jpeg(jpeg_uint8_bytes):
    jpeg_buf = np.asarray(jpeg_uint8_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode JPEG bytes")
    return img


def _crop_center_half(bgr):
    """Center-crop to a 2/3 × 2/3 inner region (matches upstream extract_latent_ours.py)."""
    h, w = bgr.shape[:2]
    ch, cw = 2 * h // 3, 2 * w // 3
    y0 = (h - ch) // 3
    x0 = (w - cw) // 3
    return bgr[y0:y0 + ch, x0:x0 + cw]


def _ee_state_to_7dim(ee_flat, gripper):
    ee = ee_flat.reshape(4, 4).T  # Franka column-major
    pos = ee[:3, 3]
    euler = R.from_matrix(ee[:3, :3]).as_euler("xyz")
    gripper_scaled = (gripper - FRANKA_GRIPPER_MIN_WIDTH) / (
        FRANKA_GRIPPER_MAX_WIDTH - FRANKA_GRIPPER_MIN_WIDTH
    )
    gripper_norm = 1.0 - np.clip(gripper_scaled, 0.0, 1.0)
    return np.concatenate([pos, euler, [gripper_norm]])


def load_hdf5_trajectory(
    hdf5_path,
    vae,
    device,
    dtype,
    camera_order=(0, 1, 2),
    rgb_skip=3,
    center_crop_half=True,
):
    """Load one HDF5 trajectory: decode frames, reorder cameras, VAE-encode each view.

    Returns:
        eef_states:    np.ndarray (N, 7)   — 7-dim end-effector states per skipped frame
        video_latents: list of 3 tensors, each (N, 4, H_lat, W_lat) on `device`

    `camera_order[i]` is the upstream camera_id mapped into ctrl-world slot i
    (0=left, 1=right, 2=wrist). Slot 2 (wrist) skips `_crop_center_half`.
    """
    import h5py

    assert len(camera_order) == 3, f"camera_order must list 3 cam ids, got {camera_order}"

    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        total_steps = data["actions"].shape[0]
        frame_ids = np.arange(0, total_steps, rgb_skip)

        ee_raw = np.array(data["ee_states"])[frame_ids]
        gripper_raw = np.array(data["gripper_states"])[frame_ids]
        eef_states = np.array(
            [_ee_state_to_7dim(ee_raw[i], gripper_raw[i]) for i in range(len(frame_ids))],
            dtype=np.float32,
        )

        video_latents = []
        for view_id, cam_id in enumerate(camera_order):
            cam_key = f"camera_{cam_id}"
            frames = []
            for fid in frame_ids:
                jpeg_bytes = np.array(data[cam_key][fid])
                bgr = _decode_jpeg(jpeg_bytes)
                if center_crop_half and view_id != 2:
                    bgr = _crop_center_half(bgr)
                img = cv2.resize(bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                frames.append(img)
            video_np = np.stack(frames, axis=0)  # (N, H, W, 3) uint8 (BGR)

            x = torch.from_numpy(video_np).to(dtype).to(device)
            x = x.permute(0, 3, 1, 2) / 255.0 * 2 - 1   # (N, 3, H, W) in [-1, 1]
            with torch.no_grad():
                bsz = 32
                lats = []
                for i in range(0, len(x), bsz):
                    batch = x[i:i + bsz]
                    lat = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    lats.append(lat)
                latents = torch.cat(lats, dim=0).to(device=device, dtype=dtype)
            video_latents.append(latents)

    return eef_states, video_latents


def normalize_bound(data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
    """Linear normalisation to [clip_min, clip_max] using percentile bounds."""
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max)


def build_cond_latent_3views(views_list, target_height_total=72, target_w=40, device=None, dtype=None):
    """Stack 3 view latents along height to produce a (1, 4, 72, 40) conditioning latent.

    Each view is resized to (4, 24, 40) if needed, then concatenated on the height axis.
    """
    h_per_view = target_height_total // 3
    out_list = []
    for v in views_list:
        if v.dim() == 3:
            v = v.unsqueeze(0)
        _, c, h, w = v.shape
        if h != h_per_view or w != target_w:
            v = F.interpolate(v.float(), size=(h_per_view, target_w), mode="bilinear", align_corners=False)
        out_list.append(v.squeeze(0))
    out = torch.cat(out_list, dim=1).unsqueeze(0)
    if dtype is not None:
        out = out.to(dtype)
    if device is not None:
        out = out.to(device)
    return out


def forward_wm_with_noise(
    model, pipeline, action_cond, image_cond, his_cond, text_token, args,
    custom_noise=None, device=None, dtype=None,
):
    """Run the Ctrl-World diffusion pipeline with explicit initial noise.

    Returns latents reshaped to (num_views, num_frames, C, H, W) where num_views=3.
    """
    _, latents = CtrlWorldDiffusionPipeline.__call__(
        pipeline,
        image=image_cond,
        text=text_token,
        width=args.width,
        height=int(args.height * 3),
        num_frames=args.num_frames,
        history=his_cond,
        num_inference_steps=args.num_inference_steps,
        decode_chunk_size=args.decode_chunk_size,
        max_guidance_scale=args.guidance_scale,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        mask=None,
        output_type="latent",
        return_dict=False,
        frame_level_cond=True,
        latents=custom_noise,
    )
    latents = einops.rearrange(latents, "b f c (m h) w -> (b m) f c h w", m=3, h=24)
    return latents


def decode_latents_with_grad(vae, latents, decode_chunk_size=8):
    """Decode (B, T, C, H, W) latents to pixels in [-1, 1] with gradient support."""
    bsz, frame_num = latents.shape[:2]
    x = latents.flatten(0, 1)

    decoded_list = []
    for i in range(0, x.shape[0], decode_chunk_size):
        chunk = x[i:i + decode_chunk_size] / vae.config.scaling_factor
        decoded = vae.decode(chunk, num_frames=chunk.shape[0]).sample
        decoded_list.append(decoded)

    videos = torch.cat(decoded_list, dim=0)
    return videos.reshape(bsz, frame_num, *videos.shape[1:])


