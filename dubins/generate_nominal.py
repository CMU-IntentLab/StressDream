"""
Generate nominal (no noise optimization) Dubins-car rollouts for comparison
against `run_steering.py`.

Same model, initial frame, and rollout action sequence as `run_steering.py`,
but skips the inner optimisation loop: each clip is a single forward pass
with fresh random noise. Multiple seeds are concatenated horizontally into
one video so the spread is visually obvious.

Mirrors the "Nominal rollouts (5 samples)" cell in `dubins/demo.ipynb`.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import torch
import imageio
from pathlib import Path

from dubins.steer import (
    load_steering_model,
    rollout_no_grad,
    decode_latents_to_images,
)
from dubins.env import DubinsConfig, simulate_trajectory

_DUBINS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_CONFIG = os.path.join(_DUBINS_DIR, "config.yaml")
_EXAMPLE_DIR = os.path.join(_DUBINS_DIR, "example_traj")
DEFAULT_INIT_STATE_NPY = os.path.join(_EXAMPLE_DIR, "init_state.npy")
DEFAULT_ACTIONS_NPY = os.path.join(_EXAMPLE_DIR, "actions.npy")


def build_actions(action_spec, traj_length, dubins_config):
    u_max = dubins_config.turnRate
    if action_spec == "ones":
        return torch.ones(traj_length) * u_max
    elif action_spec == "random":
        return torch.rand(traj_length) * 2 * u_max - u_max
    elif action_spec == "zeros":
        return torch.zeros(traj_length)
    else:
        a = torch.from_numpy(np.load(action_spec)).float()
        if a.ndim > 1:
            a = a.squeeze(-1)
        if traj_length is not None and traj_length < len(a):
            a = a[:traj_length]
        return a


def save_video(frames_list, path, fps=10):
    """Concatenate (T, H, W, 3) clips horizontally and save as mp4."""
    T = min(len(f) for f in frames_list)
    writer = imageio.get_writer(path, fps=fps, codec="libx264")
    for t in range(T):
        frame = np.concatenate([f[t] for f in frames_list], axis=1)
        writer.append_data(frame)
    writer.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate nominal (no-optimization) Dubins-car rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_config", type=str, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--init_state", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "THETA"))
    p.add_argument("--actions", type=str, default=DEFAULT_ACTIONS_NPY)
    p.add_argument("--traj_length", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=5,
                   help="Number of nominal rollouts (different seeds) to concatenate.")
    p.add_argument("--seed", type=int, default=0,
                   help="First seed; subsequent samples use seed, seed+1, ...")
    p.add_argument("--vae_checkpoint", type=str, default=None)
    p.add_argument("--wm_checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="steering_results")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    dubins_cfg = DubinsConfig()
    if "prediction_heads" in model_config:
        ph = model_config["prediction_heads"]
        dubins_cfg.obs_x = ph.get("obs_x", 0.0)
        dubins_cfg.obs_y = ph.get("obs_y", 0.0)
        dubins_cfg.obs_r = ph.get("obs_r", 0.25)

    print("=" * 60)
    print("Dubins Car -- Nominal (no-optimization) Rollouts")
    print("=" * 60)

    ckpts = model_config.get("checkpoints", {})
    vae_checkpoint = args.vae_checkpoint or ckpts.get("vae")
    wm_checkpoint = args.wm_checkpoint or ckpts.get("world_model")
    assert vae_checkpoint and wm_checkpoint, "Set checkpoints in config.yaml or pass CLI flags."

    model = load_steering_model(wm_checkpoint, vae_checkpoint, model_config, device)

    if args.init_state is not None:
        initial_state = torch.tensor(args.init_state, dtype=torch.float32)
    else:
        initial_state = torch.from_numpy(np.load(DEFAULT_INIT_STATE_NPY)).float()

    actions = build_actions(args.actions, args.traj_length, dubins_cfg)
    print(f"Initial state: {initial_state.tolist()}")
    print(f"Actions: {args.actions} x {len(actions)} steps")
    print(f"Samples: {args.num_samples} (seeds {args.seed}..{args.seed + args.num_samples - 1})")

    gt_states, gt_images = simulate_trajectory(initial_state, actions, dubins_cfg)

    num_steps_cond = model_config["denoiser"]["num_steps_conditioning"]
    init_img_t = torch.from_numpy(gt_images[0]).float().to(device).permute(2, 0, 1) / 127.5 - 1.0

    with torch.no_grad():
        z_init = model.encode_images(init_img_t.unsqueeze(0), sample=False)
        z_init = z_init.unsqueeze(0).repeat(1, num_steps_cond, 1, 1, 1)

    historical_actions = torch.zeros(1, num_steps_cond - 1, 1, device=device)
    future_actions = actions.unsqueeze(0).unsqueeze(-1).to(device)
    rollout_actions = torch.cat([historical_actions, future_actions], dim=1)

    latent_shape = z_init.shape[2:]
    T_future = rollout_actions.shape[1] - (num_steps_cond - 1)
    init_frame = np.array(gt_images[0])[np.newaxis]

    nominal_clips = []
    for i in range(args.num_samples):
        seed_i = args.seed + i
        torch.manual_seed(seed_i)
        noise = torch.randn(T_future, *latent_shape, device=device)
        with torch.no_grad():
            latents = rollout_no_grad(model, z_init, rollout_actions, noise, num_steps_cond)
            frames = decode_latents_to_images(model, latents)
        nominal_clips.append(np.concatenate([init_frame, frames], axis=0))
        print(f"  Sample {i + 1}/{args.num_samples} (seed={seed_i}) done.")

    video_path = output_dir / "nominal.mp4"
    save_video(nominal_clips, video_path, fps=args.fps)
    print(f"\nNominal video saved to {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
