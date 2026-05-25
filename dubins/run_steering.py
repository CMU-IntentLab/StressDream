"""
Steer the diffusion world model's imagination for the Dubins car.

Takes an initial state and action sequence, runs the world model, and optimizes
the initial noise to steer the imagined trajectory toward/away from failure.

Hyperparameters are loaded from a YAML config file (default: dubins/steering_config.yaml).
CLI arguments override individual values from the config.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

from dubins.steer import (
    load_steering_model,
    optimize_noise,
    rollout_no_grad,
    decode_latents_to_images,
    create_steering_video,
)
from dubins.env import DubinsConfig, simulate_trajectory, compute_gt_margins

_DUBINS_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_DIR = os.path.dirname(_DUBINS_DIR)
_PROJECT_ROOT = os.path.dirname(_RELEASE_DIR)

DEFAULT_MODEL_CONFIG = os.path.join(_DUBINS_DIR, "config.yaml")
DEFAULT_STEERING_CONFIG = os.path.join(_DUBINS_DIR, "steering_config.yaml")

# Default example trajectory (traj_000510 from eval dataset, gt_failure=True)
_EXAMPLE_DIR = os.path.join(_DUBINS_DIR, "example_traj")
DEFAULT_INIT_STATE_NPY = os.path.join(_EXAMPLE_DIR, "init_state.npy")
DEFAULT_ACTIONS_NPY = os.path.join(_EXAMPLE_DIR, "actions.npy")


def load_steering_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_actions(action_spec, traj_length, dubins_config):
    u_max = dubins_config.turnRate
    if action_spec == 'ones':
        return torch.ones(traj_length) * u_max
    elif action_spec == 'random':
        return torch.rand(traj_length) * 2 * u_max - u_max
    elif action_spec == 'zeros':
        return torch.zeros(traj_length)
    else:
        a = torch.from_numpy(np.load(action_spec)).float()
        if a.ndim > 1:
            a = a.squeeze(-1)  # (T, 1) -> (T,)
        if traj_length is not None and traj_length < len(a):
            a = a[:traj_length]
        return a


def parse_args(steering_defaults):
    p = argparse.ArgumentParser(
        description="Steer diffusion WM for Dubins car",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config files
    p.add_argument('--model_config', type=str, default=DEFAULT_MODEL_CONFIG,
                   help='Path to model architecture config (config.yaml)')
    p.add_argument('--steering_config', type=str, default=DEFAULT_STEERING_CONFIG,
                   help='Path to steering hyperparameter config (steering_config.yaml)')

    # Run-specific: what to roll out
    p.add_argument('--init_state', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'THETA'),
                   help='Initial state [x y theta]. Defaults to example_traj/init_state.npy')
    p.add_argument('--actions', type=str, default=DEFAULT_ACTIONS_NPY,
                   help='"ones", "random", "zeros", or path to .npy file')
    p.add_argument('--traj_length', type=int, default=None,
                   help='Number of steps. Inferred from actions file if not given')
    p.add_argument('--seed', type=int, default=42)

    # Checkpoints (default loaded from config.yaml checkpoints section)
    p.add_argument('--vae_checkpoint', type=str, default=None,
                   help='Path to VAE checkpoint. Overrides config.yaml checkpoints.vae')
    p.add_argument('--wm_checkpoint', type=str, default=None,
                   help='Path to WM checkpoint. Overrides config.yaml checkpoints.world_model')

    # Steering mode (commonly overridden, so also available as CLI)
    p.add_argument('--mode', type=str, default=steering_defaults.get('mode', 'pessimistic'),
                   choices=['pessimistic', 'optimistic'])
    p.add_argument('--grad_mode', type=str, default=steering_defaults.get('grad_mode', 'full'),
                   choices=['full', 'approx'])

    # Hyperparameter overrides (loaded from steering_config.yaml by default)
    p.add_argument('--iters', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--margin_coeff', type=float, default=None)
    p.add_argument('--grad_scale', type=float, default=None)
    p.add_argument('--gradient_checkpointing', action='store_true', default=None)
    p.add_argument('--kl_coeff', type=float, default=None)
    p.add_argument('--kl_coeff_spherical', type=float, default=None)
    p.add_argument('--std_coeff', type=float, default=None)
    p.add_argument('--spectral_coeff', type=float, default=None)
    p.add_argument('--std_permutation_coeff', type=float, default=None)
    p.add_argument('--num_perms', type=int, default=None)
    p.add_argument('--use_full_regularizer', action='store_true', default=None)
    p.add_argument('--use_simple_regularizer', dest='use_full_regularizer', action='store_false')
    p.add_argument('--margin_range_coeff', type=float, default=None)
    p.add_argument('--margin_min', type=float, default=None)
    p.add_argument('--margin_max', type=float, default=None)
    p.add_argument('--normalize_noise', action='store_true', default=None)
    p.add_argument('--no_normalize_noise', dest='normalize_noise', action='store_false')
    p.add_argument('--noise_norm_threshold', type=float, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--fps', type=int, default=None)
    p.add_argument('--log_every', type=int, default=None)
    p.add_argument('--debug', action='store_true', default=None)

    return p.parse_args()


def merge_config(steering_cfg, cli_args):
    """Return a namespace with steering_cfg as base, overridden by non-None CLI args."""
    merged = dict(steering_cfg)
    for key, val in vars(cli_args).items():
        if val is not None:
            merged[key] = val

    class _Namespace:
        pass

    ns = _Namespace()
    for key, val in merged.items():
        setattr(ns, key, val)
    return ns


def main():
    # Two-pass: first load the steering config, then parse CLI with those as defaults
    steering_cfg = load_steering_config(DEFAULT_STEERING_CONFIG)

    # Check if --steering_config was passed before full parse
    import sys as _sys
    for i, arg in enumerate(_sys.argv[1:]):
        if arg == '--steering_config' and i + 1 < len(_sys.argv) - 1:
            steering_cfg = load_steering_config(_sys.argv[i + 2])
            break

    cli_args = parse_args(steering_cfg)

    # Re-load in case --steering_config was passed
    if cli_args.steering_config != DEFAULT_STEERING_CONFIG:
        steering_cfg = load_steering_config(cli_args.steering_config)

    args = merge_config(steering_cfg, cli_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(cli_args.model_config) as f:
        model_config = yaml.safe_load(f)

    dubins_cfg = DubinsConfig()
    if "prediction_heads" in model_config:
        ph = model_config["prediction_heads"]
        dubins_cfg.obs_x = ph.get("obs_x", 0.0)
        dubins_cfg.obs_y = ph.get("obs_y", 0.0)
        dubins_cfg.obs_r = ph.get("obs_r", 0.25)

    print("=" * 60)
    print("Dubins Car -- Diffusion WM Steering")
    print("=" * 60)
    print(f"Mode: {args.mode}  |  Grad: {args.grad_mode}  |  Iters: {args.iters}")

    # Resolve checkpoint paths: CLI > config.yaml checkpoints section
    ckpts = model_config.get("checkpoints", {})
    vae_checkpoint = cli_args.vae_checkpoint or ckpts.get("vae")
    wm_checkpoint = cli_args.wm_checkpoint or ckpts.get("world_model")
    assert vae_checkpoint, "VAE checkpoint not set (add checkpoints.vae to config.yaml or pass --vae_checkpoint)"
    assert wm_checkpoint, "WM checkpoint not set (add checkpoints.world_model to config.yaml or pass --wm_checkpoint)"

    model = load_steering_model(wm_checkpoint, vae_checkpoint, model_config, device)
    assert model.prediction_heads is not None, "Model must have prediction heads for steering"

    # Resolve init_state: CLI > example_traj/init_state.npy
    if cli_args.init_state is not None:
        initial_state = torch.tensor(cli_args.init_state, dtype=torch.float32)
    else:
        initial_state = torch.from_numpy(np.load(DEFAULT_INIT_STATE_NPY)).float()

    actions = build_actions(cli_args.actions, cli_args.traj_length, dubins_cfg)
    # If traj_length was not specified, use full action sequence
    if cli_args.traj_length is None:
        args.traj_length = len(actions)
    print(f"Initial state: {initial_state.tolist()}")
    print(f"Actions: {cli_args.actions} x {len(actions)} steps")

    gt_states, gt_images = simulate_trajectory(initial_state, actions, dubins_cfg)
    gt_margins = compute_gt_margins(gt_states, dubins_cfg)

    num_steps_cond = model_config["denoiser"]["num_steps_conditioning"]
    init_img = gt_images[0]
    init_img_t = torch.from_numpy(init_img).float().to(device).permute(2, 0, 1) / 127.5 - 1.0

    with torch.no_grad():
        z_init = model.encode_images(init_img_t.unsqueeze(0), sample=False)
        z_init = z_init.unsqueeze(0).repeat(1, num_steps_cond, 1, 1, 1)

    historical_actions = torch.zeros(1, num_steps_cond - 1, 1, device=device)
    future_actions = actions.unsqueeze(0).unsqueeze(-1).to(device)
    rollout_actions = torch.cat([historical_actions, future_actions], dim=1)

    latent_shape = z_init.shape[2:]
    T_future = rollout_actions.shape[1] - (num_steps_cond - 1)

    print("\nGenerating unoptimized baseline...")
    torch.manual_seed(cli_args.seed)
    baseline_noise = torch.randn(T_future, *latent_shape, device=device)

    with torch.no_grad():
        baseline_latents = rollout_no_grad(model, z_init, rollout_actions, baseline_noise, num_steps_cond)
        baseline_images = decode_latents_to_images(model, baseline_latents)
        baseline_margins = model.prediction_heads(baseline_latents)["margin"].squeeze(-1).cpu().numpy()

    print(f"  Baseline: avg={baseline_margins.mean():.4f}  "
          f"min={baseline_margins.min():.4f}  max={baseline_margins.max():.4f}")

    print(f"\nRunning {args.mode} optimization...")
    opt_latents, margin_hist, opt_noise, _ = optimize_noise(
        model, z_init, rollout_actions, model_config, device, args
    )

    opt_images = decode_latents_to_images(model, opt_latents.to(device))
    with torch.no_grad():
        opt_margins = model.prediction_heads(opt_latents.to(device))["margin"].squeeze(-1).cpu().numpy()

    improvement = opt_margins.mean() - baseline_margins.mean()
    better = "higher" if args.mode == "pessimistic" else "lower"
    print(f"  Optimized: avg={opt_margins.mean():.4f}  "
          f"min={opt_margins.min():.4f}  max={opt_margins.max():.4f}  "
          f"(delta={improvement:+.4f}, {better} is better)")

    init_frame = np.array(gt_images[0])[np.newaxis]
    base_vid = np.concatenate([init_frame, baseline_images], axis=0)
    opt_vid = np.concatenate([init_frame, opt_images], axis=0)

    T_min = min(len(base_vid), len(opt_vid))
    base_m = np.concatenate([[gt_margins[0]], baseline_margins])[:T_min]
    opt_m = np.concatenate([[gt_margins[0]], opt_margins])[:T_min]

    video_path = output_dir / f"steering_{args.mode}.mp4"
    create_steering_video(
        base_vid[:T_min],
        [opt_vid[:T_min]],
        base_m,
        [opt_m],
        ["Baseline", args.mode.capitalize()],
        output_path=video_path,
        fps=args.fps,
    )

    print(f"\nVideo saved to {video_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
