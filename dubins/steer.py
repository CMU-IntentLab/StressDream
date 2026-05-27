"""
Diffusion Noise Optimization (DNO) for steering world model imaginations.

Optimizes the initial noise vectors for each autoregressive rollout step to push
predicted trajectories toward pessimistic (maximize margin) or optimistic (minimize
margin) outcomes.

Two gradient modes:
  - full:   Backpropagate through the entire denoising chain (exact gradients).
  - approx: Forward pass without gradients, then use latent-space gradient as a
            proxy for the noise gradient. Faster and lower memory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont

from dubins.models import (
    DiffusionWorldModel,
    DiffusionWorldModelConfig,
    VAEConfig,
    DiffusionSamplerConfig,
)
from dubins.models.prediction_heads import PredictionHeads, PredictionHeadConfig
from wm_steer.regularizer import (
    spectral_whiteness_loss,
    compute_single_noise_metrics,
    compute_reg,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_steering_model(wm_checkpoint, vae_checkpoint, config, device="cuda"):
    """Load trained world model + VAE + prediction heads."""
    vae_cfg = VAEConfig(**config["vae"])
    sampler_cfg = DiffusionSamplerConfig(**config["sampler"])

    pred_heads_cfg = None
    if "prediction_heads" in config:
        ph = config["prediction_heads"]
        pred_heads_cfg = PredictionHeadConfig(
            latent_dim=config["vae"]["z_channels"],
            latent_spatial_size=config["vae"]["resolution"] // (2 ** len(config["vae"]["ch_mult"])),
            hidden_dim=ph["hidden_dim"],
            num_conv_layers=ph["num_conv_layers"],
            num_mlp_layers=ph["num_mlp_layers"],
        )

    model_cfg = DiffusionWorldModelConfig(
        vae=vae_cfg,
        denoiser=config["denoiser"],
        sampler=sampler_cfg,
        prediction_heads=pred_heads_cfg,
    )
    model = DiffusionWorldModel(model_cfg).to(device)

    wm_ckpt = torch.load(wm_checkpoint, map_location=device)
    model.load_state_dict(wm_ckpt["model_state_dict"], strict=False)
    print(f"Loaded world model from {wm_checkpoint}")

    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    model.vae.load_state_dict(vae_ckpt["model_state_dict"])
    print(f"Loaded VAE from {vae_checkpoint}")

    if "prediction_heads_state_dict" in vae_ckpt and model.prediction_heads is not None:
        model.prediction_heads.load_state_dict(vae_ckpt["prediction_heads_state_dict"])
        print("Loaded prediction heads from VAE checkpoint")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Gradient-enabled diffusion sampling
# ---------------------------------------------------------------------------

def _denoise_step_grad(denoiser, x, sigma, s_in, z_history, actions_history):
    """Single denoising step with gradient flow (bypasses @torch.no_grad decorators)."""
    cs = denoiser.compute_conditioners(sigma * s_in)
    model_output = denoiser.compute_model_output(x, z_history, actions_history, cs)
    denoised = cs.c_skip * x + cs.c_out * model_output
    return (x - denoised) / sigma


def sample_one_step_with_grad(sampler, z_history, actions_history, noise, use_checkpointing=False):
    """Predict next latent from noise with full gradient flow through the denoising ODE."""
    with torch.enable_grad():
        b = noise.shape[0]
        device = noise.device
        s_in = torch.ones(b, device=device)
        x = noise * sampler.sigmas[0].item()

        def _step(x_in, sigma, next_sigma):
            d = _denoise_step_grad(sampler.denoiser, x_in, sigma, s_in, z_history, actions_history)
            return x_in + d * (next_sigma - sigma)

        for sigma, next_sigma in zip(sampler.sigmas[:-1], sampler.sigmas[1:]):
            if use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(_step, x, sigma, next_sigma, use_reentrant=False)
            else:
                x = _step(x, sigma, next_sigma)
        return x


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def rollout_no_grad(model, z_init, actions, noise_vectors, num_steps_cond):
    """Autoregressive rollout without gradient tracking. Returns latents (T, C, H, W)."""
    z_buffer = z_init.clone()
    T_future = noise_vectors.shape[0]
    latents = []
    with torch.no_grad():
        for t in range(T_future):
            z_history = z_buffer
            actions_hist = actions[:, t:t + num_steps_cond]
            noise_t = noise_vectors[t:t + 1].detach()
            z_next, _ = model.sampler.sample(z_history, actions_hist, noise_t)
            latents.append(z_next.squeeze(0))
            z_buffer = torch.cat([z_buffer[:, 1:], z_next.unsqueeze(1)], dim=1)
    return torch.stack(latents, dim=0)


def rollout_with_grad(model, z_init, actions, noise_vectors, num_steps_cond, use_checkpointing=False):
    """Autoregressive rollout with gradient flow. Returns latents (T, C, H, W)."""
    z_buffer = z_init.detach()
    T_future = noise_vectors.shape[0]
    latents = []
    for t in range(T_future):
        z_history = z_buffer
        actions_hist = actions[:, t:t + num_steps_cond]
        noise_t = noise_vectors[t:t + 1]
        z_next = sample_one_step_with_grad(model.sampler, z_history, actions_hist, noise_t, use_checkpointing)
        latents.append(z_next.squeeze(0))
        z_buffer = torch.cat([z_buffer[:, 1:], z_next.unsqueeze(1)], dim=1)
    return torch.stack(latents, dim=0)


# ---------------------------------------------------------------------------
# Image / video helpers
# ---------------------------------------------------------------------------

def decode_latents_to_images(model, latents):
    """(T, C, H, W) latents -> (T, H, W, 3) uint8 numpy."""
    with torch.no_grad():
        imgs = model.decode_latents(latents.unsqueeze(0))[0]
        imgs = torch.clamp(imgs, -1, 1)
        imgs = ((imgs + 1.0) * 0.5 * 255.0).permute(0, 2, 3, 1)
        return imgs.cpu().numpy().astype(np.uint8)


def _get_fonts():
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    try:
        font_small = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font_small = ImageFont.load_default()
    return font, font_small


def create_steering_video(baseline_images, opt_images_list, baseline_margins, opt_margins_list,
                          labels, output_path, fps=10):
    """Create [Baseline | Optimised_1 | ...] side-by-side video."""
    T, H, W, C = baseline_images.shape
    font, font_small = _get_fonts()
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
    for t in range(T):
        frames = [baseline_images[t]] + [r[min(t, len(r) - 1)] for r in opt_images_list]
        pil_img = Image.fromarray(np.concatenate(frames, axis=1))
        draw = ImageDraw.Draw(pil_img)
        draw.text((5, 5), labels[0], fill="cyan", font=font)
        if baseline_margins is not None and t < len(baseline_margins):
            m = baseline_margins[t]
            draw.text((5, 25), f"Margin: {m:.4f}", fill="green" if m > 0 else "red", font=font_small)
        for i in range(len(opt_images_list)):
            x_pos = (i + 1) * W + 5
            lbl = labels[i + 1] if (i + 1) < len(labels) else f"Opt {i+1}"
            draw.text((x_pos, 5), lbl, fill="yellow", font=font)
            if i < len(opt_margins_list) and opt_margins_list[i] is not None and t < len(opt_margins_list[i]):
                m = opt_margins_list[i][t]
                draw.text((x_pos, 25), f"Margin: {m:.4f}", fill="green" if m > 0 else "red", font=font_small)
        draw.text((5, H - 25), f"Frame {t + 1}/{T}", fill="white", font=font)
        writer.append_data(np.array(pil_img))
    writer.close()
    print(f"Saved video to {output_path}")


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def optimize_noise(model, z_init, actions, config, device, args):
    """
    Optimize initial noise independently for each autoregressive timestep.

    For t = 0 ... T-1:
      1. Use the optimized z_buffer as conditioning.
      2. Run args.iters gradient steps on noise[t] to steer margin(z_t).
      3. Fix z_t with best noise found, advance z_buffer.

    Returns:
        optimised_latents       (T, C, H, W)
        margin_history          list-of-lists: margin_history[t] = per-iter margins
        optimised_noise         (T, C, H, W)
        noise_metrics_history   list-of-dicts: per-timestep noise metrics
    """
    num_steps_cond = config["denoiser"]["num_steps_conditioning"]
    T_future = actions.shape[1] - (num_steps_cond - 1)
    latent_shape = z_init.shape[2:]
    use_full_reg = getattr(args, 'use_full_regularizer', True)

    torch.manual_seed(args.seed)
    all_init_noise = torch.randn(T_future, *latent_shape, device=device)

    optimised_latents = []
    optimised_noise = []
    margin_history = []
    noise_metrics_history = []
    z_buffer = z_init.clone()

    print(f"\nOptimising noise: mode={args.mode}, grad_mode={args.grad_mode}, "
          f"iters/step={args.iters}, T_future={T_future}")

    for t in range(T_future):
        z_history = z_buffer.detach()
        actions_hist = actions[:, t:t + num_steps_cond]

        noise_t = all_init_noise[t:t + 1].clone().detach().requires_grad_(True)
        initial_noise_t = noise_t.detach().clone()
        optimizer = torch.optim.SGD([noise_t], lr=args.lr, momentum=0.9)

        best_metric = float("inf")
        best_z = None
        best_n = None
        step_margins = []

        with torch.no_grad():
            init_noise_5d = initial_noise_t.unsqueeze(0)
            spec_threshold_t = spectral_whiteness_loss(init_noise_5d).item()

        for it in range(1, args.iters + 1):
            if args.normalize_noise:
                with torch.no_grad():
                    nv_norm = noise_t.norm().item()
                    target_n = math.sqrt(noise_t.numel())
                    if abs(nv_norm - target_n) > args.noise_norm_threshold:
                        std = noise_t.std() + 1e-10
                        mean = noise_t.mean()
                        noise_t.data.copy_((noise_t.data - mean) / std)

            optimizer.zero_grad(set_to_none=True)

            if args.grad_mode == "full":
                z_next = sample_one_step_with_grad(
                    model.sampler, z_history, actions_hist, noise_t,
                    use_checkpointing=args.gradient_checkpointing,
                )
                pred = model.prediction_heads(z_next)
                margin = pred["margin"].squeeze()
                margin_loss = -margin if args.mode == "pessimistic" else margin
                margin_range_penalty = (
                    torch.relu(margin - args.margin_max) ** 2
                    + torch.relu(args.margin_min - margin) ** 2
                )
                reg_loss = compute_reg(noise_t, args, use_full_reg, spectral_threshold=spec_threshold_t)
                total = args.margin_coeff * margin_loss + args.margin_range_coeff * margin_range_penalty + reg_loss
                total.backward()

            elif args.grad_mode == "approx":
                with torch.no_grad():
                    z_next_det, _ = model.sampler.sample(z_history, actions_hist, noise_t.detach())
                z_grad = z_next_det.detach().requires_grad_(True)
                pred = model.prediction_heads(z_grad)
                margin = pred["margin"].squeeze()
                margin_loss = -margin if args.mode == "pessimistic" else margin
                margin_range_penalty = (
                    torch.relu(margin - args.margin_max) ** 2
                    + torch.relu(args.margin_min - margin) ** 2
                )
                approx_loss = margin_loss + args.margin_range_coeff * margin_range_penalty
                grad_z = torch.autograd.grad(approx_loss, z_grad)[0]
                grad_z = grad_z * args.grad_scale
                g_norm = grad_z.norm()
                if g_norm > 20.0:
                    grad_z = grad_z * (20.0 / (g_norm + 1e-6))
                grad_z = torch.clamp(grad_z, -0.5, 0.5)
                (noise_t * grad_z.detach()).sum().backward()
                reg_loss = compute_reg(noise_t, args, use_full_reg, spectral_threshold=spec_threshold_t)
                reg_loss.backward()
            else:
                raise ValueError(f"Unknown grad_mode: {args.grad_mode}")

            if noise_t.grad is not None:
                torch.nn.utils.clip_grad_norm_([noise_t], 5.0)
                noise_t.grad.clamp_(-0.5, 0.5)

            optimizer.step()

            m_val = margin.item()
            step_margins.append(m_val)
            metric = -m_val if args.mode == "pessimistic" else m_val
            if metric < best_metric:
                best_metric = metric
                with torch.no_grad():
                    best_z, _ = model.sampler.sample(z_history, actions_hist, noise_t.detach())
                    best_n = noise_t.detach().clone()

            if getattr(args, 'debug', False) and (it == 1 or it % args.log_every == 0 or it == args.iters):
                print(f"  t={t:3d} iter={it:4d} | margin={m_val:.4f} | reg={reg_loss.item():.4f}")

            optimizer.zero_grad(set_to_none=True)
            noise_t.grad = None

        optimised_latents.append(best_z.squeeze(0))
        optimised_noise.append(best_n.squeeze(0))
        margin_history.append(step_margins)
        noise_metrics_history.append(compute_single_noise_metrics(best_n))

        z_buffer = torch.cat([z_buffer[:, 1:], best_z.unsqueeze(1)], dim=1)
        torch.cuda.empty_cache()

    return (
        torch.stack(optimised_latents, dim=0),
        margin_history,
        torch.stack(optimised_noise, dim=0),
        noise_metrics_history,
    )
