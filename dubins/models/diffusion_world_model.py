from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import ImageVAE, VAEConfig
from .latent_denoiser import LatentDenoiser, LatentDenoiserConfig, SigmaDistributionConfig
from .prediction_heads import PredictionHeads, PredictionHeadConfig


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int = 20
    sigma_min: float = 2e-3
    sigma_max: float = 80.0
    rho: int = 7
    order: int = 2
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0


class DiffusionSampler:
    def __init__(self, denoiser, config: DiffusionSamplerConfig):
        self.denoiser = denoiser
        self.config = config
        self.sigmas = self._build_sigmas(
            config.num_steps_denoising, config.sigma_min, config.sigma_max, config.rho, denoiser.device
        )

    @staticmethod
    def _build_sigmas(num_steps, sigma_min, sigma_max, rho, device):
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        lambda_vals = torch.linspace(0, 1, num_steps, device=device)
        sigmas = (max_inv_rho + lambda_vals * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat((sigmas, sigmas.new_zeros(1)))

    @torch.no_grad()
    def sample(self, z_history, actions_history, initial_noise=None):
        device = z_history.device
        b, n, c, h, w = z_history.shape

        if initial_noise is not None:
            x = initial_noise.to(device) * self.sigmas[0].item()
        else:
            x = torch.randn(b, c, h, w, device=device) * self.sigmas[0].item()
        s_in = torch.ones(b, device=device)

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            denoised = self.denoiser.denoise(x, sigma * s_in, z_history, actions_history)
            d = (x - denoised) / sigma
            dt = next_sigma - sigma
            if self.config.order == 1 or float(next_sigma.item()) == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, z_history, actions_history)
                d_2 = (x_2 - denoised_2) / next_sigma
                x = x + (d + d_2) / 2 * dt

        return x, []

    def sample_with_grad(self, z_history, actions_history, initial_noise, use_gradient_checkpointing=False):
        with torch.enable_grad():
            device = z_history.device
            b, n, c, h, w = z_history.shape
            if not initial_noise.requires_grad:
                initial_noise = initial_noise.detach().requires_grad_(True)
            x = initial_noise * self.sigmas[0].item()
            s_in = torch.ones(b, device=device)

            def denoise_step(x_in, sigma, next_sigma):
                sigma_hat = sigma
                denoised = self.denoiser.denoise(x_in, sigma_hat * s_in, z_history, actions_history)
                d = (x_in - denoised) / sigma_hat
                return x_in + d * (next_sigma - sigma_hat)

            for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
                if use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(denoise_step, x, sigma, next_sigma, use_reentrant=False)
                else:
                    x = denoise_step(x, sigma, next_sigma)
            return x

    @torch.no_grad()
    def sample_inverse(self, z_history, actions_history, target_z, return_trajectory=False, num_steps_override=None):
        device = target_z.device
        b = target_z.shape[0]
        s_in = torch.ones(b, device=device)
        x = target_z.clone()
        trajectory = [x.clone()] if return_trajectory else None

        if num_steps_override is not None and num_steps_override > len(self.sigmas) - 1:
            step_indices = torch.arange(num_steps_override + 1, device=device)
            t_steps = (self.config.sigma_max ** (1 / self.config.rho) + step_indices / num_steps_override * (
                self.config.sigma_min ** (1 / self.config.rho) - self.config.sigma_max ** (1 / self.config.rho)
            )) ** self.config.rho
            sigmas_fine = torch.cat([t_steps, torch.zeros(1, device=device)])
        else:
            sigmas_fine = self.sigmas

        sigma_pairs = list(zip(sigmas_fine[:-1], sigmas_fine[1:]))
        eps = 1e-6

        for sigma_curr, sigma_next in reversed(sigma_pairs):
            sigma_next_val = float(sigma_next.item())
            sigma_curr_val = float(sigma_curr.item())
            sigma_hat = max(sigma_next_val, eps)
            sigma_hat_tensor = torch.full((b,), sigma_hat, device=device)
            denoised = self.denoiser.denoise(x, sigma_hat_tensor * s_in, z_history, actions_history)
            d = (x - denoised) / sigma_hat
            dt = sigma_curr_val - sigma_next_val
            if self.config.order == 1:
                x = x + d * dt
            else:
                x_predict = x + d * dt
                sigma_curr_tensor = torch.full((b,), sigma_curr_val, device=device)
                denoised_prev = self.denoiser.denoise(x_predict, sigma_curr_tensor * s_in, z_history, actions_history)
                d_prev = (x_predict - denoised_prev) / sigma_curr_val
                x = x + (d + d_prev) / 2 * dt
            if return_trajectory:
                trajectory.append(x.clone())

        return x, trajectory


def create_denoiser(config: dict):
    backbone = config.get('backbone', 'unet')
    if backbone == 'unet':
        return LatentDenoiser(LatentDenoiserConfig(
            latent_dim=config['latent_dim'],
            action_dim=config['action_dim'],
            num_steps_conditioning=config['num_steps_conditioning'],
            cond_dim=config.get('cond_dim', 256),
            hidden_dims=tuple(config.get('hidden_dims', [128, 256, 512])),
            sigma_data=config.get('sigma_data', 0.5),
            sigma_offset_noise=config.get('sigma_offset_noise', 0.0),
        ))
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Only 'unet' is supported.")


@dataclass
class DiffusionWorldModelConfig:
    vae: VAEConfig
    denoiser: dict
    sampler: DiffusionSamplerConfig
    prediction_heads: Optional[PredictionHeadConfig] = None


class DiffusionWorldModel(nn.Module):
    def __init__(self, config: DiffusionWorldModelConfig):
        super().__init__()
        self.config = config
        self.vae = ImageVAE(config.vae)
        self.denoiser = create_denoiser(config.denoiser)
        self.backbone_type = config.denoiser.get('backbone', 'unet')
        self.prediction_heads = PredictionHeads(config.prediction_heads) if config.prediction_heads is not None else None
        self._sampler = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def sampler(self):
        if self._sampler is None:
            self._sampler = DiffusionSampler(self.denoiser, self.config.sampler)
        return self._sampler

    def setup_training(self, sigma_cfg):
        self.denoiser.setup_training(sigma_cfg)

    @torch.no_grad()
    def encode_images(self, images, sample=False):
        original_shape = images.shape
        if len(original_shape) == 5:
            b, t, c, h, w = original_shape
            images = images.reshape(b * t, c, h, w)
        latents = self.vae.encode_to_z(images, sample=sample)
        if len(original_shape) == 5:
            latents = latents.reshape(b, t, *latents.shape[1:])
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        original_shape = latents.shape
        if len(original_shape) == 5:
            b, t, c, h, w = original_shape
            latents = latents.reshape(b * t, c, h, w)
        images = self.vae.decode_from_z(latents)
        if len(original_shape) == 5:
            images = images.reshape(b, t, *images.shape[1:])
        return images

    def train_denoiser(self, images, actions):
        b, t_total, c, h, w = images.shape
        n = self.config.denoiser['num_steps_conditioning']
        assert t_total > n

        with torch.no_grad():
            latents = self.encode_images(images, sample=False)

        valid_length = t_total - n
        z_history_list, z_next_list, actions_history_list = [], [], []
        for i in range(valid_length):
            z_history_list.append(latents[:, i:i + n])
            z_next_list.append(latents[:, i + n])
            actions_history_list.append(actions[:, i:i + n])

        z_history = torch.stack(z_history_list, dim=1)
        z_next = torch.stack(z_next_list, dim=1)
        actions_history = torch.stack(actions_history_list, dim=1)
        return self.denoiser(z_history, z_next, actions_history)

    @torch.no_grad()
    def predict_next_latent(self, z_history, actions_history, initial_noise=None):
        next_z, _ = self.sampler.sample(z_history, actions_history, initial_noise)
        return next_z

    def predict_next_latent_with_grad(self, z_history, actions_history, initial_noise):
        return self.sampler.sample_with_grad(z_history, actions_history, initial_noise)

    @torch.no_grad()
    def rollout(self, initial_images, actions, return_latents=False, initial_noise=None):
        b, n, c, h, w = initial_images.shape
        num_steps_cond = self.config.denoiser['num_steps_conditioning']
        assert n == num_steps_cond
        t_future = actions.shape[1] - (n - 1)

        z_buffer = self.encode_images(initial_images, sample=False)
        image_list = [initial_images[:, i] for i in range(n)]
        latent_list = [z_buffer[:, i] for i in range(n)] if return_latents else None

        for i in range(t_future):
            actions_history = actions[:, i:i + n]
            z_next = self.predict_next_latent(z_buffer, actions_history)
            z_buffer = torch.cat([z_buffer[:, 1:], z_next.unsqueeze(1)], dim=1)
            image_list.append(self.decode_latents(z_next))
            if return_latents:
                latent_list.append(z_next)

        images = torch.stack(image_list, dim=1)
        latents = torch.stack(latent_list, dim=1) if return_latents else None
        return images, latents
