from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_dims(input: torch.Tensor, n: int) -> torch.Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: torch.Tensor
    c_out: torch.Tensor
    c_skip: torch.Tensor
    c_noise: torch.Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float = -1.2
    scale: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 80.0


class FourierFeatures(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.register_buffer('weight', torch.randn(dim // 2) * scale)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        x = sigma.unsqueeze(-1) * self.weight.unsqueeze(0) * 2 * torch.pi
        return torch.cat([x.cos(), x.sin()], dim=-1)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_channels * 2))
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class LatentUNet(nn.Module):
    def __init__(self, latent_dim, action_dim, num_steps_conditioning=2,
                 cond_dim=256, hidden_dims=(128, 256, 512)):
        super().__init__()
        self.num_steps_conditioning = num_steps_conditioning
        self.noise_emb = FourierFeatures(cond_dim)
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim * num_steps_conditioning, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        input_channels = latent_dim * (num_steps_conditioning + 1)
        self.conv_in = nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = hidden_dims[0]
        for out_ch in hidden_dims:
            self.down_blocks.append(ResNetBlock(in_ch, out_ch, cond_dim))
            in_ch = out_ch

        self.mid_block = ResNetBlock(hidden_dims[-1], hidden_dims[-1], cond_dim)

        self.up_blocks = nn.ModuleList()
        in_ch = hidden_dims[-1]
        for i, out_ch in enumerate(reversed(hidden_dims[:-1])):
            skip_ch = hidden_dims[-(i + 2)]
            self.up_blocks.append(ResNetBlock(in_ch + skip_ch, out_ch, cond_dim))
            in_ch = out_ch

        self.norm_out = nn.GroupNorm(32, hidden_dims[0])
        self.conv_out = nn.Conv2d(hidden_dims[0], latent_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, noisy_next_z, c_noise, z_history, actions_history):
        B, T, C, H, W = z_history.shape
        z_flat = z_history.reshape(B, T * C, H, W)
        actions_flat = actions_history.reshape(B, -1)

        cond = self.cond_proj(self.noise_emb(c_noise) + self.action_emb(actions_flat))

        x = self.conv_in(torch.cat([z_flat, noisy_next_z], dim=1))

        skip_connections = []
        for block in self.down_blocks:
            x = block(x, cond)
            skip_connections.append(x)

        x = self.mid_block(x, cond)

        for block, skip in zip(self.up_blocks, reversed(skip_connections[:-1])):
            x = block(torch.cat([x, skip], dim=1), cond)

        return self.conv_out(F.silu(self.norm_out(x)))


@dataclass
class LatentDenoiserConfig:
    latent_dim: int = 4
    action_dim: int = 1
    num_steps_conditioning: int = 2
    cond_dim: int = 256
    hidden_dims: Tuple[int, ...] = (128, 256, 512)
    sigma_data: float = 0.5
    sigma_offset_noise: float = 0.0


class LatentDenoiser(nn.Module):
    def __init__(self, config: LatentDenoiserConfig):
        super().__init__()
        self.config = config
        self.unet = LatentUNet(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            num_steps_conditioning=config.num_steps_conditioning,
            cond_dim=config.cond_dim,
            hidden_dims=config.hidden_dims,
        )
        self.sample_sigma_training = None

    @property
    def device(self):
        return next(self.parameters()).device

    def setup_training(self, sigma_cfg: SigmaDistributionConfig):
        assert self.sample_sigma_training is None

        def sample_sigma(n, device):
            s = torch.randn(n, device=device) * sigma_cfg.scale + sigma_cfg.loc
            return s.exp().clip(sigma_cfg.sigma_min, sigma_cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def apply_noise(self, z, sigma, sigma_offset_noise):
        b, c, h, w = z.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return z + offset_noise + torch.randn_like(z) * add_dims(sigma, z.ndim)

    def compute_conditioners(self, sigma):
        sigma = (sigma ** 2 + self.config.sigma_offset_noise ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.config.sigma_data ** 2).sqrt()
        c_skip = self.config.sigma_data ** 2 / (sigma ** 2 + self.config.sigma_data ** 2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(
            c_in=add_dims(c_in, 4), c_out=add_dims(c_out, 4),
            c_skip=add_dims(c_skip, 4), c_noise=c_noise,
        )

    def compute_model_output(self, noisy_next_z, z_history, actions_history, cs):
        rescaled_z_history = z_history / self.config.sigma_data
        return self.unet(noisy_next_z * cs.c_in, cs.c_noise, rescaled_z_history, actions_history)

    @torch.no_grad()
    def denoise(self, noisy_next_z, sigma, z_history, actions_history):
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_z, z_history, actions_history, cs)
        return cs.c_skip * noisy_next_z + cs.c_out * model_output

    def forward(self, z_history, next_z, actions_history):
        assert self.sample_sigma_training is not None, "Call setup_training first"
        b, t, n, c, h, w = z_history.shape
        assert n == self.config.num_steps_conditioning

        z_history_flat = z_history.reshape(b * t, n, c, h, w)
        next_z_flat = next_z.reshape(b * t, c, h, w)
        actions_history_flat = actions_history.reshape(b * t, n, -1)

        sigma = self.sample_sigma_training(b * t, self.device)
        noisy_next_z = self.apply_noise(next_z_flat, sigma, self.config.sigma_offset_noise)
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_z, z_history_flat, actions_history_flat, cs)
        target = (next_z_flat - cs.c_skip * noisy_next_z) / cs.c_out
        loss = F.mse_loss(model_output, target)
        return loss, {"loss_denoising": loss.item()}
