from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distribution import DiagonalGaussianDistribution


class Normalize(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.norm(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = F.softmax(torch.bmm(q, k) * (c ** -0.5), dim=2)
        v = v.reshape(b, c, h * w)
        h_ = torch.bmm(v, w_.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,2,4,4), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, resolution=128, z_channels=4, double_z=True):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            m = nn.Module()
            m.block = block
            m.attn = attn
            if i_level != self.num_resolutions - 1:
                m.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(m)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.norm_out = Normalize(block_in)
        out_ch = z_channels * 2 if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1])))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, out_channels=3, ch=128, ch_mult=(1,2,4,4), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, resolution=128, z_channels=4):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            m = nn.Module()
            m.block = block
            m.attn = attn
            if i_level != 0:
                m.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, m)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


@dataclass
class VAEConfig:
    in_channels: int = 3
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.0
    resolution: int = 128
    z_channels: int = 4
    embed_dim: int = 4
    double_z: bool = True


class ImageVAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            in_channels=config.in_channels, ch=config.ch, ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks, attn_resolutions=config.attn_resolutions,
            dropout=config.dropout, resolution=config.resolution,
            z_channels=config.z_channels, double_z=config.double_z,
        )
        self.decoder = Decoder(
            out_channels=config.in_channels, ch=config.ch, ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks, attn_resolutions=config.attn_resolutions,
            dropout=config.dropout, resolution=config.resolution, z_channels=config.z_channels,
        )
        assert config.double_z
        self.quant_conv = nn.Conv2d(2 * config.z_channels, 2 * config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.z_channels, 1)

    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    @torch.no_grad()
    def encode_to_z(self, x, sample=False):
        posterior = self.encode(x)
        return posterior.sample() if sample else posterior.mode()

    @torch.no_grad()
    def decode_from_z(self, z):
        return self.decode(z)
