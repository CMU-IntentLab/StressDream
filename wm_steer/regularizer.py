"""
Noise regularizers for Diffusion Noise Optimization (DNO).

These functions penalize the optimized noise to keep it close to N(0, I),
preventing degenerate solutions while allowing meaningful steering.
"""

import math
from typing import Optional, Tuple

import torch
import torch.fft as fft
import torch.nn.functional as F


def spectral_whiteness_loss(noise, eps=1e-8):
    """Spatial power-spectrum flatness. Input: (B, T, C, H, W)."""
    B, T, C, H, W = noise.shape
    freq = fft.fftn(noise, dim=(-2, -1))
    power = torch.abs(freq) ** 2
    power_mean = power.mean(dim=(0, 1, 2)) / (H * W)
    return torch.log(power_mean + eps).pow(2).mean()


def temporal_spectral_flatness_loss(noise, eps=1e-8):
    """Temporal power-spectrum flatness. Input: (B, T, C, H, W)."""
    if noise.shape[1] < 2:
        return noise.new_zeros(())
    freq = fft.fft(noise, dim=1)
    power = (freq.abs() ** 2).mean(dim=(0, 2, 3, 4))[1:]
    logp = torch.log(power + eps)
    logp = logp - logp.mean()
    return (logp ** 2).mean()


def flat_iid_gram_regularizer(x, group_size=16, num_perms=100, eps=1e-8):
    """
    Flat i.i.d. independence regularizer. Input: (1, T, C, H, W).
    Penalizes deviation of random group Gram matrices from identity.
    """
    D = x.numel()
    flat = x.reshape(-1)
    k = group_size
    if k < 2 or D < k:
        return x.new_zeros(())
    n_groups = D // k
    total = x.new_zeros(())
    for _ in range(num_perms):
        perm = torch.randperm(D, device=x.device)
        Z = flat[perm[:n_groups * k]].reshape(n_groups, k)
        mu = Z.mean(dim=1, keepdim=True)
        mean_loss = (mu * mu).mean()
        Zc = Z - mu
        target_diag = float(k)
        diagG = (Zc * Zc).sum(dim=1)
        S = Zc.T @ Zc
        frobG2 = (S * S).sum()
        offdiag2 = frobG2 - (diagG * diagG).sum()
        diag_dev2 = ((diagG - target_diag) ** 2).sum()
        cov_loss = (offdiag2 + diag_dev2) / (n_groups * n_groups)
        total = total + (mean_loss + cov_loss)
    return (total / max(num_perms, 1)) / max(k, 1)


def compute_noise_regularizer(noise, kl_coeff_spherical=0.5, kl_coeff=10.0, std_coeff=0.1,
                               spectral_coeff=100.0, std_permutation_coeff=10.0,
                               num_perms=100, spectral_threshold=None):
    """Regularize optimized noise to stay close to N(0, I). noise: (1, T, C, H, W)."""
    B = noise.shape[0]
    noise_flat = noise.reshape(B, -1)

    target_norm = math.sqrt(noise_flat.shape[1])
    shrinkage_reg = ((noise_flat.norm(dim=1) - target_norm) ** 2).mean()

    threshold = 3.5
    outlier_penalty = ((torch.relu(torch.abs(noise) - threshold)) ** 2).mean()

    spec_loss = spectral_whiteness_loss(noise)

    sample_mean = torch.mean(noise)
    sample_mean_dimwise = torch.mean(noise, dim=(-1, -2))
    kl_reg = (
        sample_mean ** 2
        + (sample_mean_dimwise ** 2).mean()
        + outlier_penalty
        + spectral_coeff * torch.relu(spec_loss - (spectral_threshold if spectral_threshold is not None else spec_loss.detach()))
    )

    std_reg = ((noise_flat.std(dim=1) - 1.0) ** 2).mean()
    std_reg_dimwise = ((noise.std(dim=(-1, -2)) - 1.0) ** 2).mean()
    perm_reg = flat_iid_gram_regularizer(noise, group_size=16, num_perms=num_perms)
    std_reg = std_reg + std_reg_dimwise + std_permutation_coeff * torch.relu(perm_reg - 0.96)

    total = kl_coeff_spherical * shrinkage_reg + kl_coeff * kl_reg + std_coeff * std_reg
    metrics = {
        "reg/total": total.item(), "reg/shrinkage": shrinkage_reg.item(),
        "reg/kl": kl_reg.item(), "reg/std": std_reg.item(),
        "reg/spectral": spec_loss.item(), "reg/perm_iid": perm_reg.item(),
    }
    return total, metrics


def single_noise_regularizer(noise_t, kl_coeff_spherical=0.5, kl_coeff=10.0, std_coeff=0.1):
    """Lightweight regularizer for a single noise vector (1, C, H, W)."""
    flat = noise_t.reshape(1, -1)
    target_norm = math.sqrt(flat.shape[1])
    shrinkage = ((flat.norm(dim=1) - target_norm) ** 2).mean()
    mean_penalty = noise_t.mean() ** 2
    std_penalty = (noise_t.std() - 1.0) ** 2
    outlier = (torch.relu(torch.abs(noise_t) - 4.7) ** 2).mean()
    return (kl_coeff_spherical * shrinkage + kl_coeff * (mean_penalty + outlier) + std_coeff * std_penalty)


def compute_single_noise_metrics(noise_t):
    with torch.no_grad():
        noise_5d = noise_t.detach().unsqueeze(1)
        _, metrics = compute_noise_regularizer(noise_5d)
    return metrics


def iid_gram_regularizer_thw_unfold_efficient(
    x: torch.Tensor,
    patch_size=(5, 6, 8),
    stride=(5, 6, 8),
    num_perms: int = 100,
    eps: float = 1e-8,
    normalize: bool = False,
) -> torch.Tensor:
    """Volumetric i.i.d. gram regularizer over (T, H, W) patches. Input: (1, T, D, H, W)."""
    B, T, D, H, W = x.shape
    assert B == 1
    pt, ph, pw = patch_size
    st, sh, sw = stride
    k = pt * ph * pw

    xd = x[0].permute(1, 0, 2, 3).contiguous()  # (D, T, H, W)
    total = x.new_zeros(())

    for _ in range(num_perms):
        perm = torch.randperm(T * H * W, device=x.device)
        xd_perm = xd.reshape(D, -1).index_select(1, perm).reshape(D, T, H, W)

        patches = (
            xd_perm
            .unfold(1, pt, st)
            .unfold(2, ph, sh)
            .unfold(3, pw, sw)
        )
        Z = patches.reshape(-1, k)
        m = Z.shape[0]
        if m < 2:
            continue

        mu = Z.mean(dim=1, keepdim=True)
        mean_loss = (mu * mu).mean()
        Zc = Z - mu
        if normalize:
            Zc = Zc / (Zc.norm(dim=1, keepdim=True) + eps)
            target_diag = 1.0
        else:
            target_diag = float(k)

        diagG = (Zc * Zc).sum(dim=1)
        S = Zc.T @ Zc
        frobG2 = (S * S).sum()
        diagG2 = (diagG * diagG).sum()
        offdiag2 = frobG2 - diagG2
        diag_dev2 = ((diagG - target_diag) ** 2).sum()
        cov_loss = (offdiag2 + diag_dev2) / (m * m)
        total = total + (mean_loss + cov_loss)

    return (total / max(num_perms, 1)) / max(k, 1)


def patched_iid_gram_regularizer_efficient(
    x: torch.Tensor,
    patch_size=(12, 16),
    stride=(12, 16),
    num_perms: int = 100,
    eps: float = 1e-8,
    normalize: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """2D-patch i.i.d. gram regularizer over (H, W). Input: (1, T, D, H, W)."""
    B, T, D, H, W = x.shape
    assert B == 1

    if isinstance(patch_size, int):
        ph, pw = patch_size, patch_size
    else:
        ph, pw = patch_size
    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride
    k = ph * pw

    maps = x.reshape(B * T * D, 1, H, W).contiguous()
    nmap = maps.shape[0]
    flat0 = maps.reshape(nmap, 1, H * W)

    total = x.new_zeros(())

    for _ in range(num_perms):
        perm_hw = torch.randperm(H * W, device=x.device, generator=generator)
        maps_perm = flat0.index_select(2, perm_hw).reshape(nmap, 1, H, W)
        patches = F.unfold(maps_perm, kernel_size=(ph, pw), stride=(sh, sw))
        Z = patches.transpose(1, 2).contiguous().view(-1, k)
        m = Z.shape[0]
        if m < 2:
            continue

        mu = Z.mean(dim=1, keepdim=True)
        mean_loss = (mu * mu).mean()
        Zc = Z - mu
        if normalize:
            Zc = Zc / (Zc.norm(dim=1, keepdim=True) + eps)
            target_diag = 1.0
        else:
            target_diag = float(k)

        diagG = (Zc * Zc).sum(dim=1)
        S = Zc.T @ Zc
        frobG2 = (S * S).sum()
        diagG2 = (diagG * diagG).sum()
        offdiag2 = frobG2 - diagG2
        diag_dev2 = ((diagG - target_diag) ** 2).sum()
        cov_loss = (offdiag2 + diag_dev2) / (m * m)
        total = total + (mean_loss + cov_loss)

    return (total / max(num_perms, 1)) / max(k, 1)


def compute_regularizer_video(
    noise_mean: torch.Tensor,
    *,
    kl_coeff: float,
    kl_coeff_spherical: float,
    std_coeff: float,
    spectral_coeff: float = 100.0,
    std_permutation_coeff: float = 100.0,
    gram_normalize: bool = False,
    spectral_threshold: float = 0.0098,
    std_perm_activation: str = "relu_threshold",
    std_perm_threshold: float = 0.992,
    std_perm_include_patched: bool = False,
    outlier_threshold: float = 4.7,
    num_gram_perms: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Unified video noise regularizer subsuming Vista and Ctrl-World variants.

    noise_mean: (B, T, C, H, W). Vista uses gram_normalize=False, spectral_threshold=0.0098,
    std_perm_activation="relu_threshold", std_perm_threshold=0.992, std_perm_include_patched=False.
    Ctrl-World uses gram_normalize=True, spectral_threshold=0.055, std_perm_activation="square",
    std_perm_include_patched=True.

    Returns (reg_loss, shrink, kl, std, std_perm_mean, logpo_T, spec_loss).
    """
    B = noise_mean.shape[0]
    noise_flat = noise_mean.reshape(B, -1)

    target_norm = math.sqrt(noise_flat.shape[1])
    shrink_reg = ((noise_flat.norm(dim=1) - target_norm) ** 2).mean()

    excess = torch.relu(torch.abs(noise_mean) - outlier_threshold)
    outlier_penalty = (excess ** 2).mean()

    spec_loss = spectral_whiteness_loss(noise_mean)
    spec_loss_temporal = temporal_spectral_flatness_loss(noise_mean)

    sample_mean = torch.mean(noise_mean)
    sample_mean_dimwise = torch.mean(noise_mean, dim=[-1, -2])

    kl_reg = (
        sample_mean ** 2
        + (sample_mean_dimwise ** 2).mean()
        + outlier_penalty
        + spectral_coeff * torch.relu(spec_loss - spectral_threshold)
        + spectral_coeff * (spec_loss_temporal ** 2)
    )

    std_reg = ((noise_flat.std(dim=1) - 1.0) ** 2).mean()
    std_reg_dimwise = ((noise_mean.std(dim=[-1, -2]) - 1.0) ** 2).mean()

    perm_thw = iid_gram_regularizer_thw_unfold_efficient(
        noise_mean, normalize=gram_normalize, num_perms=num_gram_perms,
    )
    perm_patched = patched_iid_gram_regularizer_efficient(
        noise_mean, normalize=gram_normalize, num_perms=num_gram_perms,
    )
    perm_mean = (perm_thw + perm_patched) / 2.0

    if std_perm_activation == "relu_threshold":
        std_reg = std_reg + std_reg_dimwise + std_permutation_coeff * torch.relu(perm_thw - std_perm_threshold)
        if std_perm_include_patched:
            std_reg = std_reg + std_permutation_coeff * torch.relu(perm_patched - std_perm_threshold)
    elif std_perm_activation == "square":
        std_reg = std_reg + std_reg_dimwise + std_permutation_coeff * (perm_thw ** 2)
        if std_perm_include_patched:
            std_reg = std_reg + std_permutation_coeff * (perm_patched ** 2)
    else:
        raise ValueError(f"Unknown std_perm_activation: {std_perm_activation}")

    reg_loss = (
        kl_coeff_spherical * shrink_reg
        + kl_coeff * kl_reg
        + std_coeff * std_reg
    )

    logpo_T = None
    return reg_loss, shrink_reg, kl_reg, std_reg, perm_mean, logpo_T, spec_loss


def compute_reg(noise_t, args, use_full, spectral_threshold=None):
    """Select full or lightweight regularizer based on args."""
    if use_full:
        noise_5d = noise_t.unsqueeze(1)
        total, _ = compute_noise_regularizer(
            noise_5d,
            kl_coeff_spherical=args.kl_coeff_spherical,
            kl_coeff=args.kl_coeff,
            std_coeff=args.std_coeff,
            spectral_coeff=getattr(args, 'spectral_coeff', 100.0),
            std_permutation_coeff=getattr(args, 'std_permutation_coeff', 10.0),
            num_perms=getattr(args, 'num_perms', 100),
            spectral_threshold=spectral_threshold,
        )
        return total
    else:
        return single_noise_regularizer(
            noise_t,
            kl_coeff_spherical=args.kl_coeff_spherical,
            kl_coeff=args.kl_coeff,
            std_coeff=args.std_coeff,
        )
