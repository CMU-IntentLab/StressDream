"""
Noise regularizers for Diffusion Noise Optimization (DNO).

These functions penalize the optimized noise to keep it close to N(0, I),
preventing degenerate solutions while allowing meaningful steering.
"""

import math
import torch
import torch.fft as fft


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
