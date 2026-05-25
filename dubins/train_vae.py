"""
Train the VAE + prediction heads for the Dubins car world model.

Usage:
    python dubins/train_vae.py --config dubins/config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from dubins.models.vae import ImageVAE, VAEConfig
from dubins.models.prediction_heads import PredictionHeads, PredictionHeadConfig
from wm_steer.utils.logger import Logger
from dubins.dataset import create_dataloader


def convert_to_float(value):
    return float(value)


def visualize_reconstruction(images, recon, num_samples=4):
    try:
        import wandb
    except ImportError:
        return None
    num_samples = min(num_samples, images.shape[0])
    images = ((images[:num_samples] + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
    recon = ((recon[:num_samples] + 1) / 2).detach().cpu().numpy().transpose(0, 2, 3, 1)
    images = np.clip(images, 0, 1)
    recon = np.clip(recon, 0, 1)
    grid = np.concatenate([np.concatenate([images[i], recon[i]], axis=1) for i in range(num_samples)], axis=0)
    return wandb.Image((grid * 255).astype(np.uint8), caption="Left: Original | Right: Reconstruction")


def train_vae(config):
    device = torch.device(config['training']['device'])
    curr_time = datetime.now().strftime("%m%d/%H%M%S")

    finetune_heads_only = config.get('training', {}).get('finetune_heads_only', False)
    checkpoint_path_for_finetune = config.get('training', {}).get('checkpoint_path', None)

    checkpoint_dir = Path(config['training']['checkpoint_dir']) / "vae" / curr_time
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(use_wandb=True)
    wandb_config = config.get('wandb', {})
    logger.init_wandb(
        project=wandb_config.get('project', 'diffusion_world_model'),
        name=f"{curr_time}_{wandb_config.get('vae_run_name', 'vae')}",
        config=config,
    )

    vae = ImageVAE(VAEConfig(**config['vae'])).to(device)

    prediction_heads = None
    if 'prediction_heads' in config:
        ph = config['prediction_heads']
        latent_spatial_size = config['vae']['resolution'] // (2 ** len(config['vae']['ch_mult']))
        pred_cfg = PredictionHeadConfig(
            latent_dim=config['vae']['z_channels'],
            latent_spatial_size=latent_spatial_size,
            hidden_dim=ph['hidden_dim'],
            num_conv_layers=ph['num_conv_layers'],
            num_mlp_layers=ph['num_mlp_layers'],
        )
        prediction_heads = PredictionHeads(pred_cfg).to(device)

    if finetune_heads_only:
        assert checkpoint_path_for_finetune is not None, "checkpoint_path required for finetune_heads_only"
        ckpt = torch.load(checkpoint_path_for_finetune, map_location=device)
        vae.load_state_dict(ckpt['model_state_dict'])
        if 'prediction_heads_state_dict' in ckpt and prediction_heads is not None:
            prediction_heads.load_state_dict(ckpt['prediction_heads_state_dict'])
        for p in vae.parameters():
            p.requires_grad = False
        logger.print("VAE frozen — head-only finetuning")

    if finetune_heads_only:
        optimizer = Adam(prediction_heads.parameters(), lr=convert_to_float(config['training']['vae_lr']))
    else:
        params = list(vae.parameters())
        if prediction_heads is not None:
            params += list(prediction_heads.parameters())
        optimizer = Adam(params, lr=convert_to_float(config['training']['vae_lr']))

    kl_weight = config.get('vae_loss', {}).get('kl_weight', 1e-6)

    obs_config = None
    if prediction_heads is not None:
        ph = config['prediction_heads']
        obs_config = {'x': ph['obs_x'], 'y': ph['obs_y'], 'r': ph['obs_r']}

    train_loader = create_dataloader(
        dataset_path=config['data']['dataset_path'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        split='train',
        num_train_trajs=config['data']['num_train_trajs'],
        num_workers=config['data']['num_workers'],
        shuffle=True,
        obs_config=obs_config,
    )
    val_loader = create_dataloader(
        dataset_path=config['data']['dataset_path'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        split='val',
        num_train_trajs=config['data']['num_train_trajs'],
        num_workers=config['data']['num_workers'],
        shuffle=False,
        obs_config=obs_config,
    )

    logger.print(f"Training VAE: {len(train_loader)} batches/epoch")
    global_step = 0

    for epoch in range(config['training']['vae_epochs']):
        vae.train()
        if prediction_heads is not None:
            prediction_heads.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['vae_epochs']}")
        for batch in pbar:
            images = batch['images'].to(device)
            b, t, c, h, w = images.shape
            images_flat = images.reshape(b * t, c, h, w)

            recon, posterior = vae(images_flat, sample_posterior=True)

            if finetune_heads_only:
                total_loss = torch.tensor(0.0, device=device)
                recon_loss = kl_loss = total_loss
            else:
                recon_loss = F.mse_loss(recon, images_flat)
                kl_loss = posterior.kl().mean()
                total_loss = recon_loss + kl_weight * kl_loss

            heads_metrics = {}
            if prediction_heads is not None:
                margins = batch['margins'].to(device).reshape(b * t)
                dones = batch['dones'].to(device).reshape(b * t)
                with torch.no_grad():
                    latents = vae.encode(images_flat).mode()
                predictions = prediction_heads(latents)
                heads_loss, heads_metrics = prediction_heads.compute_losses(
                    predictions, {'margin': margins, 'done': dones}
                )
                margin_weight = config['prediction_heads'].get('margin_weight', 1.0)
                done_weight = config['prediction_heads'].get('done_weight', 1.0)
                total_loss = total_loss + heads_loss * (margin_weight + done_weight) / 2.0

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if global_step % config['training']['log_every'] == 0:
                log_dict = {
                    'train/total_loss': total_loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                }
                if heads_metrics:
                    log_dict.update({f'train/{k}': v for k, v in heads_metrics.items()})
                logger.log(log_dict, step=global_step)

                if global_step % (config['training']['log_every'] * 10) == 0:
                    vis = visualize_reconstruction(images_flat, recon)
                    if vis is not None:
                        logger.log_images({'train/reconstruction': vis}, step=global_step)

            postfix = {'total': f"{total_loss.item():.4f}", 'recon': f"{recon_loss.item():.4f}"}
            if heads_metrics:
                postfix['m_mae'] = f"{heads_metrics.get('margin_mae', 0):.4f}"
            pbar.set_postfix(postfix)
            global_step += 1

        # Validation
        if (epoch + 1) % config['training']['save_every'] == 0:
            vae.eval()
            if prediction_heads is not None:
                prediction_heads.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(device)
                    b, t, c, h, w = images.shape
                    images_flat = images.reshape(b * t, c, h, w)
                    recon, posterior = vae(images_flat, sample_posterior=False)
                    loss = F.mse_loss(recon, images_flat) + kl_weight * posterior.kl().mean()
                    val_loss += loss.item()
            logger.print(f"Epoch {epoch+1}: val_loss={val_loss / len(val_loader):.4f}")
            logger.log({'val/loss': val_loss / len(val_loader)}, step=global_step)

            # Save checkpoint
            ckpt_dict = {
                'epoch': epoch + 1,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }
            if prediction_heads is not None:
                ckpt_dict['prediction_heads_state_dict'] = prediction_heads.state_dict()
            torch.save(ckpt_dict, checkpoint_dir / f"vae_epoch_{epoch+1}.pt")

    final_dict = {
        'epoch': config['training']['vae_epochs'],
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    if prediction_heads is not None:
        final_dict['prediction_heads_state_dict'] = prediction_heads.state_dict()
    torch.save(final_dict, checkpoint_dir / "vae_final.pt")
    logger.print(f"Saved final VAE to {checkpoint_dir / 'vae_final.pt'}")
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_vae(config)
