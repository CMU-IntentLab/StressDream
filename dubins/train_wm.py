"""
Train the diffusion world model (denoiser) for the Dubins car.
Requires a pretrained VAE checkpoint.

Usage:
    python dubins/train_wm.py --config dubins/config.yaml \
        --vae_checkpoint dubins/checkpoints/vae/vae_final.pt
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
from torch.optim import Adam
from tqdm import tqdm

from dubins.models import DiffusionWorldModel, DiffusionWorldModelConfig, VAEConfig, DiffusionSamplerConfig
from dubins.models.latent_denoiser import SigmaDistributionConfig
from wm_steer.utils.logger import Logger
from dubins.dataset import create_dataloader


def convert_to_float(value):
    return float(value)


def train_world_model(config, vae_checkpoint_path):
    device = torch.device(config['training']['device'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    curr_time = datetime.now().strftime("%m%d/%H%M%S")

    checkpoint_dir = Path(config['training']['checkpoint_dir']) / "world_model" / f"{curr_time}_{config['training'].get('remark', 'wm')}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config['training'].get('log_dir', 'logs')) / "world_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f"training_{timestamp}.log")

    logger = Logger(use_wandb=True, log_file=log_file)
    wandb_config = config.get('wandb', {})
    logger.init_wandb(
        project=wandb_config.get('project', 'diffusion_world_model'),
        name=f"{curr_time}_{config['training'].get('remark', 'wm')}",
        config=config,
    )

    model_config = DiffusionWorldModelConfig(
        vae=VAEConfig(**config['vae']),
        denoiser=config['denoiser'],
        sampler=DiffusionSamplerConfig(**config['sampler']),
        prediction_heads=None,
    )
    model = DiffusionWorldModel(model_config).to(device)

    logger.print(f"Loading pretrained VAE from {vae_checkpoint_path}")
    vae_ckpt = torch.load(vae_checkpoint_path, map_location=device)
    model.vae.load_state_dict(vae_ckpt['model_state_dict'])
    for p in model.vae.parameters():
        p.requires_grad = False
    model.vae.eval()

    sigma_cfg = SigmaDistributionConfig(**config['sigma_distribution'])
    model.setup_training(sigma_cfg)

    resume_checkpoint = config['training'].get('resume_checkpoint', None)
    if resume_checkpoint is not None:
        logger.print(f"Resuming from {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    optimizer = Adam(model.denoiser.parameters(), lr=convert_to_float(config['training']['denoiser_lr']))

    train_loader = create_dataloader(
        dataset_path=config['data']['dataset_path'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        split='train',
        num_train_trajs=config['data']['num_train_trajs'],
        num_workers=config['data']['num_workers'],
        shuffle=True,
    )
    val_loader = create_dataloader(
        dataset_path=config['data']['dataset_path'],
        sequence_length=config['data']['sequence_length'],
        batch_size=config['data']['batch_size'],
        split='val',
        num_train_trajs=None,
        num_workers=config['data']['num_workers'],
        shuffle=False,
    )

    logger.print(f"Training world model: {len(train_loader)} batches/epoch, "
                 f"{sum(p.numel() for p in model.denoiser.parameters())} denoiser params")

    global_step = 0
    for epoch in range(config['training']['denoiser_epochs']):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['denoiser_epochs']}")
        for batch in pbar:
            model.denoiser.train()
            images = batch['images'].to(device)
            actions = batch['actions'].to(device)

            loss, metrics = model.train_denoiser(images, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if global_step % config['training']['log_every'] == 0:
                logger.log({'train/denoising_loss': loss.item(), 'train/epoch': epoch + 1}, step=global_step)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        logger.print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        if (epoch + 1) % config['training'].get('eval_every', 1) == 0:
            model.denoiser.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(device)
                    actions = batch['actions'].to(device)
                    loss, _ = model.train_denoiser(images, actions)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            logger.print(f"Validation: loss={avg_val:.4f}")
            logger.log({'val/denoising_loss': avg_val}, step=global_step)

        if (epoch + 1) % config['training']['save_every'] == 0:
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'denoiser_state_dict': model.denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }
            torch.save(ckpt, checkpoint_dir / "world_model_latest.pt")
            with open(checkpoint_dir / "config.yaml", 'w') as f:
                yaml.dump(config, f)
            logger.print(f"Saved checkpoint to {checkpoint_dir / 'world_model_latest.pt'}")

    final_path = checkpoint_dir / "world_model_final.pt"
    torch.save({
        'epoch': config['training']['denoiser_epochs'],
        'model_state_dict': model.state_dict(),
        'denoiser_state_dict': model.denoiser.state_dict(),
        'config': config,
    }, final_path)
    logger.print(f"Training complete! Saved to {final_path}")
    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to pretrained VAE checkpoint')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_world_model(config, args.vae_checkpoint)
