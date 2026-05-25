from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictionHeadConfig:
    latent_dim: int = 4
    latent_spatial_size: int = 16
    hidden_dim: int = 256
    num_conv_layers: int = 2
    num_mlp_layers: int = 2


class LatentEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        in_ch = latent_dim
        out_ch = hidden_dim // 4
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            ])
            in_ch = out_ch
            out_ch = min(out_ch * 2, hidden_dim)
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, hidden_dim)

    def forward(self, z):
        h = self.pool(self.conv_layers(z)).squeeze(-1).squeeze(-1)
        return self.proj(h)


class MarginHead(nn.Module):
    def __init__(self, config: PredictionHeadConfig):
        super().__init__()
        self.encoder = LatentEncoder(config.latent_dim, config.hidden_dim, config.num_conv_layers)
        layers = []
        for _ in range(config.num_mlp_layers - 1):
            layers.extend([nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        return self.mlp(self.encoder(z))


class DoneHead(nn.Module):
    def __init__(self, config: PredictionHeadConfig):
        super().__init__()
        self.encoder = LatentEncoder(config.latent_dim, config.hidden_dim, config.num_conv_layers)
        layers = []
        for _ in range(config.num_mlp_layers - 1):
            layers.extend([nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        return self.mlp(self.encoder(z))


class PredictionHeads(nn.Module):
    """Prediction heads with continuous margin (MSE) and binary done (BCE)."""

    def __init__(self, config: PredictionHeadConfig):
        super().__init__()
        self.config = config
        self.margin_head = MarginHead(config)
        self.done_head = DoneHead(config)

    def forward(self, z):
        return {'margin': self.margin_head(z), 'done': self.done_head(z)}

    def compute_losses(self, predictions, targets):
        def ensure_shape(x):
            return x.unsqueeze(-1) if x.ndim == 1 else x

        pred_margin = predictions['margin']
        pred_done = predictions['done']
        target_margin = ensure_shape(targets['margin'])
        target_done = ensure_shape(targets['done'])

        loss_margin = F.mse_loss(pred_margin, target_margin)
        loss_done = F.binary_cross_entropy_with_logits(pred_done, target_done)
        total_loss = loss_margin + loss_done

        with torch.no_grad():
            margin_mae = (pred_margin - target_margin).abs().mean()
            done_acc = ((torch.sigmoid(pred_done) > 0.5).float() == target_done).float().mean()

        return total_loss, {
            'loss_margin': loss_margin.item(),
            'loss_done': loss_done.item(),
            'margin_mae': margin_mae.item(),
            'done_accuracy': done_acc.item(),
        }
