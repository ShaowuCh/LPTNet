# -*- coding: utf-8 -*-
"""
Training script for LPTNet.

Example usage:
    python train.py --config configs/lptnet_gf1.yaml
"""
import os
import sys
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.lptnet import LPTNet
from data.dataset import PanDataModule
from utils.metrics import calculate_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Forward
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out['sr'], batch['GT'])

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    metrics_sum = {'q2n': 0., 'sam': 0., 'ergas': 0., 'scc': 0.}

    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        out = model(batch)
        metrics = calculate_metrics(out['sr'], batch['GT'])

        for k, v in metrics.items():
            metrics_sum[k] += v

    # Average metrics
    for k in metrics_sum:
        metrics_sum[k] /= len(dataloader)

    return metrics_sum


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model_cfg = config['model']
    model = LPTNet(**model_cfg)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloaders
    data_cfg = config['data']
    datamodule = PanDataModule(
        train_root=data_cfg.get('train_root'),
        val_root=data_cfg.get('val_root'),
        batch_size=data_cfg.get('batch_size', 4),
        num_workers=data_cfg.get('num_workers', 4),
        img_range=data_cfg.get('img_range', 1023),
        lr_size=data_cfg.get('lr_size', 64)
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(0.9, 0.999),
        weight_decay=0.02
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler_step'],
        gamma=config['training']['scheduler_gamma']
    )

    # Create log directory
    log_dir = os.path.join('logs', config['name'] + '_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Tensorboard
    writer = SummaryWriter(log_dir)

    # Training loop
    best_q2n = 0
    epochs = config['training']['epochs']

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.6f}")
        writer.add_scalar('train/loss', train_loss, epoch)

        # Validate
        if epoch % config['training']['val_every'] == 0:
            val_metrics = validate(model, val_loader, device)
            print(f"Val Metrics: Q2N={val_metrics['q2n']:.4f}, SAM={val_metrics['sam']:.4f}, "
                  f"ERGAS={val_metrics['ergas']:.4f}, SCC={val_metrics['scc']:.4f}")

            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            # Save best model
            if val_metrics['q2n'] > best_q2n:
                best_q2n = val_metrics['q2n']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                }, os.path.join(log_dir, 'checkpoints', 'best.pth'))
                print(f"Saved best model with Q2N: {best_q2n:.4f}")

        # Step scheduler
        scheduler.step()
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Save last model
        if epoch % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(log_dir, 'checkpoints', f'epoch_{epoch:03d}.pth'))

    print(f"\nTraining completed! Best Q2N: {best_q2n:.4f}")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LPTNet')
    parser.add_argument('--config', type=str, default='configs/lptnet_gf1.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    main(args)
