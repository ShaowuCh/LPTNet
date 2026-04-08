# -*- coding: utf-8 -*-
"""
Test script for LPTNet.

Example usage:
    python test.py --config configs/lptnet_gf1.yaml --checkpoint checkpoints/best.pth
"""
import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.lptnet import LPTNet
from data.dataset import PanDataset
from utils.metrics import calculate_metrics, calculate_metrics_fr


@torch.no_grad()
def test_rr(model, dataloader, device, save_dir=None):
    """
    Test in Reduced Resolution (RR) mode.

    Args:
        model: LPTNet model
        dataloader: Test dataloader
        device: Device to run on
        save_dir: Directory to save results (optional)
    """
    model.eval()
    metrics_sum = {'q2n': 0., 'sam': 0., 'ergas': 0., 'scc': 0.}
    results = []

    pbar = tqdm(dataloader, desc='Testing (RR)')
    for idx, batch in enumerate(pbar):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        out = model(batch)
        metrics = calculate_metrics(out['sr'], batch['GT'])

        for k, v in metrics.items():
            metrics_sum[k] += v

        results.append({
            'sr': out['sr'].cpu(),
            'gt': batch['GT'].cpu(),
            'lms': batch['LR'].cpu(),
            'pan': batch['REF'].cpu(),
            'metrics': metrics
        })

        pbar.set_postfix({k: f'{v:.4f}' for k, v in metrics.items()})

    # Average metrics
    for k in metrics_sum:
        metrics_sum[k] /= len(dataloader)

    print("\n" + "=" * 50)
    print("Reduced Resolution Test Results:")
    print("=" * 50)
    for k, v in metrics_sum.items():
        print(f"{k.upper()}: {v:.4f}")
    print("=" * 50)

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_results(results, save_dir, dataloader.dataset.img_range)

    return metrics_sum, results


@torch.no_grad()
def test_fr(model, dataloader, device, sensor='GF1', save_dir=None):
    """
    Test in Full Resolution (FR) mode.

    Args:
        model: LPTNet model
        dataloader: Test dataloader
        device: Device to run on
        sensor: Sensor type for metric calculation
        save_dir: Directory to save results (optional)
    """
    model.eval()
    metrics_sum = {'D_lambda': 0., 'D_S': 0., 'HQNR': 0.}
    results = []

    pbar = tqdm(dataloader, desc='Testing (FR)')
    for idx, batch in enumerate(pbar):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        out = model(batch)

        # Interpolate LR for metric calculation
        lms_up = nn.functional.interpolate(batch['LR'], scale_factor=4, mode='bilinear')

        metrics = calculate_metrics_fr(
            out['sr'], batch['LR'], batch['REF'],
            sensor, lms_up, batch['GT']
        )

        for k, v in metrics.items():
            metrics_sum[k] += v

        results.append({
            'sr': out['sr'].cpu(),
            'lms': batch['LR'].cpu(),
            'pan': batch['REF'].cpu(),
            'gt': batch['GT'].cpu(),
            'metrics': metrics
        })

        pbar.set_postfix({k: f'{v:.4f}' for k, v in metrics.items()})

    # Average metrics
    for k in metrics_sum:
        metrics_sum[k] /= len(dataloader)

    print("\n" + "=" * 50)
    print("Full Resolution Test Results:")
    print("=" * 50)
    for k, v in metrics_sum.items():
        print(f"{k.upper()}: {v:.4f}")
    print("=" * 50)

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_results(results, save_dir, dataloader.dataset.img_range, is_fr=True)

    return metrics_sum, results


def save_results(results, save_dir, img_range, is_fr=False):
    """Save test results as numpy arrays."""
    print(f"Saving results to {save_dir}...")

    sr_list = []
    lms_list = []
    pan_list = []
    gt_list = []

    for idx, res in enumerate(results):
        # Denormalize and convert to numpy
        def to_numpy(x):
            x = x.squeeze(0) * img_range
            return x.permute(1, 2, 0).numpy()

        sr_list.append(to_numpy(res['sr']))
        lms_list.append(to_numpy(res['lms']))
        pan_list.append(to_numpy(res['pan']))
        gt_list.append(to_numpy(res['gt']))

    np.save(os.path.join(save_dir, 'sr.npy'), np.array(sr_list))
    np.save(os.path.join(save_dir, 'lms.npy'), np.array(lms_list))
    np.save(os.path.join(save_dir, 'pan.npy'), np.array(pan_list))
    np.save(os.path.join(save_dir, 'gt.npy'), np.array(gt_list))

    print("Results saved!")


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

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Create dataloaders
    data_cfg = config['data']

    # Test in Reduced Resolution (RR) mode
    if data_cfg.get('test_rr_root'):
        print("\nTesting in Reduced Resolution mode...")
        rr_dataset = PanDataset(
            data_cfg['test_rr_root'],
            phase='test',
            img_range=data_cfg.get('img_range', 1023)
        )
        rr_loader = torch.utils.data.DataLoader(
            rr_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        rr_metrics, rr_results = test_rr(
            model, rr_loader, device,
            save_dir=os.path.join(args.output_dir, 'rr') if args.output_dir else None
        )

    # Test in Full Resolution (FR) mode
    if data_cfg.get('test_fr_root'):
        print("\nTesting in Full Resolution mode...")
        fr_dataset = PanDataset(
            data_cfg['test_fr_root'],
            phase='test',
            img_range=data_cfg.get('img_range', 1023)
        )
        fr_loader = torch.utils.data.DataLoader(
            fr_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        sensor = data_cfg.get('sensor', 'GF1')
        fr_metrics, fr_results = test_fr(
            model, fr_loader, device, sensor,
            save_dir=os.path.join(args.output_dir, 'fr') if args.output_dir else None
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test LPTNet')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save output results')
    args = parser.parse_args()

    main(args)
