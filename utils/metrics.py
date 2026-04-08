# -*- coding: utf-8 -*-
"""
Evaluation metrics for pansharpening.

Includes:
    - Reduced Resolution metrics: Q2n, SAM, ERGAS, SCC
    - Full Resolution metrics: D_lambda, D_S, HQNR
"""
import torch
import numpy as np
from einops import rearrange


def calculate_metrics(sr, gt):
    """
    Calculate Reduced Resolution (RR) metrics.

    Args:
        sr: Super-resolved image (B, C, H, W) in range [0, 1]
        gt: Ground truth image (B, C, H, W) in range [0, 1]

    Returns:
        Dictionary containing Q2n, SAM, ERGAS, SCC
    """
    metrics = {"q2n": 0., "sam": 0., "ergas": 0., "scc": 0.}

    with torch.no_grad():
        sr = torch.clamp(sr, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        for i in range(sr.shape[0]):
            img_sr = rearrange(sr[i], 'c h w -> h w c').cpu().numpy()
            img_gt = rearrange(gt[i], 'c h w -> h w c').cpu().numpy()

            # Calculate metrics for single image
            q2n_val = q2n_index(img_sr, img_gt)
            sam_val = sam_index(img_sr, img_gt)
            ergas_val = ergas_index(img_sr, img_gt, ratio=4)
            scc_val = scc_index(img_sr, img_gt)

            metrics["q2n"] += q2n_val
            metrics["sam"] += sam_val
            metrics["ergas"] += ergas_val
            metrics["scc"] += scc_val

        # Average over batch
        for k in metrics:
            metrics[k] /= sr.shape[0]

    return metrics


def calculate_metrics_fr(sr, lr, pan, sensor, ms, gt):
    """
    Calculate Full Resolution (FR) metrics.

    Args:
        sr: Super-resolved image (B, C, H, W)
        lr: Low-resolution multi-spectral image (B, C, H//4, W//4)
        pan: Panchromatic image (B, 1, H, W)
        sensor: Sensor type (e.g., 'GF1', 'WV2')
        ms: Original multi-spectral image (upsampled)
        gt: Ground truth for FR (original LR upsampled)

    Returns:
        Dictionary containing D_lambda, D_S, HQNR
    """
    metrics = {'D_lambda': 0., 'D_S': 0., 'HQNR': 0.}

    with torch.no_grad():
        sr = torch.clamp(sr, 0.0, 1.0)

        for i in range(sr.shape[0]):
            img_sr = rearrange(sr[i], 'c h w -> h w c').cpu().numpy()
            img_lr = rearrange(lr[i], 'c h w -> h w c').cpu().numpy()
            img_pan = rearrange(pan[i], 'c h w -> h w c').cpu().numpy()
            img_ms = rearrange(ms[i], 'c h w -> h w c').cpu().numpy()

            # Calculate D_lambda and D_S
            d_lambda = d_lambda_index(img_sr, img_ms, img_pan, ratio=4)
            d_s = d_s_index(img_sr, img_ms, img_pan, ratio=4)
            hqnr = (1 - abs(d_lambda)) * (1 - abs(d_s))

            metrics['D_lambda'] += d_lambda
            metrics['D_S'] += d_s
            metrics['HQNR'] += hqnr

        # Average over batch
        for k in metrics:
            metrics[k] /= sr.shape[0]

    return metrics


def q2n_index(img1, img2, block_size=32):
    """Calculate Q2n (Quality with No Reference) index."""
    N, M, _ = img1.shape
    # Simplified implementation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Compute means
    mu1 = np.mean(img1, axis=(0, 1))
    mu2 = np.mean(img2, axis=(0, 1))

    # Compute variances
    sigma1_sq = np.var(img1, axis=(0, 1))
    sigma2_sq = np.var(img2, axis=(0, 1))

    # Compute covariance
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(0, 1))

    # Q2n calculation
    c1 = 0.0001
    c2 = 0.0001

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    q = numerator / denominator
    return np.mean(q)


def sam_index(img1, img2):
    """Calculate SAM (Spectral Angle Mapper) index."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    numerator = np.sum(img1 * img2, axis=2)
    denominator = np.sqrt(np.sum(img1**2, axis=2) * np.sum(img2**2, axis=2))

    sam = np.arccos(np.clip(numerator / (denominator + 1e-8), -1, 1))
    return np.mean(sam) * 180 / np.pi


def ergas_index(img1, img2, ratio=4):
    """Calculate ERGAS (Relative Dimensionless Global Error) index."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    bands = img1.shape[2]
    sum_squared_error = 0

    for b in range(bands):
        band1 = img1[:, :, b]
        band2 = img2[:, :, b]
        rmse = np.sqrt(np.mean((band1 - band2)**2))
        mean_ref = np.mean(band2)
        sum_squared_error += (rmse / mean_ref)**2 if mean_ref > 0 else 0

    ergas = 100 * ratio * np.sqrt(sum_squared_error / bands)
    return ergas


def scc_index(img1, img2):
    """Calculate SCC (Spatial Correlation Coefficient) index."""
    from scipy import signal

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    bands = img1.shape[2]
    correlations = []

    for b in range(bands):
        band1 = img1[:, :, b]
        band2 = img2[:, :, b]

        # Compute correlation coefficient
        corr = np.corrcoef(band1.flatten(), band2.flatten())[0, 1]
        correlations.append(corr)

    return np.mean(correlations)


def d_lambda_index(img_fused, img_ms, img_pan, ratio=4, Qblocks_size=32):
    """Calculate D_lambda (Spectral Distortion) index."""
    # Simplified implementation
    img_fused = img_fused.astype(np.float64)
    img_ms = img_ms.astype(np.float64)

    # Downsample fused image
    h, w = img_ms.shape[:2]
    img_fused_down = img_fused[::ratio, ::ratio, :]

    if img_fused_down.shape[0] > h:
        img_fused_down = img_fused_down[:h, :w, :]

    # Calculate correlation between bands
    d_lambda = 0
    bands = img_ms.shape[2]

    for b in range(bands):
        corr = np.corrcoef(img_fused_down[:, :, b].flatten(),
                          img_ms[:, :, b].flatten())[0, 1]
        d_lambda += abs(1 - corr)

    return d_lambda / bands


def d_s_index(img_fused, img_ms, img_pan, ratio=4, Qblocks_size=32):
    """Calculate D_S (Spatial Distortion) index."""
    from scipy import signal

    img_fused = img_fused.astype(np.float64)
    img_pan = img_pan.astype(np.float64)

    # Calculate correlation with panchromatic band
    d_s = 0
    bands = img_fused.shape[2]

    for b in range(bands):
        corr = np.corrcoef(img_fused[:, :, b].flatten(),
                          img_pan.flatten())[0, 1]
        d_s += abs(1 - corr)

    return d_s / bands


if __name__ == '__main__':
    # Test metrics
    sr = torch.rand(2, 8, 256, 256)
    gt = torch.rand(2, 8, 256, 256)

    metrics = calculate_metrics(sr, gt)
    print("Test metrics:", metrics)
