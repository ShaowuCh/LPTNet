# LPTNet: Modeling Dataset-level Priors with Learnable Probability Tables for Pansharpening

This is the official PyTorch implementation of **LPTNet** from the paper:

> **Modeling Dataset-level Priors with Learnable Probability Tables for Pansharpening**
> Shaowu Wu, Lihui Chen, Lihua Jian, Gemine Vivone, Kejiang Xiao, Xiaoguang Niu
> *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*

## Overview

LPTNet introduces a novel approach to pansharpening by modeling **dataset-level priors** through **Learnable Probability Tables (LPT)**. Unlike traditional methods that rely solely on image-level information, LPTNet captures global statistical patterns across the entire training dataset, enabling more robust and generalized pansharpening performance.

### Key Features

- **Learnable Probability Table (LPT)**: A dataset-level learnable module that captures global statistical priors
- **Token Adaptive Transformer (TAT)**: Integrates LPT for token-adaptive feature transformation
- **Multi-Scale Architecture**: Employs Token Adaptive Transformers at multiple scales for comprehensive feature extraction
- **State-of-the-Art Performance**: Achieves superior results on multiple satellite datasets

## Architecture

```
LPTNet
├── Patch Embedding
├── Encoder (Multi-Scale)
│   ├── Encoder Level 1 (TAT)
│   ├── Encoder Level 2 (TAT)
│   └── Encoder Level 3 (TAT)
├── Latent (TAT)
└── Decoder (Multi-Scale with Skip Connections)
    ├── Decoder Level 3 (TAT)
    ├── Decoder Level 2 (TAT)
    └── Decoder Level 1 (TAT)
```

**Core Modules:**
- **LearnableProbabilityTable**: Implements the dataset-level learnable probability table
- **TokenAdaptiveTransformer**: Combines LPT with feed-forward networks for adaptive feature transformation
- **MultiScaleLearnableProbabilityBlock**: The main U-Net style architecture with TAT modules

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0 (for GPU training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LPTNet.git
cd LPTNet

# Create conda environment
conda create -n lptnet python=3.9
conda activate lptnet

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops numpy scipy pyyaml tqdm tensorboard
```

## Dataset Preparation

The code expects the following directory structure:

```
data/
├── GF1/
│   ├── train/
│   │   ├── LR/          # Low-resolution MS images (.npy format)
│   │   ├── REF/         # High-resolution PAN images (.npy format)
│   │   └── GT/          # Ground truth HR-MS images (.npy format)
│   ├── val/
│   │   ├── LR/
│   │   ├── REF/
│   │   └── GT/
│   ├── test_rr/         # Reduced resolution test set
│   │   ├── LR/
│   │   ├── REF/
│   │   └── GT/
│   └── test_fr/         # Full resolution test set
│       ├── LR/
│       ├── REF/
│       └── GT/
└── WV2/                 # Similar structure for other datasets
```

**Note**: Images should be saved as `.npy` files with shape `(H, W, C)` and dtype `float32` or `uint16`.

## Training

### Quick Start

Train LPTNet on GF1 dataset:

```bash
python train.py --config configs/lptnet_gf1.yaml
```

### Custom Configuration

You can modify the configuration file or pass arguments directly:

```bash
python train.py --config configs/lptnet_gf1.yaml \
    --resume checkpoints/best.pth
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.dim` | Base feature dimension | 32 |
| `model.num_tokens` | Number of tokens in LPT | 256 |
| `model.ms_chans` | Number of MS bands | 4 (QB, GF2) or 8 (WV2) |
| `data.batch_size` | Training batch size | 4 |
| `training.lr` | Learning rate | 1e-4 |
| `training.epochs` | Total training epochs | 600 |

## Testing

Test a trained model:

```bash
# Reduced Resolution (RR) Test
python test.py \
    --config configs/lptnet_gf1.yaml \
    --checkpoint checkpoints/best.pth \
    --output_dir results/gf1

# The test script will automatically test both RR and FR modes
```

## Evaluation Metrics

### Reduced Resolution (RR) Metrics
- **Q2n** (Quality with No Reference): Overall image quality
- **SAM** (Spectral Angle Mapper): Spectral distortion
- **ERGAS** (Relative Dimensionless Global Error): Global error
- **SCC** (Spatial Correlation Coefficient): Spatial correlation

### Full Resolution (FR) Metrics
- **D_λ** (Spectral Distortion Index): Spectral fidelity
- **D_S** (Spatial Distortion Index): Spatial fidelity
- **HQNR** (Hybrid Quality with No Reference): Overall quality

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{wu2026lptnet,
  title={Modeling Dataset-level Priors with Learnable Probability Tables for Pansharpening},
  author={Wu, Shaowu and Chen, Lihui and Jian, Lihua and Vivone, Gemine and Xiao, Kejiang and Niu, Xiaoguang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

