# -*- coding: utf-8 -*-
"""
Dataset module for LPTNet pansharpening.

Supports loading multi-spectral and panchromatic image pairs for training and testing.
"""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def get_image_paths(path, ext):
    """Get all image paths with given extension from a directory."""
    assert os.path.isdir(path), f'[Error] [{path}] is not a valid directory'
    images = glob.glob(os.path.join(path, '*' + ext))
    images.sort()
    assert images, f'[{path}] has no valid file'
    return images


def read_img(path, ext):
    """Read image from file."""
    if ext == '.npy':
        img = np.load(path)
    else:
        raise NotImplementedError(f'Cannot read this type ({ext}) of data')
    if isinstance(img, np.ndarray) and img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def np2tensor(img, img_range, run_range=1.0):
    """Convert numpy array to tensor and normalize."""
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) / 1.0
    tensor = torch.from_numpy(np_transpose).float()
    tensor = tensor.mul_(run_range / img_range)
    return tensor


def augment(input_dict, hflip=True, rot=True):
    """Apply random augmentation (horizontal flip and rotation)."""
    hflip_flag = hflip and random.random() < 0.5
    vflip_flag = rot and random.random() < 0.5
    rot90_flag = rot and random.random() < 0.5

    def _augment(img):
        if hflip_flag:
            img = img[:, ::-1, :]
        if vflip_flag:
            img = img[::-1, :, :]
        if rot90_flag:
            img = img.transpose(1, 0, 2)
        return img

    return {k: _augment(v) for k, v in input_dict.items()}


class PanDataset(Dataset):
    """
    Dataset for pansharpening.

    Expected directory structure:
        data_root/
            LR/     - Low-resolution multi-spectral images
            REF/    - High-resolution panchromatic images
            GT/     - Ground truth high-resolution multi-spectral images (for training/val)
    """
    def __init__(self, data_root, phase='train', img_range=1023, lr_size=64):
        """
        Args:
            data_root: Root directory of the dataset
            phase: 'train', 'val', or 'test'
            img_range: Maximum value of the images (for normalization)
            lr_size: Size of low-resolution patches for training
        """
        super(PanDataset, self).__init__()
        self.phase = phase
        self.lr_size = lr_size
        self.img_range = img_range

        # Get image paths
        self.img_paths_dict = {}
        for key in ['LR', 'REF', 'GT']:
            path = os.path.join(data_root, key)
            if os.path.exists(path):
                self.img_paths_dict[key] = get_image_paths(path, ".npy")

        self.data_len = len(self.img_paths_dict['REF'])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # Read images
        pathdict = {k: v[idx] for k, v in self.img_paths_dict.items()}
        imgbatch_dict = {k: read_img(path, ".npy") for k, path in pathdict.items()}

        # Data augmentation for training
        if self.phase == 'train':
            imgbatch_dict = augment(imgbatch_dict)

        # Convert to tensor
        imgbatch_dict = {k: np2tensor(v, self.img_range) for k, v in imgbatch_dict.items()}

        # Add metadata
        imgbatch_dict["img_range"] = self.img_range

        return imgbatch_dict


class PanDataModule:
    """Data module for handling train/val/test dataloaders."""
    def __init__(self, train_root=None, val_root=None, test_root=None,
                 batch_size=4, num_workers=4, img_range=1023, lr_size=64):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_range = img_range
        self.lr_size = lr_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if train_root:
            self.train_dataset = PanDataset(train_root, 'train', img_range, lr_size)
        if val_root:
            self.val_dataset = PanDataset(val_root, 'val', img_range, lr_size)
        if test_root:
            self.test_dataset = PanDataset(test_root, 'test', img_range, lr_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    # Test dataset
    dataset = PanDataset('../../dataSet/NBU_GF1/train', phase='train')
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
