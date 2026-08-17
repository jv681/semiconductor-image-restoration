"""
dataset.py  --  Semiconductor Denoising + 2x SR Dataset
Supports patch-based training for fast convergence.

NoisyLR: (128, 128) float32  ->  patches of (patch_size, patch_size)
GT:      (256, 256) float32  ->  patches of (patch_size*2, patch_size*2)
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir, file_list=None,
                 patch_size=None, augment=False):
        """
        Args:
            patch_size : LR patch size (HR patch = patch_size * 2).
                         If None, returns full images.
            augment    : random flips + 90-deg rotations
        """
        self.noisy_dir  = noisy_dir
        self.gt_dir     = gt_dir
        self.patch_size = patch_size
        self.augment    = augment

        if file_list is not None:
            self.stems = file_list
        else:
            files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))
            self.stems = [os.path.splitext(os.path.basename(f))[0] for f in files]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem  = self.stems[idx]
        noisy = np.load(os.path.join(self.noisy_dir, stem + ".npy")).astype(np.float32)
        gt    = np.load(os.path.join(self.gt_dir,    stem + ".npy")).astype(np.float32)

        # Clip to [0, 1]
        noisy = np.clip(noisy, 0.0, 1.0)
        gt    = np.clip(gt,    0.0, 1.0)

        # Random patch crop (consistent for LR and HR)
        if self.patch_size is not None:
            ps     = self.patch_size
            lr_h, lr_w = noisy.shape
            top  = np.random.randint(0, lr_h - ps + 1)
            left = np.random.randint(0, lr_w - ps + 1)
            noisy = noisy[top:top+ps, left:left+ps]
            # HR crop is exactly 2x the LR crop
            gt = gt[top*2:(top+ps)*2, left*2:(left+ps)*2]

        # (H, W) -> (1, H, W)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt    = torch.from_numpy(gt).unsqueeze(0)

        # Augmentation: random flips + 90-deg rotations
        if self.augment:
            # Horizontal flip
            if torch.rand(1).item() > 0.5:
                noisy = torch.flip(noisy, [-1])
                gt    = torch.flip(gt,    [-1])
            # Vertical flip
            if torch.rand(1).item() > 0.5:
                noisy = torch.flip(noisy, [-2])
                gt    = torch.flip(gt,    [-2])
            # Random 90-deg rotation (0, 90, 180, 270)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                noisy = torch.rot90(noisy, k, [-2, -1])
                gt    = torch.rot90(gt,    k, [-2, -1])

        return noisy, gt


def make_dataloaders(noisy_dir, gt_dir, val_split=0.1,
                     batch_size=16, num_workers=2,
                     patch_size=64, seed=42):
    all_files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))
    stems     = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(len(stems))
    n_val   = max(1, int(len(stems) * val_split))

    val_stems   = [stems[i] for i in idx[:n_val]]
    train_stems = [stems[i] for i in idx[n_val:]]

    train_ds = SemiconductorDataset(noisy_dir, gt_dir, train_stems,
                                    patch_size=patch_size, augment=True)
    val_ds   = SemiconductorDataset(noisy_dir, gt_dir, val_stems,
                                    patch_size=None, augment=False)  # full image for val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Patch LR={patch_size}x{patch_size} HR={patch_size*2}x{patch_size*2}")
    return train_loader, val_loader


if __name__ == "__main__":
    noisy_dir = r"C:\Users\hasin\Downloads\train\train\NoisyLR"
    gt_dir    = r"C:\Users\hasin\Downloads\train\train\GT"
    tl, vl = make_dataloaders(noisy_dir, gt_dir, batch_size=4, patch_size=64)
    n, g = next(iter(tl))
    print("LR patch:", n.shape, "| HR patch:", g.shape)
    n2, g2 = next(iter(vl))
    print("Val LR full:", n2.shape, "| Val HR full:", g2.shape)