"""
overfit_test.py
Sanity check: overfit NAFNet on a single image for 500 steps.
If loss drops to near 0, the pipeline is correct.
Saves a side-by-side comparison: noisy_lr | pred_hr | gt_hr
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

from model import NAFNet


NOISY_PATH = r"C:\Users\hasin\Downloads\train\train\NoisyLR\000000.npy"
GT_PATH    = r"C:\Users\hasin\Downloads\train\train\GT\000000.npy"
STEPS      = 500
LR         = 1e-3
SAVE_PATH  = "overfit_result.png"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load single sample
    noisy = np.clip(np.load(NOISY_PATH).astype(np.float32), 0, 1)
    gt    = np.clip(np.load(GT_PATH).astype(np.float32),    0, 1)

    noisy_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,128)
    gt_t    = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)     # (1,1,256,256)

    # Smaller model for fast sanity check
    model = NAFNet(in_ch=1, width=16,
                   enc_blocks=[1, 1, 1, 2],
                   dec_blocks=[1, 1, 1, 1],
                   middle_blocks=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    print(f"\nOverfitting on 1 image for {STEPS} steps...")
    print(f"{'Step':>6} | {'L1 Loss':>10} | {'PSNR':>8}")
    print("-" * 34)

    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            pred = model(noisy_t)
            loss = criterion(pred, gt_t)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                mse  = torch.mean((pred - gt_t) ** 2).item()
                psnr = 20 * np.log10(1.0) - 10 * np.log10(max(mse, 1e-10))
            print(f"{step:6d} | {loss.item():10.6f} | {psnr:8.2f} dB")

    # ---- Save comparison image ----
    model.eval()
    with torch.no_grad():
        pred = model(noisy_t).squeeze().cpu().numpy()

    noisy_up = np.repeat(np.repeat(noisy, 2, axis=0), 2, axis=1)  # 128->256 for display
    gt_np    = gt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(noisy_up, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Noisy LR (upscaled 2x)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("NAFNet Prediction (256x256)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth (256x256)", fontsize=13)
    axes[2].axis("off")

    plt.suptitle("NAFNet Overfit Test - Sanity Check", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison to: {SAVE_PATH}")

    # Final verdict
    final_loss = loss.item()
    if final_loss < 0.01:
        print(f"PASS: Loss {final_loss:.6f} < 0.01 - pipeline is correct!")
    else:
        print(f"WARNING: Loss {final_loss:.6f} did not converge as expected.")


if __name__ == "__main__":
    main()