"""
run.py  --  KLA Hackathon: AI-Based Restoration of Degraded Semiconductor Images
==================================================================================
Entry point for evaluation. Self-contained — no manual configuration needed.

Usage:
    python run.py <input-dir> <output-dir>

    <input-dir>  : directory containing noisy .npy files  (shape H x W, float32)
    <output-dir> : directory where restored .npy files will be written

Output:
    - One .npy file per input file, with the SAME filename
    - Shape: (H, W)  -- grayscale
    - Values: float32 in [0, 1], no NaN/Inf

Model:
    NAFNet (width=32, enc=[1,1,2,4], dec=[1,1,1,1], mid=4)  -- ~15.9M params
    Weights loaded from: models/best_psnr_submit.pth  (relative to this script)
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# NAFNet model — fully self-contained, no external imports
# ================================================================

class LayerNormCh(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Unflatten(1, (channels, 1, 1)),
        )

    def forward(self, x):
        return x * self.attn(x)


class NAFBlock(nn.Module):
    def __init__(self, channels, ffn_expand=2, dw_expand=1):
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand
        self.norm1 = LayerNormCh(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch * 2, 1)
        self.dw    = nn.Conv2d(dw_ch * 2, dw_ch * 2, 3, 1, 1, groups=dw_ch * 2)
        self.gate1 = SimpleGate()
        self.sca   = SimpleChannelAttention(dw_ch)
        self.proj1 = nn.Conv2d(dw_ch, channels, 1)
        self.norm2 = LayerNormCh(channels)
        self.ffn1  = nn.Conv2d(channels, ffn_ch * 2, 1)
        self.gate2 = SimpleGate()
        self.ffn2  = nn.Conv2d(ffn_ch, channels, 1)
        self.beta  = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y); y = self.dw(y); y = self.gate1(y)
        y = self.sca(y);   y = self.proj1(y)
        x = x + y * self.beta
        y = self.norm2(x)
        y = self.ffn1(y);  y = self.gate2(y); y = self.ffn2(y)
        x = x + y * self.gamma
        return x


class NAFNet(nn.Module):
    def __init__(self, in_ch=1, width=32,
                 enc_blocks=None, dec_blocks=None, middle_blocks=4):
        super().__init__()
        if enc_blocks is None: enc_blocks = [1, 1, 2, 4]
        if dec_blocks is None: dec_blocks = [1, 1, 1, 1]

        self.intro = nn.Conv2d(in_ch, width, 3, 1, 1)
        self.encoders    = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = width
        enc_channels = []
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            enc_channels.append(ch)
            self.downsamples.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blocks)])

        self.upsamples = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        for n, skip_ch in zip(dec_blocks, reversed(enc_channels)):
            self.upsamples.append(nn.Sequential(
                nn.Conv2d(ch, skip_ch * 4, 1), nn.PixelShuffle(2)
            ))
            ch = skip_ch
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self.tail = nn.Sequential(
            NAFBlock(ch),
            nn.Conv2d(ch, in_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.intro(x)
        enc_feats = []
        for enc, down in zip(self.encoders, self.downsamples):
            x = enc(x); enc_feats.append(x); x = down(x)
        x = self.middle(x)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(enc_feats)):
            x = up(x); x = x + skip; x = dec(x)
        x = self.tail(x)
        return torch.clamp(x, 0.0, 1.0)


# ================================================================
# Helpers
# ================================================================

def load_model(weights_path: str, device: torch.device) -> NAFNet:
    """Load NAFNet from checkpoint. Auto-reads architecture from cfg_snapshot."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    cfg = ckpt.get("cfg_snapshot", {})
    width      = cfg.get("width",      32)
    enc_blocks = cfg.get("enc_blocks", [1, 1, 2, 4])
    dec_blocks = cfg.get("dec_blocks", [1, 1, 1, 1])
    mid_blocks = cfg.get("mid_blocks", 4)

    model = NAFNet(in_ch=1, width=width,
                   enc_blocks=enc_blocks,
                   dec_blocks=dec_blocks,
                   middle_blocks=mid_blocks).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"[run.py] Model loaded: {n/1e6:.2f}M params  "
          f"enc={enc_blocks} dec={dec_blocks} mid={mid_blocks}")
    return model


def npy_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a raw .npy array to a (1, 1, H, W) float32 tensor in [0, 1].
    Handles shapes: (H, W), (H, W, 1), (H, W, C).
    """
    if arr.ndim == 3:
        arr = arr[..., 0]           # take first channel
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def tensor_to_npy(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 1, H, W) or (1, H, W) output tensor to (H, W) float32 array.
    Guaranteed [0, 1], no NaN/Inf.
    """
    arr = t.squeeze().cpu().float().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr.astype(np.float32)


# ================================================================
# Main
# ================================================================

def main():
    # --- Parse arguments (positional as required by spec) -----------
    parser = argparse.ArgumentParser(
        description="NAFNet inference: restore degraded semiconductor images"
    )
    parser.add_argument("input_dir",  type=str,
                        help="Directory containing noisy .npy input files")
    parser.add_argument("output_dir", type=str,
                        help="Directory to write restored .npy output files")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (default: models/best_psnr_submit.pth)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for GPU inference (default: 4)")
    args = parser.parse_args()

    # --- Paths ------------------------------------------------------
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Auto-locate model weights next to this script
    script_dir = Path(__file__).resolve().parent
    if args.model:
        weights_path = Path(args.model)
    else:
        weights_path = script_dir / "models" / "best_psnr_submit.pth"

    # Validate inputs
    if not input_dir.exists():
        print(f"[Error] Input directory not found: {input_dir}")
        sys.exit(1)
    if not weights_path.exists():
        print(f"[Error] Model weights not found: {weights_path}")
        print(f"        Place best_psnr_submit.pth in: {script_dir / 'models'}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run.py] Device      : {device}")
    print(f"[run.py] Input dir   : {input_dir}")
    print(f"[run.py] Output dir  : {output_dir}")
    print(f"[run.py] Weights     : {weights_path}")

    # --- Load model --------------------------------------------------
    model = load_model(str(weights_path), device)

    # --- Find .npy input files ---------------------------------------
    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"[Error] No .npy files found in {input_dir}")
        sys.exit(1)
    print(f"[run.py] Files found : {len(npy_files)}")
    print(f"[run.py] Batch size  : {args.batch}")

    # --- Batched inference -------------------------------------------
    t_start = time.perf_counter()
    batch_size = args.batch
    batches = [npy_files[i:i+batch_size] for i in range(0, len(npy_files), batch_size)]

    processed = 0
    with torch.no_grad():
        for batch_paths in batches:
            # Load batch
            tensors = [npy_to_tensor(np.load(str(p)), device) for p in batch_paths]

            # Stack (assumes same spatial size — standard for this dataset)
            try:
                batch_inp = torch.cat(tensors, dim=0)   # (B, 1, H, W)
                batch_out = model(batch_inp)             # single GPU forward pass
            except RuntimeError:
                # Fallback: different sizes — process individually
                batch_out = torch.cat([model(t) for t in tensors], dim=0)

            # Save each output as .npy with same filename as input
            for i, src_path in enumerate(batch_paths):
                out_arr  = tensor_to_npy(batch_out[i:i+1])   # (H, W) float32
                out_path = output_dir / src_path.name         # SAME filename
                np.save(str(out_path), out_arr)
                processed += 1

    elapsed = time.perf_counter() - t_start
    print(f"\n[run.py] Done.")
    print(f"  Processed : {processed} files")
    print(f"  Total time: {elapsed:.1f}s  ({elapsed/processed*1000:.1f} ms/image)")
    print(f"  Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
