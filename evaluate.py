"""
evaluate.py  --  NAFNet Inference Script
=========================================
Semiconductor Image Restoration: Denoising + 2x Super-Resolution

Usage:
    python evaluate.py --input /path/to/test/NoisyLR --output /path/to/outputs
    python evaluate.py --input ./Test_NoisyLR --output ./outputs --model checkpoints/best_psnr_submit.pth

This script:
  1. Loads the trained NAFNet model (EMA weights from submission checkpoint)
  2. Runs single-pass inference on every image in the input directory
  3. Writes restored outputs to the output directory
  4. Prints PSNR/SSIM/LPIPS if ground-truth images are available alongside

Supported input formats: .png, .tif, .tiff, .jpg, .jpeg, .bmp
Output format: .png (lossless)

No manual edits required. Run as-is on any machine after installing requirements.
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ================================================================
# NAFNet model definition (self-contained, no imports from train.py)
# ================================================================

class LayerNorm2d(torch.nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(channels))
        self.bias   = torch.nn.Parameter(torch.zeros(channels))
        self.eps    = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(torch.nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(torch.nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch  = c * dw_expand
        ffn_ch = c * ffn_expand
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.conv1 = torch.nn.Conv2d(c, dw_ch, 1)
        self.dw    = torch.nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sca   = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(dw_ch // 2, dw_ch // 2, 1)
        )
        self.proj1 = torch.nn.Conv2d(dw_ch // 2, c, 1)
        self.ffn1  = torch.nn.Conv2d(c, ffn_ch, 1)
        self.ffn2  = torch.nn.Conv2d(ffn_ch // 2, c, 1)
        self.gate  = SimpleGate()
        self.beta  = torch.nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = torch.nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.dw(x)
        x = self.gate(x)
        x = x * self.sca(x)
        x = self.proj1(x)
        y = inp + x * self.beta
        x = self.ffn2(self.gate(self.ffn1(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(torch.nn.Module):
    def __init__(self, in_ch=1, width=32,
                 enc_blocks=None, dec_blocks=None, middle_blocks=4):
        super().__init__()
        if enc_blocks is None: enc_blocks = [1, 1, 2, 4]
        if dec_blocks is None: dec_blocks = [1, 1, 1, 1]

        self.intro   = torch.nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending  = torch.nn.Conv2d(width, in_ch * 4, 3, padding=1)
        self.upsample = torch.nn.PixelShuffle(2)

        self.encoders, self.downs, self.ups, self.decoders = \
            torch.nn.ModuleList(), torch.nn.ModuleList(), \
            torch.nn.ModuleList(), torch.nn.ModuleList()

        chan = width
        for n in enc_blocks:
            self.encoders.append(torch.nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(torch.nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle = torch.nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blocks)])

        for n in dec_blocks:
            self.ups.append(torch.nn.Sequential(
                torch.nn.Conv2d(chan, chan * 2, 1),
                torch.nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(torch.nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))

        self.pad_size = 2 ** len(enc_blocks)

    def forward(self, x):
        _, _, H, W = x.shape
        ph = (self.pad_size - H % self.pad_size) % self.pad_size
        pw = (self.pad_size - W % self.pad_size) % self.pad_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')

        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.middle(x)
        for dec, up in zip(self.decoders, self.ups):
            x = up(x) + skips.pop(); x = dec(x)

        x = self.upsample(self.ending(x))
        if ph or pw:
            x = x[:, :, :H*2, :W*2]
        return x


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ================================================================
# Metrics
# ================================================================
def psnr(pred, gt):
    mse = F.mse_loss(pred.float(), gt.float()).item()
    return 100.0 if mse < 1e-10 else 20 * math.log10(1.0) - 10 * math.log10(mse)

def ssim_metric(pred, gt):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(pred, 11, 1, 5)
    mu2 = F.avg_pool2d(gt,   11, 1, 5)
    m1s, m2s, m12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1  = F.avg_pool2d(pred*pred, 11, 1, 5) - m1s
    s2  = F.avg_pool2d(gt*gt,     11, 1, 5) - m2s
    s12 = F.avg_pool2d(pred*gt,   11, 1, 5) - m12
    return (((2*m12+C1)*(2*s12+C2)) / ((m1s+m2s+C1)*(s1+s2+C2))).mean().item()


# ================================================================
# Image I/O
# ================================================================
def load_image(path: Path, device) -> torch.Tensor:
    """Load grayscale image -> [1,1,H,W] float32 tensor in [0,1]."""
    try:
        import tifffile
        arr = tifffile.imread(str(path)).astype(np.float32)
    except (ImportError, Exception):
        try:
            from PIL import Image
            img = Image.open(str(path)).convert('L')
            arr = np.array(img, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Cannot read {path}: {e}")

    if arr.ndim == 3:
        arr = arr[..., 0]  # take first channel if multi-channel
    # Normalize to [0,1]
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def save_image(tensor: torch.Tensor, path: Path):
    """Save [1,1,H,W] or [1,H,W] float32 tensor in [0,1] as PNG."""
    from PIL import Image
    arr = tensor.squeeze().cpu().float().clamp(0, 1).numpy()
    arr = (arr * 65535).astype(np.uint16)
    img = Image.fromarray(arr, mode='I;16')
    img.save(str(path))


# ================================================================
# Main inference
# ================================================================
IMG_EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}

def main():
    parser = argparse.ArgumentParser(
        description='NAFNet inference — semiconductor image restoration'
    )
    parser.add_argument('--input',  '-i', type=str, required=True,
                        help='Directory of noisy/LR input images')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory to write restored output images')
    parser.add_argument('--model',  '-m', type=str,
                        default='checkpoints/best_psnr_submit.pth',
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--gt',     type=str, default=None,
                        help='Optional: ground-truth directory for PSNR/SSIM/LPIPS')
    parser.add_argument('--batch',  type=int, default=1,
                        help='Batch size for GPU inference (default 1 for variable sizes)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda / cpu (auto-detected if not set)')
    args = parser.parse_args()

    # ---- Device -------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Eval] Device : {device}")

    # ---- Load checkpoint ----------------------------------------------
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        # Try fallback checkpoints in order
        fallbacks = [
            'checkpoints/best_psnr_submit.pth',
            'checkpoints/best_psnr.pth',
            'checkpoints/best_lpips_submit.pth',
            'checkpoints/latest.pth',
        ]
        for fb in fallbacks:
            if Path(fb).exists():
                ckpt_path = Path(fb)
                print(f"[Eval] Checkpoint not found at {args.model}, using {fb}")
                break
        else:
            print(f"[Error] No checkpoint found. Tried: {args.model} and fallbacks.")
            print("        Please provide --model /path/to/checkpoint.pth")
            sys.exit(1)

    print(f"[Eval] Model  : {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Read architecture from checkpoint config snapshot if available
    cfg = ckpt.get('cfg_snapshot', {})
    width      = cfg.get('width',      32)
    enc_blocks = cfg.get('enc_blocks', [1, 1, 2, 4])
    dec_blocks = cfg.get('dec_blocks', [1, 1, 1, 1])
    mid_blocks = cfg.get('mid_blocks', 4)

    model = NAFNet(in_ch=1, width=width,
                   enc_blocks=enc_blocks,
                   dec_blocks=dec_blocks,
                   middle_blocks=mid_blocks).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    n_params = count_params(model)
    print(f"[Eval] NAFNet params: {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"[Eval] Architecture: enc={enc_blocks} dec={dec_blocks} mid={mid_blocks}")
    note = ckpt.get('note', '')
    if note:
        print(f"[Eval] Note: {note}")

    # ---- Find input images --------------------------------------------
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])
    if not image_paths:
        print(f"[Error] No images found in {input_dir}")
        sys.exit(1)
    print(f"[Eval] Found {len(image_paths)} images in {input_dir}")

    # ---- Optional GT for metric computation ---------------------------
    gt_dir = Path(args.gt) if args.gt else None
    lpips_fn = None
    if gt_dir:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
            lpips_fn.eval()
        except ImportError:
            print("[Eval] lpips not installed; skipping LPIPS metric")

    # ---- Inference loop -----------------------------------------------
    total_psnr = total_ssim = total_lpips = 0.0
    n_metric = 0
    total_time = 0.0

    print(f"\n{'Image':<40} {'Time(ms)':>9} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7}")
    print('-' * 74)

    with torch.no_grad():
        for img_path in image_paths:
            # Load
            inp = load_image(img_path, device)

            # Inference (single forward pass — no TTA)
            t0   = time.perf_counter()
            pred = model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_time += elapsed_ms

            pred_np = pred.clamp(0, 1)

            # Metrics vs GT
            p_str = s_str = l_str = '  -'
            if gt_dir:
                gt_path = gt_dir / img_path.name
                # Try common GT naming patterns
                if not gt_path.exists():
                    stem = img_path.stem
                    for suf in IMG_EXTS:
                        candidate = gt_dir / (stem + suf)
                        if candidate.exists():
                            gt_path = candidate
                            break
                if gt_path.exists():
                    gt = load_image(gt_path, device)
                    gt_f = gt.float()
                    p = psnr(pred_np, gt_f)
                    s = ssim_metric(pred_np.float(), gt_f)
                    total_psnr += p; total_ssim += s; n_metric += 1
                    p_str = f'{p:8.2f}'
                    s_str = f'{s:7.4f}'
                    if lpips_fn is not None:
                        p3 = pred_np.repeat(1, 3, 1, 1) * 2 - 1
                        g3 = gt_f.repeat(1,  3, 1, 1) * 2 - 1
                        lv = lpips_fn(p3, g3).mean().item()
                        total_lpips += lv
                        l_str = f'{lv:7.4f}'

            # Save output
            out_name = img_path.stem + '_restored.png'
            out_path = output_dir / out_name
            save_image(pred_np, out_path)

            print(f'{img_path.name:<40} {elapsed_ms:9.1f} {p_str} {s_str} {l_str}')

    # ---- Summary ------------------------------------------------------
    n = len(image_paths)
    print('-' * 74)
    print(f"\n[Results]")
    print(f"  Images processed : {n}")
    print(f"  Total time       : {total_time:.1f} ms")
    print(f"  Avg time/image   : {total_time/n:.1f} ms  ({1000/(total_time/n):.1f} img/s)")
    if n_metric > 0:
        print(f"  Avg PSNR         : {total_psnr/n_metric:.4f} dB")
        print(f"  Avg SSIM         : {total_ssim/n_metric:.4f}")
        if lpips_fn is not None:
            print(f"  Avg LPIPS        : {total_lpips/n_metric:.4f}")
    print(f"  Output dir       : {output_dir.resolve()}")
    print(f"\nDone.")


if __name__ == '__main__':
    main()
