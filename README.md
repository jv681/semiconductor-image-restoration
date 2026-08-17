# NAFNet — Semiconductor Image Restoration

> **Hackathon submission** | Denoising + 2× Super-Resolution for semiconductor microscopy images.

---

## Quick Start (Inference)

### 1. Clone & Install

```bash
git clone https://github.com/jv681/semiconductor-image-restoration.git
cd semiconductor-image-restoration
pip install -r requirements.txt
```

### 2. Download Model Weights

| File | Link | Size |
|------|------|------|
| `best_psnr_submit.pth` | *(link coming after training)* | ~60 MB |

Place the downloaded file in the `train/checkpoints/` folder:
```
train/
└── checkpoints/
    └── best_psnr_submit.pth
```

### 3. Run Inference

```bash
python evaluate.py \
  --input  /path/to/test/NoisyLR \
  --output ./outputs \
  --model  train/checkpoints/best_psnr_submit.pth
```

**Restored images are written to `./outputs/` as 16-bit PNG files.**

For batched GPU inference (recommended on H100):
```bash
python evaluate.py \
  --input  /path/to/test/NoisyLR \
  --output ./outputs \
  --model  train/checkpoints/best_psnr_submit.pth \
  --batch  8
```

---

## Evaluate Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` / `-i` | **required** | Directory of noisy/LR input images |
| `--output` / `-o` | **required** | Directory to write restored output images |
| `--model` / `-m` | `checkpoints/best_psnr_submit.pth` | Path to checkpoint |
| `--batch` | `1` | Batch size (use 8+ on GPU for best throughput) |
| `--gt` | None | Optional ground-truth dir for PSNR/SSIM/LPIPS |
| `--device` | auto | `cuda` or `cpu` |

---

## Repository Structure

```
.
├── evaluate.py              # ← Standalone inference script (run this for evaluation)
├── requirements.txt         # pip freeze output
├── outputs/                 # Model outputs on test set (pre-generated)
├── README.md
└── train/
    ├── train.py             # Training script (reproduces training from scratch)
    ├── model.py             # NAFNet architecture
    ├── dataset.py           # Dataset / dataloader
    ├── overfit_test.py      # Sanity-check / overfit test
    └── checkpoints/         # Trained model weights (download separately)
        ├── best_psnr_submit.pth    # ← Use this for inference (EMA weights)
        └── best_lpips_submit.pth   # Alternative: best perceptual quality
```

---

## Reproduce Training From Scratch

### Dataset Structure

Place your data inside the `train/` folder:

```
train/
├── NoisyLR/   # Noisy low-resolution input images  (.npy format, e.g. 128×128)
└── GT/        # Clean ground-truth high-res images  (.npy format, e.g. 256×256)
```

Or pass paths explicitly via command-line arguments (see below).

### Train

```bash
cd train

# Option 1: Data in NoisyLR/ and GT/ folders next to train.py (uses defaults)
python train.py --epochs 80

# Option 2: Pass paths explicitly (works from any directory)
python train.py \
  --noisy-dir /path/to/NoisyLR \
  --gt-dir    /path/to/GT \
  --epochs    80
```

### Resume from checkpoint

```bash
cd train
python train.py \
  --resume checkpoints/best_psnr.pth \
  --epochs 100
```

> ⚠️ Resume only works when the checkpoint was saved with the **same architecture** as `CFG` in `train.py`. See architecture details below.

---

## Model

**NAFNet** (Nonlinear Activation Free Network) — UNet-style encoder-decoder with 2× pixel shuffle upsampling.

| Setting | Value |
|---------|-------|
| Width | 32 |
| Encoder blocks | `[1, 1, 2, 4]` |
| Decoder blocks | `[1, 1, 1, 1]` |
| Middle blocks | 4 |
| **Parameters** | **15.9M** (optimised for GPU throughput) |
| Input | 1-channel grayscale noisy LR (e.g. 128×128) |
| Output | 1-channel grayscale restored HR (2× upscale, e.g. 256×256) |

## Loss Function

| Component | Weight | Purpose |
|-----------|--------|---------|
| Charbonnier | 0.55 | Pixel-level fidelity |
| MS-SSIM | 0.15 | Structural similarity |
| FFT | 0.15 | High-frequency detail recovery |
| LPIPS (AlexNet) | 0.15 | Perceptual quality |

## Results (Validation Set, 80 epochs)

| Metric | Value |
|--------|-------|
| PSNR | 27.40 dB |
| SSIM | 0.7911 |
| LPIPS | 0.1612 |

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=5e-5, wd=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts |
| Batch size | 6 (grad accum ×2 = effective 12) |
| Patch size | 96×96 LR → 192×192 HR |
| EMA decay | 0.9995 |
| AMP | Enabled (CUDA) |
| TTA | Disabled (single-pass for throughput) |

---

## Notes

- **Inference is single-pass** (no TTA). Metrics during validation exactly match real inference speed.
- **Submission checkpoint** (`best_psnr_submit.pth`) uses **EMA weights** — these are what was used during validation, not raw training weights.
- `evaluate.py` **auto-reads architecture config** from the checkpoint, so it works without manual edits on any machine.
- Dataset files (NoisyLR/, GT/) are **not included** in this repo. Download or provide your own.
