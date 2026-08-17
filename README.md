# NAFNet — Semiconductor Image Restoration

> **Hackathon submission** | Denoising + 2× Super-Resolution for semiconductor microscopy images.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

### 2. Download Model Weights

| File | Link | Size |
|------|------|------|
| `best_psnr_submit.pth` | [Google Drive / HuggingFace link] | ~60 MB |

Place the downloaded file in the `checkpoints/` folder:
```
checkpoints/
└── best_psnr_submit.pth
```

### 3. Run Inference

```bash
python evaluate.py \
  --input  /path/to/test/NoisyLR \
  --output ./outputs \
  --model  checkpoints/best_psnr_submit.pth
```

**That's it.** Restored images are written to `./outputs/` as 16-bit PNG files.

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` / `-i` | **required** | Directory of noisy/LR input images |
| `--output` / `-o` | **required** | Directory to write restored outputs |
| `--model` / `-m` | `checkpoints/best_psnr_submit.pth` | Path to checkpoint |
| `--gt` | None | Optional ground-truth dir for PSNR/SSIM/LPIPS reporting |
| `--device` | auto | `cuda` or `cpu` |

---

## Repository Structure

```
.
├── evaluate.py          # ← Standalone inference script (run this for evaluation)
├── train/
│   ├── train.py         # Training script
│   ├── model.py         # NAFNet architecture
│   ├── dataset.py       # Dataset / dataloader
│   └── overfit_test.py  # Sanity-check / overfit test
├── checkpoints/
│   └── best_psnr_submit.pth   # Trained model (EMA weights)
├── outputs/             # Model outputs on test set (pre-generated)
├── requirements.txt
└── README.md
```

---

## Reproduce Training From Scratch

### Dataset Structure

```
train/
├── NoisyLR/   # Noisy low-resolution inputs  (e.g. 256×256)
└── GT/        # Clean ground-truth outputs    (e.g. 512×512, 2× scale)
```

### Train

```bash
cd train
python train.py \
  --noisy-dir /path/to/NoisyLR \
  --gt-dir    /path/to/GT \
  --epochs    100
```

### Resume from checkpoint

```bash
python train.py \
  --resume checkpoints/latest.pth \
  --epochs 120
```

> ⚠️ Resume only works when `--resume` checkpoint uses the **same architecture** as `CFG` in `train.py`.  
> If you changed `enc_blocks` / `dec_blocks`, delete old checkpoints and train from scratch.

---

## Model

**NAFNet** (Nonlinear Activation Free Network) — UNet-style encoder-decoder.

| Setting | Value |
|---------|-------|
| Width | 32 |
| Encoder blocks | `[1, 1, 2, 4]` |
| Decoder blocks | `[1, 1, 1, 1]` |
| Middle blocks | 4 |
| Parameters | ~15.9M |
| Input | 1-channel (grayscale) noisy LR |
| Output | 1-channel (grayscale) restored HR (2× upscale) |

## Loss Function

| Component | Weight | Purpose |
|-----------|--------|---------|
| Charbonnier | 0.55 | Pixel-level fidelity |
| MS-SSIM | 0.15 | Structural similarity |
| FFT | 0.15 | High-frequency detail recovery |
| LPIPS (AlexNet) | 0.15 | Perceptual quality |

## Metrics (Validation Set)

| Metric | Value |
|--------|-------|
| PSNR | ~28.5 dB |
| SSIM | ~0.82 |
| LPIPS | ~0.18 |

---

## Requirements

```
torch>=2.0
torchvision
numpy
Pillow
tifffile
lpips
scikit-image
```

Full pinned requirements: see `requirements.txt`.

---

## Notes

- **Inference is single-pass** (no TTA). Metrics during validation match inference speed.
- **Submission checkpoint** (`best_psnr_submit.pth`) uses **EMA weights** — these match what was used during validation, not raw training weights.
- The `evaluate.py` script auto-reads architecture config from the checkpoint, so it works without manual edits.
