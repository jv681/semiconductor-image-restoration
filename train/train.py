"""
train.py  --  NAFNet: Semiconductor Denoising + 2x SR
=================================================================
Hackathon-optimised version.

Model  : NAFNet (width=32, ~15.9M params, lighter enc/dec for throughput)
           enc=[1,1,2,4]  dec=[1,1,1,1]  mid=4
           Previous architecture was enc=[2,2,4,8] dec=[2,2,2,2] (~19.8M).
Loss   : Charbonnier(0.55) + MS-SSIM(0.15) + FFT(0.15) + LPIPS-Alex(0.15)
Extras : EMA, AMP, GradAccum, GradClip, CosineWarmRestarts

Key changes vs previous version:
  1. TTA disabled  -- single-pass inference matches hackathon scoring
  2. LPIPS train backbone: VGG -> AlexNet  (same quality, faster steps)
  3. Lighter architecture  -- better GPU throughput for batch inference
  4. best_psnr_submit.pth saves EMA weights as primary 'model' key
  5. Resume now restores optimizer + scheduler state (true resume)
  6. Loss-weight comment clarified: defaults, not guaranteed results

What changed (hygiene only):
  1. Paths configurable via argparse (portable to any machine)
  2. LPIPS backbone configurable; logged clearly; hard error if missing
  3. TTA configurable flag; labeled in validation output
  4. DataLoader workers seeded (reproducibility)
  5. All loss components logged at epoch level
  6. Validation reports EMA/TTA status explicitly
  7. best_lpips correctly restored on resume
  8. Reproducibility mode flag (benchmark vs deterministic)
  9. Config snapshot printed at startup

Usage:
    python train.py --resume checkpoints/best.pth --epochs 100
    python train.py --noisy-dir /path/to/NoisyLR --gt-dir /path/to/GT
    python train.py --no-tta          # disable TTA for faster validation
    python train.py --repro           # fully deterministic (slower)
"""

import os, sys, math, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from copy import deepcopy

from dataset import make_dataloaders
from model import NAFNet, count_params


# ================================================================
# Config  --  ALL hyperparameters in one place.
# Paths can be overridden via argparse below.
# ================================================================
CFG = {
    # --- Paths (overridable via --noisy-dir, --gt-dir, --ckpt-dir) ---
    # Defaults use relative paths so the script works on any machine.
    # Place data in NoisyLR/ and GT/ folders next to train.py,
    # OR pass --noisy-dir and --gt-dir arguments on the command line.
    "noisy_dir"  : "NoisyLR",
    "gt_dir"     : "GT",
    "val_split"  : 0.1,
    "ckpt_dir"   : "checkpoints",

    # --- Data ---
    "patch_size" : 96,          # LR patch; HR = 96*2 = 192
    "batch_size" : 6,
    "num_workers": 2,
    "grad_accum" : 2,           # effective batch = 12

    # --- NAFNet architecture (LIGHTER for hackathon throughput) ---
    # Verified param counts (width=32):
    #   enc=[2,2,4,8] dec=[2,2,2,2] mid=4  -> 19.84M  (original, too heavy)
    #   enc=[1,1,2,4] dec=[1,1,1,1] mid=6  -> 21.18M  (LARGER, do not use)
    #   enc=[1,1,2,4] dec=[1,1,1,1] mid=4  -> 15.90M  <-- current (verified)
    # IMPORTANT: changing these values requires deleting old checkpoints and
    # training from scratch. --resume only works if architecture matches.
    "width"      : 32,
    "enc_blocks" : [1, 1, 2, 4],
    "dec_blocks" : [1, 1, 1, 1],
    "mid_blocks" : 4,

    # --- Training schedule ---
    "epochs"     : 100,
    "lr"         : 5e-5,
    "min_lr"     : 1e-7,
    "weight_decay": 1e-4,
    "grad_clip"  : 0.5,

    # --- Loss weights ---
    # Note: different losses operate at different numerical scales;
    # equal coefficients do NOT mean equal optimization influence.
    # Change 6: these are DEFAULT starting values for this dataset.
    # Actual results depend on your data, architecture, and training run.
    # Validate empirically after training; adjust via controlled ablation.
    "charb_w"    : 0.55,
    "ssim_w"     : 0.15,
    "fft_w"      : 0.15,
    "lpips_w"    : 0.15,
    "fft_phase_w": 0.10,        # coefficient on phase term inside FFT loss

    # --- LPIPS backbones ---
    # Change 2: train backbone switched VGG -> AlexNet.
    # AlexNet is ~3x faster per forward pass than VGG with minimal
    # quality difference as a perceptual training signal. This speeds
    # up every training step meaningfully on CPU.
    # Both train and eval now use AlexNet -> consistent methodology.
    "train_lpips_net": "alex",  # frozen AlexNet as training loss (fast)
    "eval_lpips_net" : "alex",  # AlexNet as evaluation metric (consistent)

    # --- EMA ---
    "ema_decay"  : 0.9995,      # higher = slower update, more stable

    # --- Validation TTA ---
    # Change 1: TTA DISABLED for hackathon.
    # TTA runs 8 forward passes per image -> 8x slower inference.
    # Judges score batched GPU throughput, so TTA hurts the submission.
    # Validation metrics now reflect true single-pass inference speed.
    "use_tta"    : False,

    # --- Reproducibility ---
    "seed"       : 42,
    # repro_mode=False: cudnn.benchmark=True  (faster, slightly non-deterministic)
    # repro_mode=True : cudnn.deterministic=True (slower, fully reproducible)
    "repro_mode" : False,

    # --- Logging ---
    "log_every"  : 20,          # log every N batches
}


# ================================================================
# Reproducibility
# ================================================================
def set_seed(seed: int, repro_mode: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if repro_mode:
        # Fully deterministic — may be slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        # Faster; small non-determinism from cuDNN autotuning
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True


def worker_init_fn(worker_id: int):
    """Seed each DataLoader worker independently for reproducibility."""
    seed = CFG["seed"] + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================================================================
# Losses  (UNCHANGED from baseline)
# ================================================================

class CharbonnierLoss(nn.Module):
    """sqrt((x-y)^2 + eps^2) -- smoother L1, better near zero."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SSIMLoss(nn.Module):
    """Multi-scale SSIM: original + 2x downsampled."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        c = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        k = (g.unsqueeze(0) * g.unsqueeze(1))
        k = (k / k.sum()).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", k)
        self.ws = window_size

    def _ssim(self, pred, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        p = self.ws // 2
        ch = pred.shape[1]
        k  = self.kernel.expand(ch, 1, -1, -1)
        mu1  = F.conv2d(pred,   k, padding=p, groups=ch)
        mu2  = F.conv2d(target, k, padding=p, groups=ch)
        m1s, m2s, m12 = mu1*mu1, mu2*mu2, mu1*mu2
        s1  = F.conv2d(pred*pred,     k, padding=p, groups=ch) - m1s
        s2  = F.conv2d(target*target, k, padding=p, groups=ch) - m2s
        s12 = F.conv2d(pred*target,   k, padding=p, groups=ch) - m12
        return (((2*m12+C1)*(2*s12+C2)) / ((m1s+m2s+C1)*(s1+s2+C2))).mean()

    def forward(self, pred, target):
        loss  = 1.0 - self._ssim(pred, target)
        p2 = F.avg_pool2d(pred,   2)
        t2 = F.avg_pool2d(target, 2)
        loss += 0.5 * (1.0 - self._ssim(p2, t2))
        return loss / 1.5


class FFTLoss(nn.Module):
    """
    Frequency domain loss on amplitude + phase of rfft2.
    Numerically stable: rfft2 with norm='ortho' is bounded;
    angle() is computed on complex tensors (no explicit atan2 issues).
    Phase coefficient is configurable to allow ablation.
    """
    def __init__(self, phase_w: float = 0.10):
        super().__init__()
        self.phase_w = phase_w

    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred,   norm="ortho")
        tf = torch.fft.rfft2(target, norm="ortho")
        amp_loss   = F.l1_loss(pf.abs(),   tf.abs())
        phase_loss = F.l1_loss(pf.angle(), tf.angle())
        return amp_loss + self.phase_w * phase_loss


class PerceptualLPIPSLoss(nn.Module):
    """
    LPIPS as TRAINING LOSS with a configurable frozen backbone.
    Backbone is set via CFG['train_lpips_net'] (currently: AlexNet).
    Only NAFNet weights are updated; the LPIPS backbone is never trained.
    Input range: [0,1] grayscale -> internally converted to [-1,1] RGB.

    Hard error if lpips not installed and lpips_w > 0.
    (Silent skip risks accidentally training without perceptual loss.)
    """
    def __init__(self, net: str = "vgg"):
        super().__init__()
        self.net_name = net
        try:
            import lpips as _lpips
            self.fn = _lpips.LPIPS(net=net, verbose=False)
            for p in self.fn.parameters():
                p.requires_grad_(False)
            self.available = True
        except ImportError:
            raise ImportError(
                "\n[LPIPS-Loss] lpips package not found.\n"
                "  Install with:  pip install lpips\n"
                "  Or disable:    set lpips_w=0 in CFG."
            )

    def to(self, device):
        super().to(device)
        self.fn = self.fn.to(device)
        return self

    def forward(self, pred, target):
        # [0,1] gray -> [-1,1] RGB (repeat channel)
        p = pred.repeat(1, 3, 1, 1)   * 2.0 - 1.0
        t = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.fn(p, t).mean()


class CombinedLoss(nn.Module):
    def __init__(self, cw: float, sw: float, fw: float, lw: float,
                 fft_phase_w: float, train_net: str):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim  = SSIMLoss()
        self.fft   = FFTLoss(phase_w=fft_phase_w)
        self.perc  = PerceptualLPIPSLoss(net=train_net) if lw > 0 else None
        self.cw, self.sw, self.fw, self.lw = cw, sw, fw, lw

    def to(self, device):
        super().to(device)
        if self.perc is not None:
            self.perc = self.perc.to(device)
        return self

    def forward(self, pred, target):
        c = self.charb(pred, target)
        s = self.ssim(pred,  target)
        f = self.fft(pred,   target)
        l = self.perc(pred, target) if self.perc is not None \
            else torch.zeros(1, device=pred.device)
        total = self.cw*c + self.sw*s + self.fw*f + self.lw*l
        return total, c.item(), s.item(), f.item(), l.item()


# ================================================================
# Validation metrics
# ================================================================
def compute_psnr(pred, target):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return 100.0 if mse < 1e-10 else 20*math.log10(1.0) - 10*math.log10(mse)

def compute_ssim(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(pred,   11, 1, 5)
    mu2 = F.avg_pool2d(target, 11, 1, 5)
    m1s, m2s, m12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1  = F.avg_pool2d(pred*pred,     11, 1, 5) - m1s
    s2  = F.avg_pool2d(target*target, 11, 1, 5) - m2s
    s12 = F.avg_pool2d(pred*target,   11, 1, 5) - m12
    return (((2*m12+C1)*(2*s12+C2)) / ((m1s+m2s+C1)*(s1+s2+C2))).mean().item()

def load_lpips_metric(net: str = "alex"):
    """AlexNet LPIPS for evaluation (faster than VGG, good perceptual correlation)."""
    try:
        import lpips
        fn = lpips.LPIPS(net=net, verbose=False)
        fn.eval()
        return fn
    except ImportError:
        print(f"[Warning] lpips not installed; LPIPS eval metric disabled.")
        return None


# ================================================================
# TTA: 8-way geometric self-ensemble
# 4 rotations x 2 flips -- inverse transform applied to each prediction.
# Gives +0.3-0.5 dB PSNR at inference with zero extra training.
# ================================================================
@torch.no_grad()
def tta_predict(model, x):
    preds = []
    for k in range(4):
        xr = torch.rot90(x, k, [-2, -1])
        preds.append(torch.rot90(model(xr), 4-k, [-2, -1]))
        xf = torch.flip(xr, [-1])
        preds.append(torch.rot90(torch.flip(model(xf), [-1]), 4-k, [-2, -1]))
    return torch.stack(preds, dim=0).mean(dim=0)


# ================================================================
# EMA
# ================================================================
class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Validation always uses EMA model for more stable metrics.
    Original training weights are separate and unaffected.
    """
    def __init__(self, model, decay: float = 0.9995):
        self.decay = decay
        self.model = deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.model.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):         return self.model.state_dict()
    def load_state_dict(self, d): self.model.load_state_dict(d)


# ================================================================
# Train one epoch
# ================================================================
def train_epoch(model, ema, loader, optimizer, scheduler,
                scaler, criterion, device, epoch, cfg):
    model.train()
    rl = rc = rs = rf = rp = 0.0
    t0    = time.time()
    accum = cfg["grad_accum"]
    dtype = "cuda" if device.type == "cuda" else "cpu"

    optimizer.zero_grad(set_to_none=True)
    for i, (noisy, gt) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True)
        gt    = gt.to(device,    non_blocking=True)

        with autocast(device_type=dtype, enabled=(device.type == "cuda")):
            pred = model(noisy)
            loss, c, s, f, p = criterion(pred, gt)
            loss = loss / accum            # divide before backward

        scaler.scale(loss).backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            # unscale before clip so clip operates on true gradient magnitudes
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()              # step per optimizer step (not per epoch)
            ema.update(model)

        rl += loss.item() * accum         # restore pre-division scale for logging
        rc += c; rs += s; rf += f; rp += p

        if (i + 1) % cfg["log_every"] == 0:
            n  = i + 1
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [E{epoch:03d}|B{n:04d}/{len(loader)}] "
                  f"total={rl/n:.4f}  charb={rc/n:.4f}  "
                  f"ssim={rs/n:.4f}  fft={rf/n:.4f}  perc={rp/n:.4f}  "
                  f"lr={lr:.2e}  t={time.time()-t0:.0f}s")

    n = len(loader)
    return rl/n, rc/n, rs/n, rf/n, rp/n


# ================================================================
# Validate
# ================================================================
@torch.no_grad()
def validate(model, loader, criterion, device,
             lpips_metric_fn=None, use_tta=True):
    model.eval()
    tl = tp = ts = tlp = 0.0
    n  = 0
    dtype = "cuda" if device.type == "cuda" else "cpu"

    for noisy, gt in loader:
        noisy = noisy.to(device, non_blocking=True)
        gt    = gt.to(device,    non_blocking=True)

        with autocast(device_type=dtype, enabled=(device.type == "cuda")):
            pred = tta_predict(model, noisy) if use_tta else model(noisy)
            loss, *_ = criterion(pred, gt)

        pf = pred.float().clamp(0, 1)
        gf = gt.float()
        tl  += loss.item()
        tp  += compute_psnr(pf, gf)
        ts  += compute_ssim(pf, gf)
        if lpips_metric_fn is not None:
            p3 = pf.repeat(1, 3, 1, 1) * 2 - 1
            g3 = gf.repeat(1, 3, 1, 1) * 2 - 1
            tlp += lpips_metric_fn(p3.cpu(), g3.cpu()).mean().item()
        n += 1

    return tl/n, tp/n, ts/n, (tlp/n if lpips_metric_fn else None)


# ================================================================
# Main
# ================================================================
def main(args):
    # Apply argparse overrides to CFG
    if args.noisy_dir: CFG["noisy_dir"]   = args.noisy_dir
    if args.gt_dir:    CFG["gt_dir"]      = args.gt_dir
    if args.ckpt_dir:  CFG["ckpt_dir"]    = args.ckpt_dir
    if args.no_tta:    CFG["use_tta"]     = False
    if args.repro:     CFG["repro_mode"]  = True
    CFG["epochs"] = args.epochs
    CFG["lr"]     = args.lr

    set_seed(CFG["seed"], repro_mode=CFG["repro_mode"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Config summary ------------------------------------------------
    print(f"\n{'='*68}")
    print(f"  NAFNet  |  Charb({CFG['charb_w']}) + SSIM({CFG['ssim_w']}) "
          f"+ FFT({CFG['fft_w']}) + LPIPS-{CFG['train_lpips_net'].upper()}({CFG['lpips_w']})")
    print(f"  Device  : {device}  |  Target epochs: {CFG['epochs']}")
    print(f"  LPIPS train backbone : {CFG['train_lpips_net'].upper()} (frozen, training loss)")
    print(f"  LPIPS eval  backbone : {CFG['eval_lpips_net'].upper()} (evaluation metric)")
    print(f"  TTA at validation    : {CFG['use_tta']}  (8-way self-ensemble)")
    print(f"  EMA decay            : {CFG['ema_decay']}")
    print(f"  Reproducibility mode : {CFG['repro_mode']}")
    print(f"  Patch LR/HR          : {CFG['patch_size']} / {CFG['patch_size']*2}")
    print(f"  Seed                 : {CFG['seed']}")
    print(f"{'='*68}\n")

    # ---- LPIPS eval metric (AlexNet) -----------------------------------
    lpips_eval_fn = load_lpips_metric(net=CFG["eval_lpips_net"])
    if lpips_eval_fn:
        print(f"LPIPS eval ({CFG['eval_lpips_net'].upper()}): loaded")
    else:
        print(f"LPIPS eval: unavailable (pip install lpips)")

    # ---- Data ----------------------------------------------------------
    train_loader, val_loader = make_dataloaders(
        CFG["noisy_dir"], CFG["gt_dir"],
        val_split   = CFG["val_split"],
        batch_size  = CFG["batch_size"],
        num_workers = CFG["num_workers"],
        patch_size  = CFG["patch_size"],
        seed        = CFG["seed"],
    )

    # ---- Model (NAFNet, lighter architecture for throughput) ----------
    model = NAFNet(in_ch=1, width=CFG["width"],
                   enc_blocks=CFG["enc_blocks"],
                   dec_blocks=CFG["dec_blocks"],
                   middle_blocks=CFG["mid_blocks"]).to(device)
    ema = ModelEMA(model, decay=CFG["ema_decay"])
    n_params = count_params(model)
    print(f"NAFNet params: {n_params:,}  ({n_params/1e6:.2f}M, down from 19.84M)")

    # ---- Loss (LPIPS AlexNet as training loss) ------------------------
    print(f"Building CombinedLoss (train LPIPS backbone: {CFG['train_lpips_net'].upper()})...")
    criterion = CombinedLoss(
        cw=CFG["charb_w"], sw=CFG["ssim_w"],
        fw=CFG["fft_w"],   lw=CFG["lpips_w"],
        fft_phase_w=CFG["fft_phase_w"],
        train_net=CFG["train_lpips_net"],
    ).to(device)
    print("Loss ready.\n")

    # ---- Optimizer, Scheduler, Scaler ---------------------------------
    optimizer = AdamW(model.parameters(), lr=CFG["lr"],
                      weight_decay=CFG["weight_decay"], betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train_loader) / CFG["grad_accum"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch * 10,
        T_mult=1, eta_min=CFG["min_lr"]
    )
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    os.makedirs(CFG["ckpt_dir"], exist_ok=True)
    best_psnr  = 0.0
    best_lpips = float("inf")
    start_epoch = 1

    # ---- Resume (Change 5: true resume — all state restored) -----------
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        # Restore optimizer + scheduler so LR schedule continues correctly.
        # Previously these were NOT restored (caused LR warmup restart on
        # every resume, leading to instability on Colab disconnects).
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_psnr   = ckpt.get("best_psnr",  0.0)
        best_lpips  = ckpt.get("best_lpips", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (optimizer + scheduler restored)")
        print(f"  best_psnr  = {best_psnr:.2f} dB")
        lpips_str = f"{best_lpips:.4f}" if best_lpips < float("inf") else "none"
        print(f"  best_lpips = {lpips_str}")
        print(f"  LR         = {optimizer.param_groups[0]['lr']:.2e} (restored)")
        print(f"  Running epochs {start_epoch} to {CFG['epochs']}\n")

    # ---- Header -------------------------------------------------------
    tta_tag  = "[TTA]" if CFG["use_tta"] else "[no-TTA]"
    ema_tag  = "[EMA]"
    hdr = (f"{'Ep':>4} | {'TrLoss':>7} | {'ValLoss':>7} | "
           f"{'PSNR':>7}{tta_tag} | {'SSIM':>6} | {'LPIPS':>6}"
           f"{ema_tag} | {'LR':>9} | Time")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_c, tr_s, tr_f, tr_p = train_epoch(
            model, ema, train_loader, optimizer, scheduler,
            scaler, criterion, device, epoch, CFG
        )

        # Validate with EMA model weights (labeled explicitly)
        val_loss, val_psnr, val_ssim, val_lpips = validate(
            ema.model, val_loader, criterion, device,
            lpips_eval_fn, use_tta=CFG["use_tta"]
        )

        lr  = optimizer.param_groups[0]["lr"]
        dur = time.time() - t0
        lstr = f"{val_lpips:.4f}" if val_lpips is not None else "  N/A"

        # Epoch summary: all loss components + all metrics
        print(f"{epoch:4d} | {tr_loss:7.4f} | {val_loss:7.4f} | "
              f"{val_psnr:7.2f}       | {val_ssim:6.4f} | {lstr}      | "
              f"{lr:9.2e} | {dur:.0f}s")
        print(f"       train-> charb={tr_c:.4f} ssim={tr_s:.4f} "
              f"fft={tr_f:.4f} perc={tr_p:.4f}")

        # ---- Checkpoint -----------------------------------------------
        # Training checkpoint: contains raw model + EMA + full optimizer
        # state for seamless resume across sessions.
        train_ckpt = {
            "epoch"      : epoch,
            "model"      : model.state_dict(),   # raw weights (for resume)
            "ema"        : ema.state_dict(),      # EMA weights
            "optimizer"  : optimizer.state_dict(),
            "scheduler"  : scheduler.state_dict(),
            "scaler"     : scaler.state_dict(),
            "best_psnr"  : best_psnr,
            "best_lpips" : best_lpips,
            "cfg_snapshot": {
                k: CFG[k] for k in
                ("charb_w","ssim_w","fft_w","lpips_w","lr","patch_size",
                 "batch_size","grad_accum","seed","train_lpips_net",
                 "eval_lpips_net","use_tta","ema_decay",
                 "width","enc_blocks","dec_blocks","mid_blocks")
            },
        }
        # Change 4: submission checkpoint stores EMA weights as 'model'.
        # Validation already uses ema.model, so the submitted model must
        # match exactly what was evaluated — raw weights would be inconsistent.
        submit_ckpt = {
            "epoch"      : epoch,
            "model"      : ema.state_dict(),     # EMA = what was validated
            "best_psnr"  : best_psnr,
            "best_lpips" : best_lpips,
            "cfg_snapshot": train_ckpt["cfg_snapshot"],
            "note"       : "model key = EMA weights (matches validation)",
        }
        torch.save(train_ckpt, os.path.join(CFG["ckpt_dir"], "latest.pth"))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(train_ckpt, os.path.join(CFG["ckpt_dir"], "best_psnr.pth"))
            torch.save(submit_ckpt, os.path.join(CFG["ckpt_dir"], "best_psnr_submit.pth"))
            print(f"  --> New best PSNR {tta_tag}{ema_tag}: {best_psnr:.2f} dB")
            print(f"      Saved best_psnr_submit.pth (EMA weights = submission model)")

        if val_lpips is not None and val_lpips < best_lpips:
            best_lpips = val_lpips
            torch.save(train_ckpt, os.path.join(CFG["ckpt_dir"], "best_lpips.pth"))
            torch.save(submit_ckpt, os.path.join(CFG["ckpt_dir"], "best_lpips_submit.pth"))
            print(f"  --> New best LPIPS ({CFG['eval_lpips_net'].upper()})"
                  f"{ema_tag}: {best_lpips:.4f}")
            print(f"      Saved best_lpips_submit.pth (EMA weights = submission model)")

    print(f"\n{'='*55}")
    print(f"  Training complete.")
    print(f"  Best PSNR  {tta_tag}{ema_tag}: {best_psnr:.2f} dB")
    print(f"  Best LPIPS ({CFG['eval_lpips_net'].upper()}){ema_tag}: {best_lpips:.4f}")
    print(f"  Checkpoints: {CFG['ckpt_dir']}/")
    print(f"{'='*55}")


# ================================================================
# Argparse  --  only for paths and a few key flags.
# All hyperparameters live in CFG above.
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NAFNet training - semiconductor denoising + 2x SR"
    )
    # Path overrides (for portability)
    parser.add_argument("--noisy-dir", type=str, default=None,
                        help="Path to NoisyLR directory")
    parser.add_argument("--gt-dir",    type=str, default=None,
                        help="Path to GT directory")
    parser.add_argument("--ckpt-dir",  type=str, default=None,
                        help="Checkpoint output directory")
    # Resume
    parser.add_argument("--resume",    type=str, default=None,
                        help="Path to checkpoint to resume from")
    # Schedule
    parser.add_argument("--epochs",    type=int,   default=CFG["epochs"])
    parser.add_argument("--lr",        type=float, default=CFG["lr"])
    # Flags
    parser.add_argument("--no-tta",    action="store_true",
                        help="Disable TTA at validation (faster but lower PSNR)")
    parser.add_argument("--repro",     action="store_true",
                        help="Enable fully deterministic mode (slower)")
    args = parser.parse_args()
    main(args)