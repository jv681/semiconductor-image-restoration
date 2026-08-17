"""
model.py
NAFNet (Nonlinear Activation Free Network) for semiconductor denoising + 2x SR.
Paper: "Simple Baselines for Image Restoration" (ECCV 2022)

Input:  (B, 1, 128, 128)  -- noisy low-res
Output: (B, 1, 256, 256)  -- clean high-res
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------
# NAFNet Core Building Blocks
# ---------------------------------------------

class LayerNormCh(nn.Module):
    """LayerNorm over channel dimension for (B, C, H, W) tensors."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies - replaces all nonlinear activations."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    """Simplified Channel Attention: global avg pool -> 1x1 conv -> rescale."""
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
    """
    NAFNet Block:
      - Depthwise conv (spatial mixing)
      - SimpleGate (activation-free)
      - Simple Channel Attention
      - FFN with SimpleGate
      - LayerNorm on input to each sub-block
    """
    def __init__(self, channels, ffn_expand=2, dw_expand=1):
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand

        # Branch 1: depthwise conv + SimpleGate + SCA
        self.norm1 = LayerNormCh(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch * 2, 1)           # pointwise expand
        self.dw    = nn.Conv2d(dw_ch * 2, dw_ch * 2, 3, 1, 1,
                               groups=dw_ch * 2)                  # depthwise
        self.gate1 = SimpleGate()                                  # dw_ch * 2 -> dw_ch
        self.sca   = SimpleChannelAttention(dw_ch)
        self.proj1 = nn.Conv2d(dw_ch, channels, 1)               # project back

        # Branch 2: FFN with SimpleGate
        self.norm2 = LayerNormCh(channels)
        self.ffn1  = nn.Conv2d(channels, ffn_ch * 2, 1)
        self.gate2 = SimpleGate()                                  # ffn_ch * 2 -> ffn_ch
        self.ffn2  = nn.Conv2d(ffn_ch, channels, 1)

        # Learnable residual scaling
        self.beta  = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x):
        # Spatial branch
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dw(y)
        y = self.gate1(y)              # -> dw_ch
        y = self.sca(y)
        y = self.proj1(y)
        x = x + y * self.beta

        # FFN branch
        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.gate2(y)              # -> ffn_ch
        y = self.ffn2(y)
        x = x + y * self.gamma

        return x


# ---------------------------------------------
# NAFNet Encoder-Decoder with 2x SR
# ---------------------------------------------

class NAFNet(nn.Module):
    """
    NAFNet for denoising + 2x super-resolution.

    Architecture:
      head  -> [enc blocks] -> [downsample] -> bottleneck
            -> [upsample] -> [dec blocks + skip] -> tail (pixel shuffle 2x)

    Input:  (B, 1, 128, 128)
    Output: (B, 1, 256, 256)
    """
    def __init__(
        self,
        in_ch      = 1,
        width      = 32,
        enc_blocks = [2, 2, 4, 8],   # NAFBlocks at each encoder level
        dec_blocks = [2, 2, 2, 2],   # NAFBlocks at each decoder level
        middle_blocks = 4,
    ):
        super().__init__()

        self.intro  = nn.Conv2d(in_ch, width, 3, 1, 1)

        # -- Encoder ------------------------------
        self.encoders   = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = width
        enc_channels = []
        for num_blks in enc_blocks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(num_blks)])
            )
            enc_channels.append(ch)
            self.downsamples.append(
                nn.Conv2d(ch, ch * 2, 2, 2)   # strided conv downsample
            )
            ch *= 2

        # -- Bottleneck ---------------------------
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blocks)])

        # -- Decoder ------------------------------
        self.upsamples = nn.ModuleList()
        self.decoders  = nn.ModuleList()

        for num_blks, skip_ch in zip(dec_blocks, reversed(enc_channels)):
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(ch, skip_ch * 4, 1),
                    nn.PixelShuffle(2)          # upsample 2x, ch -> skip_ch
                )
            )
            ch = skip_ch
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(num_blks)])
            )

        # -- Output: restore + 2x SR --------------
        # After decoder we're back at 'width' channels and 128x128 spatial size.
        # Use pixel shuffle to go to 256x256.
        self.tail = nn.Sequential(
            NAFBlock(ch),
            nn.Conv2d(ch, in_ch * 4, 3, 1, 1),   # 4 = scale^2
            nn.PixelShuffle(2),                    # -> (B, 1, 256, 256)
        )

    def forward(self, x):
        x = self.intro(x)    # (B, width, 128, 128)

        # Encode
        enc_feats = []
        for enc, down in zip(self.encoders, self.downsamples):
            x = enc(x)
            enc_feats.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle(x)

        # Decode
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(enc_feats)):
            x = up(x)
            x = x + skip       # skip connection (same channel count)
            x = dec(x)

        # Output with 2x upscale
        x = self.tail(x)       # (B, 1, 256, 256)
        x = torch.clamp(x, 0, 1)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = NAFNet(in_ch=1, width=32,
                   enc_blocks=[2, 2, 4, 8],
                   dec_blocks=[2, 2, 2, 2],
                   middle_blocks=4)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {count_params(model):,}")
