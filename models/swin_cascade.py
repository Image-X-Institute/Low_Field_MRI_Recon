"""
SwinCascade: Cascaded Swin Transformer for MRI reconstruction.

This implementation leverages the Swin reconstruction codebase from:
    Rahman T, Bilgin A, Cabrera SD. "Multi-channel MRI reconstruction using
    cascaded Swinμ transformers with overlapped attention."
    Phys. Med. Biol. 70, 075002 (2025). https://doi.org/10.1088/1361-6560/adb933

Architecture overview:
  - SwinCascade: stacks num_cascades SwinBlock modules. Between each block,
    data consistency is enforced by replacing acquired k-space lines with
    measured values. The final block runs without data consistency.
  - SwinBlock: wraps SwinIM for image-domain refinement.
  - SwinIM: image-to-image Swin Transformer (image denoising mode, no
    upsampling). Built from stacked RSTB → BasicLayer → SwinTransformerBlock
    → WindowAttention stages, with optional Channel Attention Blocks (CAB).

Input convention:
  Images are represented as two-channel real tensors [B, 2, H, W] (real, imag).
  K-space is complex [B, 1, H, W].
  The mask is float [B, 1, H, W] with 1 at acquired lines.

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Tuple


# ===========================================================================
# FFT / complex utilities
# ===========================================================================

def fft2t(data: torch.Tensor,
          dim: Tuple[int, int] = (-1, -2),
          centered: bool = True,
          normalized: bool = True) -> torch.Tensor:
    """Centred 2-D FFT."""
    if centered:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm='ortho' if normalized else None)
    if centered:
        data = torch.fft.fftshift(data, dim=dim)
    return data


def ifft2t(data: torch.Tensor,
           dim: Tuple[int, int] = (-1, -2),
           centered: bool = True,
           normalized: bool = True) -> torch.Tensor:
    """Centred 2-D IFFT."""
    if centered:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm='ortho' if normalized else None)
    if centered:
        data = torch.fft.fftshift(data, dim=dim)
    return data


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Complex [B, 1, H, W] → real/imag channels [B, 2, H, W]."""
    if x.ndim == 3:
        return torch.stack([x.real, x.imag], dim=1)
    return torch.cat([x.real, x.imag], dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Real/imag channels [B, 2, H, W] → complex [B, 1, H, W]."""
    return torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)


# ===========================================================================
# SwinCascade
# ===========================================================================

class SwinBlock(nn.Module):
    """Single cascade block: image-domain refinement via SwinIM."""

    def __init__(self, swin_args: dict):
        super().__init__()
        self.image_domain_net = SwinIM(**swin_args)

    def forward(self, img_chan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_chan: [B, 2, H, W] — two-channel (real/imag) image.
        Returns:
            img_chan_refined: [B, 2, H, W]
        """
        return self.image_domain_net(img_chan)


class SwinCascade(nn.Module):
    """
    Cascaded Swin Transformer reconstruction network.

    For each cascade (except the last):
      1. IFFT k-space → image domain.
      2. Refine image with SwinBlock.
      3. FFT refined image → k-space.
      4. Data consistency: replace acquired lines with measured k-space.

    The final cascade skips data consistency so the network can freely
    synthesise the missing lines.

    Default hyperparameters match the trained checkpoint used in this work
    (6 cascades, embed_dim=96, 4 transformer stages).
    """

    DEFAULT_SWIN_ARGS = {
        'img_size': 96,
        'patch_size': 4,
        'in_chans': 2,
        'embed_dim': 96,
        'depths': [2, 2, 2, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 8,
        'mlp_ratio': 2,
        'qkv_bias': True,
        'use_checkpoint': False,
        'overlap': False,
    }

    def __init__(self, num_cascades: int = 6, swin_args: dict = None):
        super().__init__()
        self.num_cascades = num_cascades
        args = swin_args if swin_args is not None else self.DEFAULT_SWIN_ARGS
        self.blocks = nn.ModuleList([SwinBlock(args) for _ in range(num_cascades)])

    def forward(self, zf_img_chan: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zf_img_chan: zero-filled image [B, 2, H, W].
            mask:        sampling mask [B, 1, H, W], float, 1 = acquired.
        Returns:
            out_img_chan: reconstructed image [B, 2, H, W].
        """
        k_cplx = fft2t(channels_to_complex(zf_img_chan))
        k_zf = k_cplx.clone().detach()  # measured k-space for data consistency

        for i, block in enumerate(self.blocks):
            img_chan = complex_to_channels(ifft2t(k_cplx).squeeze(1))
            img_chan = block(img_chan)
            k_cplx = fft2t(channels_to_complex(img_chan))

            # Data consistency: keep acquired lines from measurement
            if i < self.num_cascades - 1:
                k_cplx = mask * k_zf + (1 - mask) * k_cplx

        return complex_to_channels(ifft2t(k_cplx).squeeze(1))


# ===========================================================================
# SwinIM — image-to-image Swin Transformer
# (Rahman et al. 2025, Phys. Med. Biol. 70, 075002)
# ===========================================================================

class ChannelAttention(nn.Module):
    """Channel attention (squeeze-and-excitation style, from RCAN)."""

    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    """Channel Attention Block: Conv → GELU → Conv → ChannelAttention."""

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    """Two-layer MLP with dropout, used as the FFN in transformer blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """Reverse window_partition.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (Swin Transformer V2).

    Uses cosine attention with a learned continuous relative position bias MLP
    instead of the additive bias table of V1.

    Args:
        dim: number of input channels.
        window_size: (Wh, Ww).
        num_heads: number of attention heads.
        qkv_bias: add learnable bias to Q, K, V.
        pretrained_window_size: window size used during pre-training (for bias rescaling).
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # Continuous relative position bias via small MLP
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Relative coordinate table
        relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])
        ).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = (torch.sign(relative_coords_table)
                                 * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8))
        self.register_buffer("relative_coords_table", relative_coords_table)

        # Pair-wise relative position index
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention with learnable scale
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))
        ).exp()
        attn = attn * logit_scale

        # Continuous relative position bias
        rpb_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        rpb = rpb_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        attn = attn + 16 * torch.sigmoid(rpb.permute(2, 0, 1).contiguous()).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block: (SW-)MSA + CAB + FFN with residual connections.

    Alternates between W-MSA (shift_size=0) and SW-MSA (shift_size=window_size//2)
    across consecutive blocks. CAB provides complementary channel attention.

    Args:
        dim: number of channels.
        input_resolution: (H, W) of the feature map.
        num_heads: attention heads.
        window_size: local window size.
        shift_size: cyclic shift (0 for W-MSA, window_size//2 for SW-MSA).
        CAB_scale: weight of the CAB branch (0 disables CAB).
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 compress_ratio=3, squeeze_factor=30, CAB_scale=0.01,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.CAB_scale = CAB_scale

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution) * (self.window_size // 8)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        if self.CAB_scale > 0:
            self.CAB_block = CAB(num_feat=dim, compress_ratio=compress_ratio,
                                 squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        attn_mask = self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        for h in (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None)):
            for w in (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = len(img_mask)  # unique region id per cell
        # re-implement correct region labelling
        img_mask = torch.zeros((1, H, W, 1))
        cnt = 0
        for h in (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None)):
            for w in (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(
            -1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = x.view(B, H, W, C)

        if self.CAB_scale > 0:
            CAB_x = self.CAB_block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        attn_mask = self.attn_mask if self.input_resolution == x_size else self.calculate_mask(x_size).to(x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C),
                                self.window_size, H, W)
        if self.shift_size > 0:
            attn_x = torch.roll(attn_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        attn_x = attn_x.view(B, H * W, C)

        if self.CAB_scale > 0:
            x = shortcut + self.drop_path(self.norm1(attn_x + CAB_x * self.CAB_scale))
        else:
            x = shortcut + self.drop_path(self.norm1(attn_x))

        return x + self.drop_path(self.norm2(self.mlp(x)))


class BasicLayer(nn.Module):
    """One stage of the SwinIM hierarchy: a sequence of SwinTransformerBlocks.

    Alternates W-MSA and SW-MSA across even/odd block indices.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 compress_ratio=3, squeeze_factor=30, CAB_scale=0.01,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, pretrained_window_size=0,
                 **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
                CAB_scale=CAB_scale, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)
        ])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x, x_size) if self.use_checkpoint else blk(x, x_size)
        return x


class PatchEmbed(nn.Module):
    """Flatten spatial dims for transformer input: [B, C, H, W] → [B, H*W, C]."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Unflatten transformer output: [B, H*W, C] → [B, C, H, W]."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class RSTB(nn.Module):
    """Residual Swin Transformer Block: BasicLayer + Conv residual."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 compress_ratio=3, squeeze_factor=30, CAB_scale=0.01,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv', **kwargs):
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads,
            window_size=window_size, compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor, CAB_scale=CAB_scale,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        else:  # '3conv' — bottleneck to save parameters
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.conv(self.patch_unembed(self.residual_group(self.patch_embed(x), x_size), x_size)) + x


class SwinIM(nn.Module):
    """SwinIM: image-to-image Swin Transformer (image denoising mode).

    Architecture (no upsampling):
      1. Shallow feature extraction: Conv2d → embed_dim channels.
         An optional expansion branch (exp_conv) produces a parallel feature
         map at a higher channel count that is concatenated before the final conv.
      2. Deep feature extraction: num_layers RSTB stages, each containing
         `depths[i]` SwinTransformerBlocks with alternating W-MSA / SW-MSA.
         A LayerNorm and residual Conv follow the transformer stages.
      3. Reconstruction: Conv2d back to in_chans, with global residual.

    Args:
        img_size: expected spatial size (H = W).
        patch_size: patch size for PatchEmbed (1 = pixel-level).
        in_chans: input/output channels (2 for real+imag).
        embed_dim: transformer channel dimension.
        depths: number of SwinTransformerBlocks per RSTB stage.
        num_heads: attention heads per stage.
        window_size: local attention window size.
        mlp_ratio: hidden-dim multiplier in the FFN.
        overlap: if True, use overlapped window attention (OTB2) instead of
                 standard WindowAttention. Set to False in this work.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, overlap=False, overlap_ratio=0.5,
                 compress_ratio=6, squeeze_factor=30, CAB_scale=0.01,
                 mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1.,
                 exp_conv=True, exp_factor=1, upsampler='', resi_connection='1conv',
                 **kwargs):
        super().__init__()
        self.exp_conv = exp_conv
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1) if in_chans != 3 else torch.Tensor(
            (0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        self.window_size = window_size
        self.patch_size = patch_size

        # 1. Shallow feature extraction
        if exp_conv:
            self.conv_expand_in = nn.Conv2d(in_chans, int(embed_dim * exp_factor), 3, 1, 1)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction (RSTB stages)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList([
            RSTB(dim=embed_dim,
                 input_resolution=(self.patch_embed.patches_resolution[0],
                                   self.patch_embed.patches_resolution[1]),
                 depth=depths[i],
                 num_heads=num_heads[i],
                 window_size=window_size,
                 compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
                 CAB_scale=CAB_scale, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 drop=drop_rate, attn_drop=attn_drop_rate,
                 drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                 norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                 img_size=img_size, patch_size=patch_size,
                 resi_connection=resi_connection)
            for i in range(self.num_layers)
        ])
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # 3. Reconstruction
        if exp_conv:
            self.conv_expand_out = nn.Conv2d(int(embed_dim * exp_factor), embed_dim, 3, 1, 1)
            self.conv_last = nn.Conv2d(embed_dim * 2, in_chans, 3, 1, 1)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        return self.patch_unembed(self.norm(x), x_size)

    def forward(self, x):
        C, H, W = x.shape[1:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) / self.img_range

        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first

        if self.exp_conv:
            res2 = self.conv_expand_out(self.conv_expand_in(x))
            x = x + self.conv_last(torch.cat((res, res2), dim=1))
        else:
            x = x + self.conv_last(res)

        x = x * self.img_range + self.mean
        return x[:, :, :H, :W]
