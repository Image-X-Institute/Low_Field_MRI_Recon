"""
UnrolledNet: Unrolled ISTA for MRI reconstruction with data consistency.

Based on:
    Shan S, Gao Y, Liu PZY, Whelan B, Sun H, Dong B, Liu F, Waddington DEJ.
    "Distortion-corrected image reconstruction with deep learning on an MRI-Linac."
    Magn. Reson. Med. 90, 615–629 (2023). https://doi.org/10.1002/mrm.29684

Architecture overview:
  - UnrolledNet: top-level module; stacks `num_iters` instances of UnrolledIter.
  - UnrolledIter: one unrolling step. Applies data consistency (DataConsistency),
    then a learned proximal step:
        Conv → H_forward → soft-threshold → H_backward → Conv.
  - H_Operator: two-layer Conv block approximating a learned linear transform.
    Used as both H_forward and H_backward; a symmetry loss during training
    encourages H_back ≈ H_for^-1.
  - DataConsistency: replaces acquired k-space lines with measured values then
    back-transforms. Instantiated per-slice at inference with the measured
    k-space and mask baked in.

Data format: real-valued two-channel tensors [B, 2, H, W] (real, imag).
R2C / C2R convert between this format and complex [B, 1, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class UnrolledNet(nn.Module):
    def __init__(self, num_iters, imsize, ini_flag=False, num_in=16, num_out=16):
        """
        Args:
            num_iters: number of unrolling iterations.
            imsize:    (H, W) tuple.
            ini_flag:  if True, initialise x with A^H y instead of zeros.
            num_in/num_out: feature channels in Conv layers.
        """
        super().__init__()
        self.num_iters = num_iters
        self.ini_flag = ini_flag
        self.iters = nn.ModuleList([
            UnrolledIter(imsize, num_in, num_out) for _ in range(num_iters)
        ])

    def forward(self, AHy, dc, device='cuda'):
        """
        Args:
            AHy:    zero-filled image [B, 2, H, W] (real/imag channels).
            dc:     instantiated DataConsistency module for this slice.
            device: 'cuda' or 'cpu'.
        Returns:
            x:            reconstructed image [B, 2, H, W].
            sym_losses:   list of symmetry losses (H_back ≈ H_for^-1) per iteration.
            df_losses:    list of data-fidelity residuals per iteration.
        """
        x = torch.zeros_like(AHy).to(device)
        if self.ini_flag:
            x = AHy

        sym_losses, df_losses = [], []
        for layer in self.iters:
            x, sym_loss, df_loss = layer(x, dc)
            sym_losses.append(sym_loss)
            df_losses.append(df_loss)

        return x, sym_losses, df_losses


class UnrolledIter(nn.Module):
    """One unrolled ISTA iteration."""

    def __init__(self, imsize, num_in=16, num_out=16, ks=3, pad=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.conv_D = nn.Conv2d(2, num_in, ks, padding=pad)
        self.H_for = H_Operator(num_in, num_out)
        self.H_back = H_Operator(num_out, num_out)
        self.conv_G = nn.Conv2d(num_out, 2, ks, padding=pad)

    def forward(self, x, dc):
        # Data consistency update
        x_input, df_loss, x_updated = dc(x, self.alpha)
        x = x_updated

        # Proximal step in learned transform domain
        x_D = self.conv_D(x_input)
        x_forward = self.H_for(x_D)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x_backward = self.H_back(x)
        x_pred = x_input + self.conv_G(x_backward)

        # Symmetry loss: H_back(H_for(x_D)) ≈ x_D
        sym_loss = self.H_back(x_forward) - x_D

        return x_pred, sym_loss, df_loss


class H_Operator(nn.Module):
    """Two-layer Conv block used as forward/backward transform."""

    def __init__(self, num_in=16, num_out=16, ks=3, pad=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_in, num_out, ks, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_out, num_out, ks, padding=pad),
        )

    def forward(self, x):
        return self.net(x)


class DataConsistency(nn.Module):
    """
    Data-consistency block. Replaces acquired k-space lines with measured
    values, then returns the back-projected (IFFT) result.

    Instantiated once per slice at inference time with the measured k-space
    and mask baked in.

    Args:
        imsize: (H, W).
        AHy:    zero-filled image [B, 2, H, W].
        mask:   binary sampling mask [1, 1, H, W] (byte/bool).
        ksps:   measured k-space [B, 2, H, W].
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, imsize, AHy, mask, ksps, device):
        super().__init__()
        self.mask = mask
        self.AHy = R2C(AHy.to(device))
        self.ksps = R2C(ksps.to(device))
        self.xx = torch.zeros(1, 1, *imsize, dtype=torch.float32).to(device)

    def forward(self, x, alpha):
        """
        1. FFT(x), replace acquired lines with measured k-space, IFFT → x_dc.
        2. Gradient step: R = x_dc - alpha * (A^H A x_dc - A^H y).
        """
        x = R2C(x)

        # Replace acquired k-space lines with measurements
        x = fft.fftshift(fft.fftn(fft.ifftshift(x, dim=(2, 3)), dim=(2, 3)), dim=(2, 3))
        x[self.mask.type(torch.bool)] = self.ksps[self.mask.type(torch.bool)]
        x = fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=(2, 3)), dim=(2, 3)), dim=(2, 3))

        x_dc = x
        x_dc_real = C2R(x)

        # A^H A x_dc (mask in k-space, back to image)
        x = fft.fftshift(fft.fftn(fft.ifftshift(x, dim=(2, 3)), dim=(2, 3)), dim=(2, 3))
        x = x * self.mask
        x = fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=(2, 3)), dim=(2, 3)), dim=(2, 3))

        x = torch.abs(x) + 1j * self.xx
        df_loss = x - torch.abs(self.AHy)  # A^H A x_dc - A^H y

        R = C2R(x_dc - alpha * df_loss)
        return R, C2R(df_loss), x_dc_real


# ---------------------------------------------------------------------------
# Complex ↔ two-channel real conversions
# ---------------------------------------------------------------------------

def R2C(x):
    """[B, 2, H, W] real → [B, 1, H, W] complex."""
    return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)


def C2R(x):
    """[B, 1, H, W] complex → [B, 2, H, W] real."""
    return torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2).contiguous()
