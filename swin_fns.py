import os
import numpy as np
import torch
import sigpy as sp

from trained_models.SwinCascade.SwinCascadeModel import SwinCascade
from trained_models.SwinCascade.swin_utils import (
    fft2t, ifft2t,
    complex_to_channels, channels_to_complex,
    pad_or_crop
)

def load_swincascade_model(model_path, device='cuda', num_cascades=6):
    """
    Load SwinCascade model from a given .pth checkpoint file.

    Args:
        model_path (str): Path to saved model .pth file.
        device (str): 'cuda' or 'cpu'.
        num_cascades (int): Number of cascades in SwinCascade.

    Returns:
        torch.nn.Module: Loaded SwinCascade model.
    """
    model = SwinCascade(num_cascades=num_cascades)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def reconstruct_slice_swin(kspace_lr, mask_lr, model, device='cuda', target_size=(96, 96)):
    """
    Reconstruct image from low-res undersampled k-space using SwinCascade model trained at higher resolution.

    Args:
        kspace_lr (np.ndarray): Undersampled complex-valued k-space, shape [H, W]
        mask_lr (np.ndarray): Binary mask of same shape as kspace_lr
        model (nn.Module): Trained SwinCascade model
        device (str): 'cuda' or 'cpu'
        target_size (tuple): Resolution the model expects, e.g. (96, 96)

    Returns:
        tuple:
            - recon_img_crop (np.ndarray): Magnitude image reconstructed and cropped to original resolution
            - kspace_fused_crop (np.ndarray): Complex-valued synthetic full k-space, cropped to original resolution
    """
    model.eval()
    original_size = kspace_lr.shape

    # --- Pad undersampled k-space and mask to training resolution ---
    kspace_hr = pad_or_crop(kspace_lr, target_size)
    mask_hr = pad_or_crop(mask_lr, target_size)

    # --- Zero-filled reconstruction input ---
    kspace_tensor = torch.tensor(kspace_hr).to(device)
    mask_tensor = torch.tensor(np.real(mask_hr)).unsqueeze(0).unsqueeze(0).float().to(device)

    # Inverse FT to get zero-filled complex image
    img_cplx = ifft2t(kspace_tensor.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
    zf_img = complex_to_channels(img_cplx)[0]  # [2, H, W]
    zf_img = zf_img.unsqueeze(0)  # [1, 2, H, W]

    # Scale before feeding into the model
    zf_max = torch.abs(zf_img).max()
    zf_img_scaled = zf_img / (zf_max + 1e-8)
    #print("Max magnitude after scaling:", torch.abs(zf_img_scaled).max().item())
    
    # --- Run model forward ---
    with torch.no_grad():
        recon_scaled = model(zf_img_scaled, mask_tensor)  # [1, 2, H, W]

    # Rescale the output
    recon = recon_scaled * zf_max
    
    # Convert model output to complex and compute full k-space
    recon_cplx = channels_to_complex(recon).squeeze(0).squeeze(0)  # [H, W]
    kspace_fused = fft2t(recon_cplx.unsqueeze(0).unsqueeze(0))     # [1, 1, H, W]

    # --- Crop synthetic k-space back to original resolution ---
    kspace_fused_crop = pad_or_crop(
        kspace_fused.squeeze(0).squeeze(0).cpu().numpy(),
        output_size=original_size
    )

    # --- Reconstruct final image from cropped k-space ---
    recon_cplx_crop = ifft2t(torch.tensor(kspace_fused_crop).unsqueeze(0).unsqueeze(0).to(device))  # [1, 1, H, W]
    recon_img_crop = torch.abs(recon_cplx_crop.squeeze()).cpu().numpy()

    return recon_img_crop, kspace_fused_crop

def swinRecon(kspace, mask, model, device='cuda', target_size=(96, 96)):
    """
    Wrapper to apply SwinCascade to an entire 3D volume (RO × PE1 × PE2).
    Matches pattern of automap/unroll recon functions.

    Args:
        kspace (np.ndarray): Shape [1, RO, PE1, PE2] or [RO, PE1, PE2]
        mask (np.ndarray): Shape [PE1, PE2] — must match pre-transposed k-space
        model (torch.nn.Module): SwinCascade model
        device (str): Torch device string
        target_size (tuple): Expected input size per slice (e.g. 96x96)

    Returns:
        tuple:
            - recon_vol (np.ndarray): Magnitude volume [RO, PE1, PE2]
            - kspace_filled_vol (np.ndarray): Reconstructed k-space [RO, PE1, PE2]
    """
    # Remove batch dim if present
    if kspace.ndim == 4:
        kspace = np.squeeze(kspace)  # (RO, PE1, PE2)

    # Transpose to match expected shape: (RO, PE1, PE2) ← originally (PE1, RO, PE2)
    kspace = np.transpose(kspace, (1, 0, 2))  # → (RO, PE1, PE2)

    # Validate or reshape mask to match (PE1, PE2)
    if mask.shape != (kspace.shape[1], kspace.shape[2]):
        raise ValueError(f"Mask shape {mask.shape} does not match PE dims {kspace.shape[1:]}")

    # Broadcast mask across RO
    mask_full = np.broadcast_to(mask[None, :, :], kspace.shape)  # (RO, PE1, PE2)

    # Hybrid space transform: IFFT along RO axis
    hybrid_space = sp.ifft(kspace, axes=(0,), center=True)

    recon_slices = []
    kspace_slices = []

    # Reconstruct each PE1×PE2 slice independently
    for i in range(hybrid_space.shape[0]):
        recon, kf = reconstruct_slice_swin(
            hybrid_space[i], mask_full[i], model, device=device, target_size=target_size
        )
        recon_slices.append(recon)
        kspace_slices.append(kf)

    # Stack slices back to full volume
    recon_vol = np.stack(recon_slices, axis=0)         # [RO, PE1, PE2]
    kspace_filled_vol = np.stack(kspace_slices, axis=0)

    # Transpose to match original input orientation: [PE1, RO, PE2]
    recon_vol = np.transpose(recon_vol, (1, 0, 2))         # [PE1, RO, PE2]
    kspace_filled_vol = np.transpose(kspace_filled_vol, (1, 0, 2))

    # # Call the new batched volume reconstruction function
    # batch_size=8
    # recon_vol, kspace_filled_vol = reconstruct_vol_swin(
    #     hybrid_space,
    #     mask_full,
    #     model,
    #     device=device,
    #     target_size=target_size,
    #     batch_size=batch_size
    # )
    
    # # Return in original orientation: [PE1, RO, PE2]
    # recon_vol = np.transpose(recon_vol, (1, 0, 2))
    # kspace_filled_vol = np.transpose(kspace_filled_vol, (1, 0, 2))
    
    return recon_vol, kspace_filled_vol

# def reconstruct_vol_swin(kspace_vol, mask_vol, model, device='cuda', target_size=(96, 96), batch_size=32):
#     import torch
#     import numpy as np
#     import sigpy as sp
#     from trained_models.SwinCascade.swin_utils import (
#         fft2t, ifft2t, complex_to_channels, channels_to_complex, pad_or_crop
#     )

#     model.eval()
#     RO = kspace_vol.shape[0]
#     hybrid_space = sp.ifft(kspace_vol, axes=(0,), center=True)

#     recon_slices = []
#     kspace_slices = []

#     for batch_start in range(0, RO, batch_size):
#         batch_end = min(batch_start + batch_size, RO)
#         batch_slices = hybrid_space[batch_start:batch_end]
#         batch_masks = mask_vol[batch_start:batch_end]

#         originals = [s.shape for s in batch_slices]

#         # Pad and convert to tensors
#         batch_kspace_hr = [pad_or_crop(s, target_size) for s in batch_slices]
#         batch_mask_hr = [pad_or_crop(m, target_size) for m in batch_masks]

#         kspace_tensor = torch.stack([torch.tensor(k) for k in batch_kspace_hr]).to(device)  # [B, H, W]
#         mask_tensor = torch.stack([torch.tensor(np.real(m)) for m in batch_mask_hr]).unsqueeze(1).float().to(device)  # [B, 1, H, W]

#         # IFFT to image domain
#         img_cplx = ifft2t(kspace_tensor.unsqueeze(1))  # [B, 1, H, W]
#         zf_img = complex_to_channels(img_cplx)         # [B, 2, H, W]

#         zf_max = torch.amax(torch.abs(zf_img), dim=(1,2,3), keepdim=True)  # [B, 1, 1, 1]
#         zf_img_scaled = zf_img / (zf_max + 1e-8)

#         with torch.no_grad():
#             recon_scaled = model(zf_img_scaled, mask_tensor)  # [B, 2, H, W]

#         # Rescale
#         recon = recon_scaled * zf_max
#         recon_cplx = channels_to_complex(recon)[:, 0]  # [B, H, W]

#         # Compute k-space
#         kspace_fused = fft2t(recon_cplx.unsqueeze(1))  # [B, 1, H, W]

#         for i in range(batch_end - batch_start):
#             crop_size = originals[i]
#             kf_crop = pad_or_crop(kspace_fused[i, 0].cpu().numpy(), crop_size)

#             # Convert to proper complex tensor
#             kf_tensor = torch.tensor(kf_crop).to(device)
#             recon_crop = ifft2t(kf_tensor[None, None])  # [1, 1, H, W]
#             recon_img = torch.abs(recon_crop.squeeze()).cpu().numpy()

#             recon_slices.append(recon_img)
#             kspace_slices.append(kf_crop)

#     recon_vol = np.stack(recon_slices, axis=0)         # [RO, PE1, PE2]
#     kspace_filled_vol = np.stack(kspace_slices, axis=0)
#     return recon_vol, kspace_filled_vol
