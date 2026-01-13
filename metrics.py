import numpy as np
from SSIM_PIL import compare_ssim
from PIL import Image
# E. Shimron, 2021. Modified by D. Waddington, 2025.

  
def calc_NRMSE(I_pred,I_true):    
        # Reshape the images into vectors
        I_true = np.reshape(I_true,(1,-1))   
        I_pred = np.reshape(I_pred,(1,-1))               
        # Mean Square Error
        MSE = np.square(np.subtract(I_true,I_pred)).mean()       
        # Root Mean Square Error
        RMSE = np.sqrt(MSE)
        # Normalized Root Mean Square Error
        rr = np.max(I_true) - np.min(I_true) # range
        NRMSE = RMSE/rr
        return NRMSE
        
def calc_SSIM(I_pred,I_true):
        # Note: in order to use the function compare_ssim, the images must be converted to PIL format
        # also, it can't calculate SSIM for a volume so we iterate through slices and take an average

        if I_true.ndim > 2:
            I_true_vol = abs(I_true)
            I_pred_vol = abs(I_pred)
        else:
            I_true_vol = abs(I_true[:,None])
            I_pred_vol = abs(I_pred[:,None])

        no_slices = I_true_vol.shape[2]
        SSIM_vals = np.zeros(no_slices)

        for slc in range(no_slices):
            I_true = np.squeeze(I_true_vol[:,:,slc])
            I_pred = np.squeeze(I_pred_vol[:,:,slc])
            # convert the images from float32 to uint8 format
            im1_mag_uint8 = (I_true * 255 / (np.max(I_true)+1e-8)).astype('uint8')
            im2_mag_uint8 = (I_pred * 255 / (np.max(I_pred)+1e-8)).astype('uint8')
            # convert from numpy array to PIL format
            im1_PIL = Image.fromarray(im1_mag_uint8)
            im2_PIL = Image.fromarray(im2_mag_uint8)

            SSIM_vals[slc] = compare_ssim(im1_PIL, im2_PIL)

        SSIM = np.mean(SSIM_vals)
        return SSIM


import numpy as np
from skimage.filters import sobel_h, sobel_v

def tenengrad_volume_3d(
    volume: np.ndarray,
    axis: int = -1,
    clip_percentiles: tuple | None = None,
):
    """
    Slice-wise Tenengrad (Sobel gradient energy) for a 3D MRI volume.

    Parameters
    ----------
    volume : np.ndarray
        3D complex or real array with shape (D, H, W) in any orientation.
        Complex inputs are converted to magnitude with np.abs.
    axis : int, default -1
        Axis along which to slice/average (0, 1, or 2).
    clip_percentiles : (float, float) or None
        Optional robust intensity clipping BEFORE global normalization, e.g. (1, 99).

    Returns
    -------
    mean_score : float
        Mean Tenengrad across valid slices (higher = sharper / more edges).
    std_score : float
        Standard deviation across valid slices.
    per_slice_scores : np.ndarray
        Tenengrad for each slice (np.nan where invalid).
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    # complex -> magnitude
    vol = np.abs(volume).astype(np.float32)

    # Optional robust clipping across the whole volume
    if clip_percentiles is not None:
        p_lo, p_hi = np.percentile(vol, clip_percentiles)
        if p_hi > p_lo:
            vol = np.clip(vol, p_lo, p_hi)

    # Global min-max normalization to [0,1]
    vmin, vmax = float(np.nanmin(vol)), float(np.nanmax(vol))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        S = vol.shape[axis]
        return np.nan, np.nan, np.full(S, np.nan, dtype=float)
    vol = (vol - vmin) / (vmax - vmin + 1e-12)

    # Reorder so chosen axis is first: (S, H, W)
    vol_slices = np.moveaxis(vol, axis, 0)
    S = vol_slices.shape[0]
    scores = np.full(S, np.nan, dtype=float)

    for i in range(S):
        sl = vol_slices[i]
        if sl.ndim != 2 or sl.size == 0 or not np.isfinite(sl).all():
            continue
        if sl.max() <= sl.min():
            # constant slice after normalization
            continue

        gx = sobel_h(sl)
        gy = sobel_v(sl)
        grad_mag_sq = gx * gx + gy * gy
        # Mean gradient energy = Tenengrad
        scores[i] = float(np.mean(grad_mag_sq))

    valid = np.isfinite(scores)
    if not np.any(valid):
        return np.nan, np.nan, scores
    return float(np.nanmean(scores[valid])), float(np.nanstd(scores[valid])), scores

import numpy as np

def brisque_volume_3d(
    volume: np.ndarray,
    axis: int = -1,
    clip_percentiles: tuple | None = None,
    auto_pad_min: int = 32,
):
    """
    Compute slice-wise BRISQUE on a 3D MRI volume and average along a chosen axis.
    Uses PIQ's BRISQUE implementation. Pads only when a slice is smaller than `auto_pad_min`
    in either spatial dimension (prints once per volume when padding occurs).

    Parameters
    ----------
    volume : np.ndarray
        3D complex or real array, shape (D, H, W) in any orientation.
        If complex, the magnitude (np.abs) is used.
    axis : int, default -1
        Axis along which to take slices/average (0, 1, or 2).
    clip_percentiles : (float, float) or None
        Optional robust intensity clipping before normalization, e.g. (1, 99).
    auto_pad_min : int, default 32
        Minimum spatial size (height/width) required; slices smaller than this
        are zero-padded to reach it.

    Returns
    -------
    mean_score : float
        Mean BRISQUE across valid slices (lower = better).
    std_score : float
        Standard deviation across valid slices.
    per_slice_scores : np.ndarray
        BRISQUE for each slice (np.nan where invalid).
    """
    # --- imports (kept inside so your module doesn't hard-require torch/piq on import) ---
    try:
        import torch
        import piq
    except Exception as e:
        raise ImportError(
            "This function requires PyTorch and PIQ. Install with:\n"
            "  pip install torch piq\n"
            f"Original import error: {e}"
        )

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    # Complex->magnitude and float32
    vol = np.abs(volume).astype(np.float32)

    # Optional robust clipping (helps with outliers at low field)
    if clip_percentiles is not None:
        p_lo, p_hi = np.percentile(vol, clip_percentiles)
        if p_hi > p_lo:
            vol = np.clip(vol, p_lo, p_hi)

    # Global min-max normalization to [0,1]
    vmin, vmax = float(np.nanmin(vol)), float(np.nanmax(vol))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        S = vol.shape[axis]
        return np.nan, np.nan, np.full(S, np.nan, dtype=float)
    vol = (vol - vmin) / (vmax - vmin + 1e-12)

    # Reorder so chosen axis is first: (S, H, W)
    vol_slices = np.moveaxis(vol, axis, 0)
    S = vol_slices.shape[0]
    scores = np.full(S, np.nan, dtype=float)

    padded_once = False

    for i in range(S):
        sl = vol_slices[i]
        if sl.ndim != 2 or sl.size == 0 or not np.isfinite(sl).all():
            continue
        if sl.max() <= sl.min():
            continue

        h, w = sl.shape
        if h < auto_pad_min or w < auto_pad_min:
            pad_h = max(0, auto_pad_min - h)
            pad_w = max(0, auto_pad_min - w)
            sl = np.pad(sl, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            if not padded_once:
                print(f"[BRISQUE] Zero-padded small slices to reach at least {auto_pad_min}×{auto_pad_min} (once per volume).")
                padded_once = True

        # PIQ expects torch tensor NCHW in [0,1]
        x = torch.from_numpy(sl).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        try:
            scores[i] = float(piq.brisque(x))
        except Exception:
            # If BRISQUE fails on a pathological slice, leave NaN
            pass

    valid = np.isfinite(scores)
    if not np.any(valid):
        return np.nan, np.nan, scores
    return float(np.nanmean(scores[valid])), float(np.nanstd(scores[valid])), scores
