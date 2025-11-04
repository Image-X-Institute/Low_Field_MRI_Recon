import numpy as np
import matplotlib.pyplot as plt

# functions for metrics adapted from .py file provided by Efrat
from metrics import calc_NRMSE as nrmse
from metrics import calc_SSIM as ssim

# to plot nice brain slices
def brainSlicePlot(volrecon, start_idx, end_idx, plot_type = 'mag', cmax = 1.5):
    imgflat = np.swapaxes(np.swapaxes(volrecon,1,2),0,2)
    #print(imgflat.shape)
    imgflat = np.reshape(imgflat[:,start_idx:end_idx,:],(imgflat.shape[0],-1))
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    if plot_type == 'mag':
      im = ax.imshow(abs(imgflat),cmap='gray', vmin = 0, vmax = cmax)
    elif plot_type == 'phase':
      im = ax.imshow(np.angle(imgflat),cmap='plasma')
    fig.colorbar(im,fraction=0.01, pad=0.06)
    fig.set_size_inches(16,16)

    asp_ratio = (volrecon.shape[0]*3.5)/(volrecon.shape[1]*2.5)*0.9
    ax.set_aspect(asp_ratio)
    return fig


# compare a two reconstructed images to a GT    
def comparePlot(GT,vol1,vol1title,vol2,vol2title,slc,scale_err = 1):
    
    GT = np.swapaxes(GT,0,1)
    vol1 = np.swapaxes(vol1,0,1)
    vol2 = np.swapaxes(vol2,0,1)
    
    # RO 64, 2.5 mm
    # PE1 75, 3.5mm
    # PE2 15, 8 mm
    asp_ratio = (GT.shape[0]*3.5)/(GT.shape[1]*2.5)
    
    fig, axs = plt.subplots(2, 3,figsize = (7,5))
    
    max_GT = np.amax(np.abs(GT[:,:,slc]))
    
    im = axs[0, 0].imshow(np.abs(GT[:,:,slc]), cmap= 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[0, 0].set_title('Ground Truth')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_aspect(asp_ratio)
    #axs[0, 0].set_clim(0, (1.05*max_GT))
    
    im = axs[0, 1].imshow(np.abs(vol1[:,:,slc]), cmap = 'gray', vmin = 0,vmax = 1.05*max_GT)
    axs[0, 1].set_title(vol1title)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_aspect(asp_ratio)
    
    im = axs[1, 1].imshow(scale_err*np.abs(np.abs(vol1[:,:,slc])-np.abs(GT[:,:,slc])), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[1, 1].set_title(vol1title + ' Diff')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_aspect(asp_ratio)
    axs[1, 1].set_xlabel('SSIM ='+str(round(ssim(np.abs(vol1[:,:,:]),np.abs(GT[:,:,:])),2))
                         +'\n NRMSE ='+str(round(nrmse(np.abs(vol1[:,:,:]),np.abs(GT[:,:,:])),3)))

    
    im = axs[0, 2].imshow(np.abs(vol2[:,:,slc]), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[0, 2].set_title(vol2title)
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_aspect(asp_ratio)
    
    im = axs[1, 2].imshow(scale_err*np.abs((np.abs(vol2[:,:,slc])-np.abs(GT[:,:,slc]))), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[1, 2].set_title(vol2title + ' Diff')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_aspect(asp_ratio)
    axs[1, 2].set_xlabel('SSIM ='+str(round(ssim(np.abs(vol2[:,:,:]),np.abs(GT[:,:,:])),2))
                         +'\n NRMSE ='+str(round(nrmse(np.abs(vol2[:,:,:]),np.abs(GT[:,:,:])),3)))
    
    axs[1, 0].axis('off')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
    
    print('Difference Maps are scaled by ',scale_err)
    
    return fig

# compare a two reconstructed images to a GT    
def compareFourPlot(GT,vol1,vol1title,vol2,vol2title,vol3,vol3title,vol4,vol4title,slc,scale_err = 1):
    GT = np.swapaxes(GT,0,1)
    vol1 = np.swapaxes(vol1,0,1)
    vol2 = np.swapaxes(vol2,0,1)
    vol3 = np.swapaxes(vol3,0,1)
    vol4 = np.swapaxes(vol4,0,1)
    
    # RO 64, 2.5 mm
    # PE1 75, 3.5mm
    # PE2 15, 8 mm
    asp_ratio = (GT.shape[0]*3.5)/(GT.shape[1]*2.5)
    
    fig, axs = plt.subplots(2, 5,figsize = (11,5))
    
    max_GT = np.amax(np.abs(GT[:,:,slc]))
    
    im = axs[0, 0].imshow(np.abs(GT[:,:,slc]), cmap= 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[0, 0].set_title('Ground Truth')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_aspect(asp_ratio)
    #axs[0, 0].set_clim(0, (1.05*max_GT))
    
    im = axs[0, 1].imshow(np.abs(vol1[:,:,slc]), cmap = 'gray', vmin = 0,vmax = 1.05*max_GT)
    axs[0, 1].set_title(vol1title)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_aspect(asp_ratio)
    
    im = axs[1, 1].imshow(scale_err*np.abs(np.abs(vol1[:,:,slc])-np.abs(GT[:,:,slc])), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    #axs[1, 1].set_title(vol1title + ' Diff')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_aspect(asp_ratio)
    axs[1, 1].set_xlabel('SSIM ='+str(round(ssim(np.abs(vol1[:,:,:]),np.abs(GT[:,:,:])),2))
                         +'\n NRMSE ='+str(round(nrmse(np.abs(vol1[:,:,:]),np.abs(GT[:,:,:])),3)))
    axs[1, 1].set_ylabel('Difference',fontsize=11)

    
    im = axs[0, 2].imshow(np.abs(vol2[:,:,slc]), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[0, 2].set_title(vol2title)
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_aspect(asp_ratio)
    
    im = axs[1, 2].imshow(scale_err*np.abs((np.abs(vol2[:,:,slc])-np.abs(GT[:,:,slc]))), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    #axs[1, 2].set_title(vol2title + ' Diff')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_aspect(asp_ratio)
    axs[1, 2].set_xlabel('SSIM ='+str(round(ssim(np.abs(vol2[:,:,:]),np.abs(GT[:,:,:])),2))
                         +'\n NRMSE ='+str(round(nrmse(np.abs(vol2[:,:,:]),np.abs(GT[:,:,:])),3)))
    
    im = axs[0, 3].imshow(np.abs(vol3[:,:,slc]), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[0, 3].set_title(vol3title)
    axs[0, 3].set_xticks([])
    axs[0, 3].set_yticks([])
    axs[0, 3].set_aspect(asp_ratio)
    
    im = axs[1, 3].imshow(scale_err*np.abs((np.abs(vol3[:,:,slc])-np.abs(GT[:,:,slc]))), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    #axs[1, 3].set_title(vol3title + ' Diff')
    axs[1, 3].set_xticks([])
    axs[1, 3].set_yticks([])
    axs[1, 3].set_aspect(asp_ratio)
    axs[1, 3].set_xlabel('SSIM ='+str(round(ssim(np.abs(vol3[:,:,:]),np.abs(GT[:,:,:])),2))
                         +'\n NRMSE ='+str(round(nrmse(np.abs(vol3[:,:,:]),np.abs(GT[:,:,:])),3)))
    
    im = axs[0, 4].imshow(np.abs(vol4[:,:,slc]), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    axs[0, 4].set_title(vol4title)
    axs[0, 4].set_xticks([])
    axs[0, 4].set_yticks([])
    axs[0, 4].set_aspect(asp_ratio)
    
    im = axs[1, 4].imshow(scale_err*np.abs((np.abs(vol4[:,:,slc])-np.abs(GT[:,:,slc]))), cmap = 'gray', vmin = 0, vmax = 1.05*max_GT)
    #axs[1, 4].set_title(vol4title + ' Diff')
    axs[1, 4].set_xticks([])
    axs[1, 4].set_yticks([])
    axs[1, 4].set_aspect(asp_ratio)
    axs[1, 4].set_xlabel('SSIM ='+str(round(ssim(np.abs(vol4[:,:,:]),np.abs(GT[:,:,:])),2))
                         +'\n NRMSE ='+str(round(nrmse(np.abs(vol4[:,:,:]),np.abs(GT[:,:,:])),3)))
    
    
    axs[1, 0].axis('off')
    
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(hspace=-0.3)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
    
    print('Difference Maps are scaled by ',scale_err)
    
    return fig

def compare_n_plot(GT, volumes, titles, slc, scale_err=1.0):
    """
    Compare N reconstructions to a ground truth volume and plot their slices + error maps.

    Args:
        GT (np.ndarray): Ground truth 3D volume [H, W, D] (complex or real).
        volumes (list of np.ndarray): List of reconstructed 3D volumes.
        titles (list of str): List of corresponding titles.
        slc (int): Slice index to display.
        scale_err (float): Scale factor for error map visualization.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
    """
    n = len(volumes)
    assert len(titles) == n, "Number of titles must match number of volumes"

    # Transpose to match plotting convention
    GT = np.swapaxes(GT, 0, 1)
    volumes = [np.swapaxes(vol, 0, 1) for vol in volumes]

    # RO 64, 2.5 mm | PE1 75, 3.5 mm
    asp_ratio = (GT.shape[0] * 3.5) / (GT.shape[1] * 2.5)
    max_GT = np.amax(np.abs(GT[:, :, slc]))

    fig, axs = plt.subplots(2, n + 1, figsize=(2.2 * (n + 1), 5))

    # Plot GT
    axs[0, 0].imshow(np.abs(GT[:, :, slc]), cmap='gray', vmin=0, vmax=1.05 * max_GT)
    axs[0, 0].set_title('Ground Truth')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_aspect(asp_ratio)
    axs[1, 0].axis('off')

    # Plot reconstructions and differences
    for i in range(n):
        vol = (volumes[i])
        title = titles[i]

        axs[0, i + 1].imshow(np.abs(vol[:, :, slc]), cmap='gray', vmin=0, vmax=1.05 * max_GT)
        axs[0, i + 1].set_title(title)
        axs[0, i + 1].set_xticks([])
        axs[0, i + 1].set_yticks([])
        axs[0, i + 1].set_aspect(asp_ratio)

        diff_map = scale_err * np.abs(np.abs(vol[:, :, slc]) - np.abs(GT[:, :, slc]))
        im = axs[1, i + 1].imshow(diff_map, cmap='gray', vmin=0, vmax=1.05 * max_GT)
        axs[1, i + 1].set_xticks([])
        axs[1, i + 1].set_yticks([])
        axs[1, i + 1].set_aspect(asp_ratio)

        # Use your custom metrics
        ssim_val = ssim(np.abs(vol), np.abs(GT))
        nrmse_val = nrmse(np.abs(vol), np.abs(GT))
        axs[1, i + 1].set_xlabel(f'SSIM = {ssim_val:.2f}\nNRMSE = {nrmse_val:.3f}')

    # Colorbar for difference maps
    fig.subplots_adjust(right=0.85, hspace=-0.3)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()
    print('Difference Maps are scaled by', scale_err)
    return fig

import numpy as np
import matplotlib.pyplot as plt
from dipy.denoise.noise_estimate import estimate_sigma
from PIL import Image

# --- Try SSIM_PIL first; fall back to skimage if unavailable ---
try:
    from SSIM_PIL import compare_ssim  # pip install SSIM-PIL
except ImportError:
    # Fallback: skimage (will be slower and slightly different numerically)
    from skimage.metrics import structural_similarity as _ssim_skimage

    def compare_ssim(im1_PIL, im2_PIL):
        im1 = np.array(im1_PIL, dtype=np.uint8)
        im2 = np.array(im2_PIL, dtype=np.uint8)
        # data_range=255 for uint8 images
        return _ssim_skimage(im1, im2, data_range=255)

# ---------------------------------------------------------------------
# Local variance-stabilizing transform for Rician / non-central-χ noise
# (Foi et al., IEEE TMI 2013; simple analytic approximation)
# ---------------------------------------------------------------------
def ncchi_vst(x, sigma=1.0, L=1):
    """
    Approximate variance-stabilizing transform for MRI magnitude data.
    x : ndarray (magnitude)
    sigma : noise std dev (per channel)
    L : number of coils (L=1 → Rician; L>1 → non-central-χ)
    y = sqrt(max(x^2 - 2 L sigma^2, eps))
    """
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    sigma = max(float(sigma), 1e-8)
    bias = 2.0 * int(L) * (sigma ** 2)
    eps = 1e-8
    return np.sqrt(np.maximum(x * x - bias, eps))


# ===========================================================
# ---- Internal metric definitions (from your metrics.py) ----
# ===========================================================

# def calc_NRMSE(I_pred, I_true):
#     """Normalized RMSE using max(I_true) as baseline (0 always min)."""
#     I_true = np.reshape(I_true, (-1,))
#     I_pred = np.reshape(I_pred, (-1,))
#     MSE = np.mean(np.square(I_true - I_pred))
#     RMSE = np.sqrt(MSE)
#     rr = np.max(I_true)  # ensures baseline = 0
#     return RMSE / (rr + 1e-8)
def calc_NRMSE(I_pred, I_true):
    I_true = np.reshape(I_true, (-1,))
    I_pred = np.reshape(I_pred, (-1,))
    MSE = np.mean((I_true - I_pred)**2)
    RMSE = np.sqrt(MSE)
    rr = np.max(I_true) - np.min(I_true)  # <- original behavior
    return RMSE / (rr + 1e-8)



def calc_SSIM(I_pred, I_true):
    """Slice-wise SSIM average (replicates your PIL-based version)."""
    if I_true.ndim == 3:
        I_true_vol = np.abs(I_true)
        I_pred_vol = np.abs(I_pred)
    else:
        I_true_vol = np.abs(I_true[:, None])
        I_pred_vol = np.abs(I_pred[:, None])

    no_slices = I_true_vol.shape[2]
    SSIM_vals = np.zeros(no_slices, dtype=np.float32)

    for slc in range(no_slices):
        I_true_slice = np.squeeze(I_true_vol[:, :, slc])
        I_pred_slice = np.squeeze(I_pred_vol[:, :, slc])
        # convert to 8-bit for PIL/skimage SSIM
        im1_uint8 = (I_true_slice * 255.0 / (np.max(I_true_slice) + 1e-8)).astype('uint8')
        im2_uint8 = (I_pred_slice * 255.0 / (np.max(I_pred_slice) + 1e-8)).astype('uint8')
        im1_PIL = Image.fromarray(im1_uint8)
        im2_PIL = Image.fromarray(im2_uint8)
        SSIM_vals[slc] = compare_ssim(im1_PIL, im2_PIL)

    return float(np.mean(SSIM_vals))


# ===========================================================
# ---- Main plotting + metric function ----
# ===========================================================

def compare_n_plot_dipy(
    GT,
    volumes,
    titles,
    slc,
    scale_err=1.0,
    mask_metrics=None,
    use_mask_for_metrics=True,     # Toggle: exclude masked voxels from metrics
    coils_L=1,
    sigma_ref=None,
    show_noise_metrics=True,
    use_intensity_p95_norm=False
):
    """
    Compare reconstructions to ground truth, compute NRMSE/SSIM/VST-RMSE,
    and visualize with consistent normalization and masking.

    Args:
        GT (np.ndarray): Ground truth 3D volume [H, W, D].
        volumes (list of np.ndarray): List of reconstructed volumes.
        titles (list of str): Titles corresponding to reconstructions.
        slc (int): Slice index for visualization.
        scale_err (float): Scale factor for difference maps.
        mask_metrics (np.ndarray): Boolean mask (True = exclude from metrics).
        use_mask_for_metrics (bool): If False, mask is ignored.
        coils_L (int): Number of coils (1=Rician, >1=nc-chi).
        sigma_ref (float): Noise std; if None, auto-estimated with DIPY.
        show_noise_metrics (bool): If True, compute VST-RMSE.
        use_intensity_p95_norm (bool): Normalize by 95th percentile intensity.

    Returns:
        fig (matplotlib.figure.Figure)
    """
    n = len(volumes)
    assert len(titles) == n, "titles length must match number of volumes"

    # --- Reorient for consistent plotting (match your original helper) ---
    GT = np.swapaxes(GT, 0, 1)
    volumes = [np.swapaxes(v, 0, 1) for v in volumes]
    GT_mag = np.abs(GT)

    # --- NEW: transpose the mask to match the swapped orientation ---
    mask_swapped = None
    if mask_metrics is not None:
        mask_swapped = np.swapaxes(mask_metrics, 0, 1).astype(bool)
        if mask_swapped.shape != GT_mag.shape:
            raise ValueError(
                f"mask_metrics (after swap) shape {mask_swapped.shape} "
                f"does not match GT shape {GT_mag.shape}"
            )

    # --- Normalization (optional) ---
    if use_intensity_p95_norm:
        p95 = np.percentile(GT_mag, 95)
        scale = (p95 + 1e-8)
        GT_mag = GT_mag / scale
        volumes = [np.abs(v) / scale for v in volumes]
    else:
        volumes = [np.abs(v) for v in volumes]

    # --- Noise estimation ---
    sigma = float(sigma_ref) if sigma_ref is not None else estimate_sigma(GT_mag, N=coils_L)
    sigma = max(sigma, 1e-8)

    # --- Plot setup ---
    asp_ratio = (GT.shape[0] * 3.5) / (GT.shape[1] * 2.5)
    vmax = np.percentile(GT_mag, 99)
    fig, axs = plt.subplots(2, n + 1, figsize=(2.2 * (n + 1), 5))
    fig.subplots_adjust(right=0.85, hspace=-0.25)

    # --- Helper: mask logic ---
    def _mask_for_metrics_like_exclusion(vol, ref, mask):
        if mask is None or not use_mask_for_metrics:
            return vol.flatten(), ref.flatten()
        mask_inv = ~mask
        return vol[mask_inv].flatten(), ref[mask_inv].flatten()

    # --- Plot ground truth ---
    axs[0, 0].imshow(GT_mag[:, :, slc], cmap="gray", vmin=0, vmax=vmax)
    axs[0, 0].set_title("Ground Truth")
    axs[0, 0].set_xticks([]); axs[0, 0].set_yticks([])
    axs[0, 0].set_aspect(asp_ratio)
    axs[1, 0].axis("off")

    # --- Iterate over reconstructions ---
    for i, vol_mag in enumerate(volumes):
        title = titles[i]
        axs[0, i + 1].imshow(vol_mag[:, :, slc], cmap="gray", vmin=0, vmax=vmax)
        axs[0, i + 1].set_title(title)
        axs[0, i + 1].set_xticks([]); axs[0, i + 1].set_yticks([])
        axs[0, i + 1].set_aspect(asp_ratio)

        # Difference map
        diff_map = scale_err * np.abs(vol_mag[:, :, slc] - GT_mag[:, :, slc])
        im = axs[1, i + 1].imshow(diff_map, cmap="gray", vmin=0, vmax=vmax)
        axs[1, i + 1].set_xticks([]); axs[1, i + 1].set_yticks([])
        axs[1, i + 1].set_aspect(asp_ratio)

        # ---- Metric computation ----
        lines = []

        # SSIM: slice-wise; zero masked voxels only if using mask
        vol_ssim = np.copy(vol_mag)
        gt_ssim = np.copy(GT_mag)
        if mask_swapped is not None and use_mask_for_metrics:
            vol_ssim[mask_swapped] = 0
            gt_ssim[mask_swapped] = 0
        ssim_val = calc_SSIM(vol_ssim, gt_ssim)
        lines.append(f"SSIM = {ssim_val:.3f}")
        
        # NRMSE (your calc_NRMSE; range-based inside calc if you set it that way)
        vol_flat, gt_flat = _mask_for_metrics_like_exclusion(vol_mag, GT_mag, mask_swapped)
        nrmse_val = calc_NRMSE(vol_flat, gt_flat)
        lines.append(f"NRMSE = {nrmse_val:.3f}")



        # VST-RMSE (noise-aware)
        if show_noise_metrics:
            GT_safe = np.nan_to_num(GT_mag, nan=0.0, posinf=0.0, neginf=0.0)
            VOL_safe = np.nan_to_num(vol_mag, nan=0.0, posinf=0.0, neginf=0.0)
            GT_vst = ncchi_vst(GT_safe, sigma=sigma, L=coils_L)
            VOL_vst = ncchi_vst(VOL_safe, sigma=sigma, L=coils_L)

            vol_vst, gt_vst = _mask_for_metrics_like_exclusion(VOL_vst, GT_vst, mask_swapped)
            if (np.any(~np.isfinite(gt_vst)) or np.ptp(gt_vst) == 0):
                nrmse_vst = np.nan
            else:
                nrmse_vst = calc_NRMSE(vol_vst, gt_vst)
            lines.append(f"VST-RMSE = {nrmse_vst if np.isfinite(nrmse_vst) else 0:.3f}")

        axs[1, i + 1].set_xlabel("\n".join(lines), fontsize=8)

    # Colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    print("Difference Maps scaled by", scale_err)

    return fig
