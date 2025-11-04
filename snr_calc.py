"""
snr_calc.py
------------
Lightweight, consistent SNR calculation and visualization utility.

Defines:
    - clamp_origin()
    - center_cube_origin()
    - snr_from_volume()
    - plot_fs_rois()
    
All functions operate on 3D magnitude volumes (numpy arrays).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# -----------------------------------------------------------
# Utility: Clamp a cube origin to stay within volume bounds
# -----------------------------------------------------------
def clamp_origin(x0, y0, z0, shape, cube_size=10):
    """Ensure ROI origin stays inside volume."""
    nx, ny, nz = shape
    return (int(np.clip(x0, 0, nx - cube_size)),
            int(np.clip(y0, 0, ny - cube_size)),
            int(np.clip(z0, 0, nz - cube_size)))


def center_cube_origin(shape, cube_size=10):
    """Return center-aligned cube origin given volume shape."""
    nx, ny, nz = shape
    return clamp_origin(nx // 2 - cube_size // 2,
                        ny // 2 - cube_size // 2,
                        nz // 2 - cube_size // 2,
                        shape, cube_size)


# -----------------------------------------------------------
# Core SNR computation
# -----------------------------------------------------------
def snr_from_volume(vol_mag, cube_size=10, ddof=1,
                    signal_origin=None, noise_origin=(0, 0, 0),
                    plot=False, dataset_name=None):
    """
    Compute simple ROI-based SNR for a 3D magnitude volume.

    Parameters
    ----------
    vol_mag : np.ndarray
        3D magnitude image volume.
    cube_size : int, optional
        Side length of cubic ROIs (default = 10 voxels).
    ddof : int, optional
        Delta degrees of freedom for np.std (default = 1).
    signal_origin : tuple of int, optional
        (x0, y0, z0) of top-left-front corner for signal ROI.
        Defaults to cube centered in volume.
    noise_origin : tuple of int, optional
        (x0, y0, z0) of top-left-front corner for noise ROI.
    plot : bool, optional
        If True, calls plot_fs_rois() to visualize the ROIs.
    dataset_name : str, optional
        Label used in ROI plot titles.

    Returns
    -------
    snr_linear : float
        Linear SNR (mean signal / std noise).
    snr_db : float
        SNR in decibels (20 * log10(snr_linear)).
    signal_mean : float
        Mean of signal ROI.
    noise_std : float
        Std of noise ROI.
    sig_origin : tuple
        (x0, y0, z0) of signal ROI used.
    noi_origin : tuple
        (x0, y0, z0) of noise ROI used.
    """
    shape = vol_mag.shape
    if signal_origin is None:
        signal_origin = center_cube_origin(shape, cube_size)

    x0c, y0c, z0c = clamp_origin(*signal_origin, shape=shape, cube_size=cube_size)
    x0n, y0n, z0n = clamp_origin(*noise_origin, shape=shape, cube_size=cube_size)

    sig = vol_mag[x0c:x0c+cube_size, y0c:y0c+cube_size, z0c:z0c+cube_size]
    noi = vol_mag[x0n:x0n+cube_size, y0n:y0n+cube_size, z0n:z0n+cube_size]

    signal_mean = float(sig.mean())
    noise_std = float(noi.std(ddof=ddof))
    snr_linear = signal_mean / (noise_std if noise_std > 0 else np.nan)
    snr_db = 20.0 * np.log10(snr_linear) if np.isfinite(snr_linear) and snr_linear > 0 else np.nan

    if plot:
        plot_fs_rois(vol_mag, dataset_name or "Dataset",
                     (x0c, y0c, z0c), (x0n, y0n, z0n),
                     cube_size=cube_size)

    return (snr_linear, snr_db, signal_mean, noise_std,
            (x0c, y0c, z0c), (x0n, y0n, z0n))


# -----------------------------------------------------------
# ROI visualization
# -----------------------------------------------------------
def plot_fs_rois(vol_mag, dataset_name, sig_origin, noise_origin, cube_size=10):
    """Visualize the signal and noise ROIs on central slices."""
    x0c, y0c, z0c = sig_origin
    x0n, y0n, z0n = noise_origin
    z_sig = z0c + cube_size // 2
    z_noi = z0n + cube_size // 2

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(vol_mag[:, :, z_sig], cmap='gray', origin='lower')
    axs[0].set_title(f'{dataset_name}: Signal ROI (z={z_sig})')
    axs[0].add_patch(patches.Rectangle((y0c, x0c), cube_size, cube_size,
                                       linewidth=2, edgecolor='red', facecolor='none'))
    axs[0].axis('off')

    axs[1].imshow(vol_mag[:, :, z_noi], cmap='gray', origin='lower')
    axs[1].set_title(f'{dataset_name}: Noise ROI (z={z_noi})')
    axs[1].add_patch(patches.Rectangle((y0n, x0n), cube_size, cube_size,
                                       linewidth=2, edgecolor='blue', facecolor='none'))
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
