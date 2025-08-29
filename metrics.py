import numpy as np
from SSIM_PIL import compare_ssim
from PIL import Image
# E. Shimron, 2021. Modified by D. Waddington, 2022.

  
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