import multiprocessing as mp
mp.set_start_method("spawn", force=True)


# CS Recon Functions for ULF MRI Experiments - Updated by D Waddington 2022
import numpy as np
import matplotlib.pyplot as plt

import sigpy as sp
sp.config.use_cuda = False
sp.config.use_opencl = False
import sigpy.mri as mr
import sigpy.plot as pl

# functions for metrics adapted from .py file provided by Efrat
from metrics import calc_NRMSE as nrmse
from metrics import calc_SSIM as ssim

# the main reconstruction function
def ulfl1recon(ksp,mask,lamda,iter=30, mps=float('nan')):  
    maskedksp = applyMask(ksp,mask)
    if  np.any(np.isnan(mps)):
      if maskedksp.shape[0] == 1:  #single-channel
        mps = np.ones(maskedksp.shape,dtype=complex)
      else:  #multi-channel
        mps = mr.app.EspiritCalib(maskedksp, calib_width=20, kernel_width=4, show_pbar=False).run()
    l1wav = mr.app.L1WaveletRecon(maskedksp, mps, lamda, show_pbar=False)
    l1wav.alg.max_iter = iter
    img_l1wav = l1wav.run()
    return img_l1wav

# the main reconstruction function but with TV regularization
def ulfTVrecon(ksp,mask,lamda,iter=30, mps=float('nan')):  
    maskedksp = applyMask(ksp,mask)
    if  np.any(np.isnan(mps)):
      if maskedksp.shape[0] == 1:  #single-channel
        mps = np.ones(maskedksp.shape,dtype=complex)
      else:  #multi-channel
        mps = mr.app.EspiritCalib(maskedksp, calib_width=20, kernel_width=4, show_pbar=False).run()
    TVrcn = mr.app.TotalVariationRecon(maskedksp, mps, lamda, show_pbar=False)
    TVrcn.alg.max_iter = iter
    img_TVrcn = TVrcn.run()
    return img_TVrcn

# retrospective undersampling function that applies a mask to fully sampled 3D kspace
def applyMask(ksp,mask):
    if len(ksp.shape) == 3:  #making single-channel the same shape as multi-channel
        ksp_channels = np.reshape(ksp,(1,ksp.shape[0],ksp.shape[1],ksp.shape[2]))
    elif len(ksp.shape) == 4:  #multi-channel
        ksp_channels = ksp
    
    maskedksp = np.zeros(ksp_channels.shape,dtype = 'complex64')
    
    for i in range(ksp_channels.shape[0]):
        for j in range(ksp_channels.shape[2]):
            maskedksp[i,:,j,:] = np.multiply(np.squeeze(ksp_channels[i,:,j,:]),mask)
    
    return maskedksp

# Mask generation using the poisson disc function from SigPy but with a calibration region added
def poissonDiscSigpy(imSize,accel,in_seed,calib_size=10):
    if accel == 1:
      maskpoissonsquare = np.ones((imSize[0],imSize[2]))
    else:
      maskpoissonsquare = sp.mri.poisson([imSize[0],imSize[2]], accel, tol = 0.1, seed = in_seed);
      maskpoissonsquare[int(imSize[0]/2-calib_size/2+1):int(imSize[0]/2+calib_size/2+1),int(imSize[2]/2-calib_size/2+1):int(imSize[2]/2+calib_size/2+1)] = np.ones((calib_size,calib_size),dtype=complex);
    return maskpoissonsquare

# finds the optimal regularization parameter lamda for reconstruction
def find_lamda_mask(ksp, GT, mps=float('nan'), calib_size=10, show_plot=True, iter = 30):
    
    if len(ksp.shape) == 3: # if single channel reshape to multi-channel with a leading 1.
        ksp = np.reshape(ksp,(1,ksp.shape[0],ksp.shape[1],ksp.shape[2]))
    
    lamda_vals = np.array([1E-4, 2E-4, 5E-4, 1E-3, 2E-3, 5E-3, 1E-2, 2E-2, 5E-2, 1E-1, 2E-1, 5E-1, 1, 2, 5, 10, 20])
    

    nrmse_vals = np.zeros((lamda_vals.size))
    ssim_vals = np.zeros((lamda_vals.size))

    mask_metrics = abs(GT) < 1e-4
    GT_masked = np.copy(GT)
    GT_masked[mask_metrics] = 0

    #mask_metrics = np.ma.getmask(np.ma.masked_less(abs(GT),0.0001))
    #GT[mask_metrics]=0
    
    i = 0
    for lamda in lamda_vals:        
        img_l1wav = ulfl1recon(ksp,np.ones((ksp.shape[1],ksp.shape[3]),dtype='complex64'),lamda,iter,mps)
        #img_l1wav = ulfl1recon(ksp,np.squeeze(masks[:,:,j]),lamda,iter)
        img_l1wav[mask_metrics]=0
        nrmse_vals[i] = nrmse(abs(img_l1wav[:,:,:]),abs(GT[:,:,:]))
        ssim_vals[i] = ssim(abs(img_l1wav[:,:,:]),abs(GT[:,:,:]))
        i = i + 1

    lamda_opt = lamda_vals[np.argmin(nrmse_vals)]
    
    if show_plot == True:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        line1 = ax1.plot(lamda_vals,nrmse_vals, 'ob-')
        ax1.set(title='Lamda Opt',
            ylabel='NRMSE',
            xlabel='lamda')
        ax1.set_xscale('log')
        ax1.legend("NRMSE")

        ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
        line2 =ax2.plot(lamda_vals,ssim_vals, 'or-')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set(ylabel='SSIM')
        ax2.legend("SSIM")

        plt.show()
        print('Minimum NRMSE for lamda value of ',lamda_opt)
    
    return lamda_opt

# finds the optimal number of iterations for reconstruction
def find_iter_mask(ksp,GT, lamda_opt, mps=float('nan'), show_plot=True):
    
    #iter_vals = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 400])
    iter_vals = np.array([1, 2, 3, 4, 5, 10, 20, 30, 50, 70, 100, 150, 200])
    nrmse_iter_vals = np.zeros((iter_vals.size))
    ssim_iter_vals = np.zeros((iter_vals.size))
    
    #mask_metrics = np.ma.getmask(np.ma.masked_less(abs(GT),0.15))
    #mask_metrics = np.ma.getmask(np.ma.masked_less(abs(GT),0.0001))
    #GT[mask_metrics]=0
    mask_metrics = abs(GT) < 1e-4
    GT_masked = np.copy(GT)
    GT_masked[mask_metrics] = 0

    
    i = 0
    for iter in iter_vals:
        img_l1wav = ulfl1recon(ksp,np.ones((ksp.shape[1],ksp.shape[3]),dtype='complex64'),lamda_opt,iter,mps)
        #img_l1wav = ulfl1recon(ksp,mask,lamda_opt,iter);
        img_l1wav[mask_metrics] = 0
        nrmse_iter_vals[i] = nrmse(abs(img_l1wav[:,:,:]),abs(GT[:,:,:]))
        ssim_iter_vals[i] = ssim(abs(img_l1wav[:,:,:]),abs(GT[:,:,:]))
        i = i + 1

    iter_opt = iter_vals[np.argmin(nrmse_iter_vals)]
    
    if show_plot == True:
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(111)
        line1 = ax1.plot(iter_vals,nrmse_iter_vals, 'ob-')
        ax1.set(title='Iter Opt',
            ylabel='NRMSE',
            xlabel='Iterations')
        #ax1.set_xscale('log')
        ax1.legend("NRMSE")

        ax2 = fig2.add_subplot(111, sharex=ax1, frameon=False)
        line2 =ax2.plot(iter_vals,ssim_iter_vals, 'or-')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set(ylabel='SSIM')
        ax2.legend("SSIM")

        plt.show()

        print('Minimum NRMSE for iter value of ',iter_opt)
    
    return iter_opt


# to combine multi-channel data via a sensitvity map (mps)
def coil_combine(imgs_mc,mps):
    img_cc = np.sum(np.multiply(np.conj(mps),imgs_mc),axis=0)
    return img_cc


# code to add white gaussian noise
# author - Mathuranathan Viswanathan (gaussianwaves.com)
# This code is part of the book Digital Modulations using Python

from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

def awgn(s,SNRdB,L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
        """
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return r

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm  # Optional: for showing progress
import pickle

# def _recon_for_lamda(args):
#     lamda, ksp, mps, GT_masked, mask_metrics, iter = args



#     for name, obj in [('ksp', ksp), ('mps', mps),
#                       ('GT_masked', GT_masked), ('mask_metrics', mask_metrics)]:
#         try:
#             pickle.dumps(obj)
#         except Exception as e:
#             print(f"[Pickle FAIL] {name}: {type(obj)} → {e}")
    
#     img = ulfl1recon(ksp, np.ones((ksp.shape[1],ksp.shape[3]),dtype='complex64'), lamda, iter=iter, mps=mps)
#     img[mask_metrics] = 0
#     return lamda, nrmse(abs(img), abs(GT_masked)), ssim(abs(img), abs(GT_masked))


# def find_lamda_mask(ksp, GT, mps, calib_size=10, show_plot=False, iter=30, use_tqdm=False):
#     lamda_vals = np.array([
#         1E-4, 2E-4, 5E-4, 1E-3, 2E-3, 5E-3, 1E-2, 2E-2, 5E-2,
#         1E-1, 2E-1, 5E-1, 1, 2, 5, 10, 20
#     ])

#     ksp = np.asarray(ksp)
#     mps = np.asarray(mps)
#     GT = np.asarray(GT)

#     if ksp.ndim == 3:
#         ksp = ksp[np.newaxis, ...]

#     mask_metrics = abs(GT) < 1e-4
#     GT_masked = np.copy(GT)
#     GT_masked[mask_metrics] = 0

#     # Package all arguments explicitly to avoid partial
#     job_args = [(lamda, ksp, mps, GT_masked, mask_metrics, iter) for lamda in lamda_vals]

#     for i, args in enumerate(job_args):
#         try:
#             pickle.dumps(args)
#         except Exception as e:
#             print(f"[Pickle FAIL] job_args[{i}] → {e}")
    
#     with Pool(processes=10) as pool:
#         if use_tqdm:
#             results = list(tqdm(pool.imap(_recon_for_lamda, job_args), total=len(job_args)))
#         else:
#             results = pool.map(_recon_for_lamda, job_args)

#     lamdas, nrmse_vals, ssim_vals = zip(*results)
#     lamda_opt = lamdas[np.argmin(nrmse_vals)]

#     if show_plot:
#         fig, ax1 = plt.subplots()
#         ax1.plot(lamdas, nrmse_vals, 'bo-', label='NRMSE')
#         ax1.set_xscale('log')
#         ax1.set_xlabel('Lamda')
#         ax1.set_ylabel('NRMSE')
#         ax2 = ax1.twinx()
#         ax2.plot(lamdas, ssim_vals, 'ro-', label='SSIM')
#         ax2.set_ylabel('SSIM')
#         plt.title(f'Lamda Optimization (opt = {lamda_opt})')
#         fig.tight_layout()
#         plt.show()

#     return lamda_opt

