import numpy as np
import sigpy as sp
from ulf_recon_fns import coil_combine

from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Function for reconstructing 3D kspace low field data with AUTOMAP
def automapRecon(ksp,mps,model_real_dir,model_imag_dir):

    ksp = np.moveaxis(ksp,-2,-1)
    #ksp is input to the functions

    # ensure single-channel data and multi-channel data have the same format
    if len(ksp.shape) == 3:  #single-channel
        ksp_channels = np.reshape(ksp,(1,ksp.shape[0],ksp.shape[1],ksp.shape[2]))
    elif len(ksp.shape) == 4:  #multi-channel
        ksp_channels = ksp

    # AUTOMAP is only trained for 2D, so we reduce perform an FFT in the z-direction
    ksp_hybrid = sp.ifft(ksp_channels,axes=[3])

    kspflat = np.reshape(ksp_hybrid,(ksp_hybrid.shape[0],-1,ksp_hybrid.shape[3]),order='F')
    kspflat = np.swapaxes(kspflat,1,2)
    kspflat = np.reshape(kspflat,(ksp_hybrid.shape[0]*ksp_hybrid.shape[3],-1),order='F')

    scale = 1
    kspflat = kspflat*scale

    amapinput = np.concatenate((np.real(kspflat),np.imag(kspflat)),axis=1)

    # # loading models for inference
    # model_real = keras.models.load_model(model_real_dir)
    # model_imag = keras.models.load_model(model_imag_dir)
        # Load models only if strings are passed
    if isinstance(model_real_dir, str):
        model_real = keras.models.load_model(model_real_dir)
    else:
        model_real = model_real_dir

    if isinstance(model_imag_dir, str):
        model_imag = keras.models.load_model(model_imag_dir)
    else:
        model_imag = model_imag_dir
    

    # run inference on real and imaginary channels then combine into one complex array
    c_2, predictions_real = model_real(amapinput, training=False)
    c_2, predictions_imag = model_imag(amapinput, training=False)
    predictions = np.array(predictions_real + np.multiply(predictions_imag,1j))/scale
    
    
    padding=4
    
    preds_img = predictions.reshape(predictions.shape[0],ksp_channels.shape[1]+2*padding,ksp_channels.shape[2]+2*padding, order = 'F')
    preds_img_crop = preds_img[:,padding:-padding,padding:-padding]
    
    # mc - multi-channel
    volume_mc = np.reshape(preds_img_crop,(ksp_channels.shape[0],ksp_channels.shape[3],ksp_channels.shape[1],ksp_channels.shape[2]), order = 'F')    
    volume_mc = np.moveaxis(volume_mc,1,2)
    
    
    training_scale = 1
    volume_mc = volume_mc/training_scale
    
    #coil combination using sensitivity maps
    volume = coil_combine(volume_mc,mps)
    
    return volume, volume_mc