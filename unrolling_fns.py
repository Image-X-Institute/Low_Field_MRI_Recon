import torch 
import torch.nn as nn
import numpy as np
import scipy.io as scio

from trained_models.Unrolling.UnrollNet_DC import * 
import time 
from collections import OrderedDict
import sigpy as sp

from ulf_recon_fns import coil_combine

def zero_filling(x, factor = 16):
    H = x.size(0)
    W = x.size(1)
    D = x.size(2)

    newH = torch.ceil(torch.tensor(H / factor)) * factor
    newW = torch.ceil(torch.tensor(W / factor)) * factor
    newD = torch.ceil(torch.tensor(D / factor)) * factor

    tmp = torch.zeros(newH.int(), newW.int(),newD.int())

    pos = torch.zeros(2, 3)

    a = torch.ceil((newH - H) / 2)
    b = torch.ceil((newW - W) / 2)
    e = torch.ceil((newD - D) / 2)

    c = (a + H)
    d = (b + W)
    f = (e + D)

    a = a.int()
    b = b.int()
    c = c.int()
    d = d.int()
    e = e.int()
    f = f.int()

    tmp[a:c, b:d, e:f] = x

    pos[0, 0] = a
    pos[0, 1] = b
    pos[1, 0] = c
    pos[1, 1] = d
    pos[0, 2] = e
    pos[1, 2] = f

    return tmp, pos


def zero_removing(x, pos):

    a = pos[0, 0]
    b = pos[0, 1]
    c = pos[1, 0]
    d = pos[1, 1]
    e = pos[0, 2]
    f = pos[1, 2]

    a = a.int()
    b = b.int()
    c = c.int()
    d = d.int()
    e = e.int()
    f = f.int()

    x = x[a:c, b:d, e:f]
    return x

def unrollingRecon(inputKspace,mask,mps,model_pth):
    with torch.no_grad():        

            
            ## load trained network 
            state_dict = torch.load(model_pth, map_location=lambda storage, loc: storage)
            # create new OrderedDict that does not contain `module.`

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[0:6]=='module':
                    name = k[7:] # remove `module.`
                else:
                    name=k  #DEJW - not sure why module is gone
                #
                new_state_dict[name] = v
            # load params
            
            #print(len(state_dict))
            #print(state_dict.keys())
            mx_size = inputKspace.shape
            #print(mx_size)
            
            ## the state_dict contains parameters, 14 items per unroll layer
            Unrolling_chi = UnrollNet(int(len(state_dict)/14), (mx_size[1], mx_size[3]), ini_flag = False)
            
            Unrolling_chi.load_state_dict(new_state_dict)

            Unrolling_chi.eval()

            #mask_name = "trained_models/Unrolling/testing_R4_rand_patt_all/ksp_R4_NA256/test" + str(idx+1) + '.mat'
            #matImage = scio.loadmat(mask_name)
            #Ahb = matImage['AHb']

            zf_vol = sp.ifft(inputKspace,axes=[1,2,3])

            recon_vol_mc = np.zeros(inputKspace.shape,dtype='complex64')


            mask = torch.from_numpy(abs(mask))
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            mask = mask.byte()

            for j in range(inputKspace.shape[0]):
                for k in range(inputKspace.shape[2]):

                    Ahb = np.array(np.squeeze(zf_vol[j,:,k,:]))
                    
                    ksp = np.fft.ifftshift(Ahb, axes=(0 ,1))
                    ksp = np.fft.fftn(ksp, axes=(0 ,1))
                    ksp = np.fft.fftshift(ksp, axes=(0 ,1))
                    
                    Ahb_in = Ahb
                    #AHb = np.abs(AHb)

                    Ahb_r = np.real(Ahb)
                    Ahb_r = torch.from_numpy(Ahb_r).float() 
                    Ahb_r = torch.unsqueeze(Ahb_r, 0)
                    #Ahb_r = torch.squeeze(Ahb_r, 0)

                    Ahb_i = np.imag(Ahb)
                    Ahb_i = torch.from_numpy(Ahb_i).float()
                    Ahb_i = torch.unsqueeze(Ahb_i, 0)
                    #Ahb_i = torch.squeeze(Ahb_i, 0)

                    Ahb = torch.cat([Ahb_r, Ahb_i], dim = 0).unsqueeze(0)
                    
                    
                    #print(Ahb.type())
                    #print(mask.shape)
                    #print(mask.type())

                    #ksp = np.squeeze(inputKspace[j,:,k,:])   # DEJW 11DEC2023
                    ksp_r = np.real(ksp)
                    ksp_r = torch.from_numpy(ksp_r).float() 
                    ksp_r = torch.unsqueeze(ksp_r, 0)
                    ksp_i = np.imag(ksp)
                    ksp_i = torch.from_numpy(ksp_i).float()
                    ksp_i = torch.unsqueeze(ksp_i, 0)
                    ksp = torch.cat([ksp_r, ksp_i], dim = 0).unsqueeze(0)
                    
                    
#                    R_cal_OP = R_cal((mx_size[1], mx_size[3]), Ahb, mask, 'cpu')  # old no DC version
                    R_cal_OP = R_cal((mx_size[1], mx_size[3]), Ahb, mask, ksp, 'cpu')  #DC version
                    #R_cal_OP = nn.DataParallel(R_cal_OP)


                    pred_chi, _, _ = Unrolling_chi(Ahb, R_cal_OP, 'cpu')
                    pred_chi = R2C(pred_chi) 
                    pred_chi = torch.squeeze(pred_chi, 0)
                    pred_chi = torch.squeeze(pred_chi, 0)
                    pred_chi = pred_chi.to('cpu')
                    pred_chi = pred_chi.numpy()

                    recon_vol_mc[j,:,k,:] = pred_chi

            recon_vol_abs = np.sum(np.abs(recon_vol_mc)**2, axis=0)**0.5

            
            recon_vol_cplx = coil_combine(recon_vol_mc,mps)
            

            return recon_vol_abs, recon_vol_cplx