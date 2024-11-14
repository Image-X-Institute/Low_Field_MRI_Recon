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