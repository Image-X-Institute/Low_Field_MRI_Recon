# Low_Field_MRI_Recon

## Accelerating Low-field MRI: Compressed Sensing and AI for Fast Noise-robust Imaging

Public repository with data and code supporting Shimron et al. 2024. A preprint of this manuscript is available at https://arxiv.org/abs/2411.06704

Authors: David Waddington, Efrat Shimron.

The following Jupyter notebooks were used to generate figures contained in the manuscript:
FigA_fast_mri_sampling_experiment.ipynb
FigB_fast_mri_noise_experiment_complete-R2_R4.ipynb
FigC_retro_ulf_3T_compare.ipynb
FigD_phantom_experiment_plot.ipynb
FigE_prospective_recon-phantom-brain.ipynb

The following python files are needed to run the Jupyter notebooks
automap_fns.py - 
display_fns.py - 
metrics.py - 
ulf_recon_fns.py - 
unrolling_fns.py - 

A requirements.txt file has been generated that details the pip packages required to run the Jupyter notebooks.

Further Unrolled reconstruction code is available here: https://github.com/shanshanshan3/DCReconNet
Further AUTOMAP code is available here: https://github.com/MattRosenLab/AUTOMAP

Raw k-space data used in the manuscript is available from the corresponding author upon reasonable request.
