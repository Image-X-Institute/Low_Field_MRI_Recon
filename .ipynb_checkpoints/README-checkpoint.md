# Low_Field_MRI_Recon

## Accelerating Low-field MRI: Compressed Sensing and AI for Fast Noise-robust Imaging

Authors: David Waddington, Efrat Shimron, Shanshan Shan, Neha Koonjoo.

Public repository with data and code supporting Shimron et al. 2024. This work investigates and compares leading compressed sensing and AI-based methods for image reconstruction at ultra-low magnetic fields. A preprint of the manuscript is available at https://arxiv.org/abs/2411.06704 . 

![Reconstruction accuracy across field strengths](overview.png)

## Key Files

The following Jupyter notebooks were used to generate figures contained in the manuscript:
- FigA_fast_mri_sampling_experiment.ipynb: Figure 2
- FigB_fast_mri_noise_experiment_complete-R2_R4.ipynb: Figure 3,4
- FigC_phantom_experiment_plot.ipynb: Figure 5
- FigD_retro_ulf_3T_compare.ipynb: Figure 6
- FigE_prospective_recon-phantom-brain.ipynb: Figure 7

The following python files are needed to run the Jupyter notebooks:
- automap_fns.py: Code for applying AUTOMAP to k-space data.
- display_fns.py: Code for displaying images as subplots that compare various reconstruction methods.
- metrics.py: Code for calculating image reconstruction metrics such as NRMSE and SSIM.
- ulf_recon_fns.py: Code for masking fully-sampled datasets and performing IFFT and CS reconstruction.
- unrolling_fns.py: Code for applying Unrolled AI to k-space data for image reconstruction.

A requirements.txt file has been generated that details the pip packages required to run the Jupyter notebooks.

## Data and model parameters

For access to raw data from the human imaging experiments, please contact the corresponding author to ensure compliance with IRB requirements. Raw data based on the fastMRI dataset and from phantom imaging can be downloaded from the following link: https://unisyd-my.sharepoint.com/:f:/g/personal/david_waddington_sydney_edu_au/ElLlWPVjQlFAt4wGwVE9tqoB2iEYAlZ9uMnISh3VFR5o8Q?e=dLrnf8 . 

The parameter files used for Unrolled AI and AUTOMAP reconstruction in this manuscript can be downloaded from the following link: https://unisyd-my.sharepoint.com/:f:/g/personal/david_waddington_sydney_edu_au/EhxbcMYRnlFEpJplI6ZQDu4BE-wHWpYcGS0x9kdcgBfv9Q?e=Kdjdhj

Further Unrolled reconstruction code is available here: https://github.com/shanshanshan3/DCReconNet
Further AUTOMAP code is available here: https://github.com/MattRosenLab/AUTOMAP


## Acknowledgements

Reconstruction approaches were adapted from code associated with the following publications:
1. F. Ong, M. Lustig, SigPy: a python package for high performance iterative reconstruction. 27th Annual Meeting of the International Society of Magnetic Resonance in Medicine (2019), p. 4819.
2. S. Shan, Y. Gao, P. Z. Y. Liu, B. Whelan, H. Sun, B. Dong, F. Liu, D. E. J. Waddington, Distortion-Corrected Image Reconstruction with Deep Learning on an MRI-Linac. Magnetic Resonance in Medicine 90, 963-977 (2023).
3. N. Koonjoo, B. Zhu, G. C. Bagnall, D. Bhutto, M. S. Rosen, Boosting the signal-to-noise of low-field MRI with deep learning image reconstruction. Scientific Reports 11, 8248-8248 (2021). 

AI models were trained using data sourced from the following publications and their public repositories:
1. Q. Fan, T. Witzel, A. Nummenmaa, K. R. A. Van Dijk, J. D. Van Horn, M. K. Drews, L. H. Somerville, M. A. Sheridan, R. M. Santillana, J. Snyder, T. Hedden, E. E. Shaw, M. O. Hollinshead, V. Renvall, R. Zanzonico, B. Keil, S. Cauley, J. R. Polimeni, D. Tisdall, R. L. Buckner, V. J. Wedeen, L. L. Wald, A. W. Toga, B. R. Rosen, MGH–USC Human Connectome Project datasets with ultra-high b-value diffusion MRI. NeuroImage 124, 1108-1114 (2016).
2. F. Knoll, J. Zbontar, A. Sriram, M. J. Muckley, M. Bruno, A. Defazio, M. Parente, K. J. Geras, J. Katsnelson, H. Chandarana, Z. Zhang, M. Drozdzalv, A. Romero, M. Rabbat, P. Vincent, J. Pinkerton, D. Wang, N. Yakubova, E. Owens, C. L. Zitnick, M. P. Recht, D. K. Sodickson, Y. W. Lui, fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning. Radiology: Artificial Intelligence 2, e190007-e190007 (2020).
