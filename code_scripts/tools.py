import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from skimage import exposure


'''
Downsamples a 3D-image along the z-axis.
'''
def downsample_z_axis(img, downsampling_factor=2):
    return block_reduce(img, block_size=(1,1,downsampling_factor), func=np.max)


'''
Denoises an MRI image (rician noise).
'''
def denoise(img):
    sigma_esti = estimate_sigma(img, N=1)
    return nlmeans(img, sigma=sigma_esti, patch_radius= 1, block_radius = 1, rician= True)


'''
Interpolation Method chosen for the project.
'''
def spline_interpolation(img, HR_shape, interp_order):
    return resize(img, output_shape=HR_shape, mode='symmetric', order=interp_order)


'''
Signal-to_Noise Ratio's definition used for the project evaluation metric : 
std(zone where information is present)/std(noisy zone).
'''
def snr(img):
    HR_ref = (img - np.min(img)) * 255.0 / (np.max(img) - np.min(img))
    
    return np.std(img[260:300,280,17:22])/np.std(img[210:250,280,9:14])