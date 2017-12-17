import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

        
        
'''
Displays some specified slice of the 3D-image.
'''
def visualize_MRI_slice(img, slice_index, slice_direction=2):
    # slice_direction : {0,1,2} corresponding to the x,y,z axis (plots the image slices along this axis)
    
    s = img.shape
    if slice_direction==0:
        plot = plt.imshow(img[slice_index,:,:], cmap = 'gray', aspect=s[2]/float(s[1]))  
                                                          # aspect: to reshape the figure so that it's square
    elif slice_direction==1:
        plot = plt.imshow(img[:,slice_index,:], cmap = 'gray', aspect=s[2]/float(s[0]))
    else:
        plot = plt.imshow(img[:,:,slice_index], cmap = 'gray')
        
    return plot
    #plt.show()
    

    
'''
Displaying the original LR image, the HR reference and the resulting image from the method used, for comparison purposes.
'''    
def improvements_visualization(HR_ref, result_img, LR_img_interp, slice_dir, slice_index):
    
    scale = (np.min(HR_ref), np.max(HR_ref))
    big_fig = plt.figure(1, figsize=(15,5))
    
    fig = plt.subplot(131)
    fig.set_title('High resolution image (reference)')
    fig = visualize_MRI_slice(HR_ref, slice_index, slice_direction=slice_dir)
    fig = plt.subplot(132)
    fig.set_title('Reconstructed image')
    fig = visualize_MRI_slice(exposure.rescale_intensity(result_img, scale), 
                                  slice_index, slice_direction=slice_dir)
    fig = plt.subplot(133)
    fig.set_title('LR image \n (algo init : denoised & interpolated)')
    fig = visualize_MRI_slice(exposure.rescale_intensity(LR_img_interp, scale), 
                                  slice_index, slice_direction=slice_dir)