import numpy as np
import matplotlib.pyplot as plt


def visualize_MRI(image, slice_direction=2):
    # slice_direction : {0,1,2} corresponding to the x,y,z axis (plots the image slices along this axis)
    
    for i in range(image.shape[slice_direction]):
        plt.imshow(image[:,:,i], cmap = 'gray')
        plt.show()
        
        

def visualize_MRI_slice(image, slice_index, slice_direction=2):
    # slice_direction : {0,1,2} corresponding to the x,y,z axis (plots the image slices along this axis)
    
    s = image.shape
    if slice_direction==0:
        a = plt.imshow(image[slice_index,:,:], cmap = 'gray', aspect=s[2]/float(s[1]))  
                                                          # aspect: to reshape the figure so that it's square
    elif slice_direction==1:
        plt.imshow(image[:,slice_index,:], cmap = 'gray', aspect=s[2]/float(s[0]))
    else:
        plt.imshow(image[:,:,slice_index], cmap = 'gray')
        
    plt.show()