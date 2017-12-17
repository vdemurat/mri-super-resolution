import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from skimage import exposure





'''
Main function, implementing the iterative self similarity method, wrapping up the two subfunctions below.
'''
def example_based_algo(HR_data, LR_data, reconstructed_data, search_radius, local_LR_radius, h_values, param_k):
    '''
    HR_data :
    LR_data : 
    reconstructed_data : starting point of the algorithm, LR image denoised and upsampled to HR domain.
    search_radius : radius of the 3D cubic research area (in the HR image) around the current voxel under assessement.
    local_LR_radius : radius of the 3D cubic local area (in the LR image) for local similarities computations.
    h_values, param_k : parameters controling the strengh of the penalties applied at each iterations.
    '''
    for step_nb,h in enumerate(h_values):
        print('Step '+str(step_nb)+' begun')
        temp_data = reconstructed_data.copy()
        avoid_edges = search_radius+local_LR_radius
        for i in range(avoid_edges, HR_data.shape[0]-avoid_edges):
            print('Small step '+str(i))
            for j in range(avoid_edges, HR_data.shape[1]-avoid_edges):
                for k in range(avoid_edges, HR_data.shape[2]-avoid_edges):

                    reconstructed_data[i,j,k] = calc_local_update(i,j,k, temp_data, HR_data, search_radius, 
                                                                  local_LR_radius, h, param_k)
        reconstructed_data = mean_correction(reconstructed_data, LR_data)
        
    return reconstructed_data


'''
Updating voxel with respect to the local similarities with HR_ref or directly observed in the image under reconstruction  
'''          
def calc_local_update(i,j,k, temp_data, HR_data, search_radius, local_LR_radius, h, param_k):    
    '''
    i,j,k : voxel indices
    temp_data : image in reconstruction
    '''
    p = local_LR_radius
    update, norm_factor = 0, 0
    
    for x in range(i-search_radius, i+search_radius+1):
        for y in range(j-search_radius, j+search_radius+1):
            for z in range(k-search_radius, k+search_radius+1):
                # calc similarity in ref
                weight_1 = np.exp(-pow(HR_data[i,j,k]-HR_data[x,y,z],2)/pow(h,2))
                # calc local similarity in reconstructed_data
                weight_2 = np.exp(-np.sum((temp_data[(i-p):(i+p+1),(j-p):(j+p+1),(k-p):(k+p+1)]
                                           -temp_data[(x-p):(x+p+1),(y-p):(y+p+1),(z-p):(z+p+1)])**2) 
                                  /(param_k*pow(h,2)))
                
                norm_factor += weight_1*weight_2                
                update += temp_data[x,y,z]*weight_1*weight_2
                
    return update/norm_factor



'''
Mean Correction, in order to the respect the following reconstruction condition : (downsampled version of reconstructed image) = LR image.
'''
def mean_correction(reconstructed_data, LR_data):
    
    downsampled_version = block_reduce(reconstructed_data, block_size=(1,1,2), func=np.max)
    diff_interp = resize(downsampled_version - LR_data, output_shape=reconstructed_data.shape, 
                         mode='symmetric', order=3)
    
    return reconstructed_data - diff_interp