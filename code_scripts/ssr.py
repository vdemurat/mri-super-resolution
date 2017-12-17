import os
import numpy as np
import matplotlib.pyplot as plt
import time
from math import pi
from skimage.transform import resize
from PIL import Image
from skimage.measure import block_reduce
from skimage import restoration, exposure
from scipy.ndimage import rotate, convolve, sobel, laplace
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from scipy import fftpack


os.chdir('code_scripts')
import visualization as viz
os.chdir('../')

'''
    Takes original, upsampled MRI image and generates a rotated y0, y_a0, and y_s0
    
    Params: original_image = original, upsampled (via cubic spline) MRI image
            kernel = rect function kernel 
            rotation_degree = degree of rotation that you want
          
    Return: rotated_image = R(y0), where R = rotation matrix
            image_a0 = upsampled y0 that is convolved with the rotated kernel
            image_s0 = R(y0) that is convolved with the non-rotated kernel
'''
def generate_rotated_images(original_img, kernel, rotation_degree): 
    
    # Rotate original image by the given degree 
    rotated_image = rotate(original_img, rotation_degree, reshape = False)
    
    # Rotate rect function kernel
    rotated_kernel = rotate(kernel, rotation_degree, reshape = False)
    rotated_kernel = np.where(rotated_kernel > 0.1, 1, 0)
    
    # Create empty arrays for y_a0 and y_s0 in the original paper
    image_a0 = np.zeros(shape = original_img.shape)
    image_s0 = np.zeros(shape = original_img.shape)
    
    # Fill in the arrays with the convolved images
    # image_a0 = rotated kernel convolved with original image
    # image_s0 = original kernel convolved with rotated image
    for i in range(original_img.shape[1]):
        image_a0[:,i,:] = convolve(original_img[:,i,:], rotated_kernel)
        image_s0[:,i,:] = convolve(rotated_image[:,i,:], kernel)
    
    return rotated_image, image_a0, image_s0


'''
    Computes first and second gradient images of y_a0 in all 3 directions 
    using Sobel filter for 1st deriv (scipy.ndimage.sobel) and Laplacian filter for 2nd deriv 
    
    Params: image_a0 = the upsampled image  
    
    Return: Sobel_x, Sobel_y, Sobel_z = 3D arrays of the 1st gradient of the image using the Sobel filter
            Laplacian_x, Laplacian_y, Laplacian_z = 3D arrays of the 2nd gradient of the image using the Laplacian filter
'''
def compute_gradient_images(image_a0):
    
    Laplacian_x = np.zeros(shape = image_a0.shape)
    Laplacian_y = np.zeros(shape = image_a0.shape)
    Laplacian_z = np.zeros(shape = image_a0.shape)
    
    # Compute the first gradient images in each respective direction using the Sobel filter 
    Sobel_x = sobel(image_a0, 0)
    Sobel_y = sobel(image_a0, 1)
    Sobel_z = sobel(image_a0, 2)
    
    # Compute the second gradient images in each respective direction using the Laplacian filter
    for i in range(image_a0.shape[0]):
        Laplacian_x[i,:,:] = laplace(image_a0[i,:,:])
    
    for k in range(image_a0.shape[1]):
        Laplacian_y[:,k,:] = laplace(image_a0[:,k,:])
    
    for j in range(image_a0.shape[2]):
        Laplacian_z[:,:,j] = laplace(image_a0[:,:,j])
    
    return Sobel_x, Sobel_y, Sobel_z, Laplacian_x, Laplacian_y, Laplacian_z


'''
    Concatenates patches of size 4x4xZ (size of Z-axis) from the 6 gradient images to form feature vectors
    
    Params: Sobel_x, Sobel_y, Sobel_z, Laplacian_x, Laplacian_y, Laplacian_z = arrays generated from compute_gradient_images
    
    Return: feature_array = 2D Array of feature vectors, where each row corresponds to a feature per image patch
'''
def extract_features(Sobel_x, Sobel_y, Sobel_z, Laplacian_x, Laplacian_y, Laplacian_z):
    
    x = int(Sobel_x.shape[0]/4)
    y = int(Sobel_x.shape[1]/4)
    feature_array = np.zeros(shape = (x*y, 6*4*4*Sobel_x.shape[2]))
    k = 0
    
    for i in range(0, Sobel_x.shape[1], 4):
        for j in range(0, Sobel_x.shape[0], 4):
            Sobel_x_feature = Sobel_x[j:(j+4), i:(i+4), :].flatten()
            Sobel_y_feature = Sobel_y[j:(j+4), i:(i+4), :].flatten()
            Sobel_z_feature = Sobel_z[j:(j+4), i:(i+4), :].flatten()
            Laplacian_x_feature = Laplacian_x[j:(j+4), i:(i+4), :].flatten()
            Laplacian_y_feature = Laplacian_y[j:(j+4), i:(i+4), :].flatten()
            Laplacian_z_feature = Laplacian_z[j:(j+4), i:(i+4), :].flatten()
            
            feature_per_patch = np.concatenate((Sobel_x_feature, Sobel_y_feature, Sobel_z_feature,
                                                Laplacian_x_feature, Laplacian_y_feature, Laplacian_z_feature))
            feature_array[k, :] = feature_per_patch
        
            k += 1
    
    return feature_array


'''
    Compresses input features by a magnitude of ~100 using PCA - use number_of_PCs = int(feature_array.shape[1]/100)
    
    Params: feature_array_train = 2D Array of concatenated patch features from the Sabel and Laplacian gradients 
                                  in each direction for the training image
            feature_array_test =  2D Array of concatenated patch features from the Sabel and Laplacian gradients 
                                  in each direction for the test image
            number_of_PCs = number of principal components you want to keep (a hyperparameter we can tweak)
            
    Return: reduced_features_train = training features that are reduced to a dimension of number_of_PCs
            reduced_features_test = test features that are reduced to a dimension of number_of_PCs
'''
def compress_features(feature_array_train, feature_array_test, number_of_PCs):
    
    pca_transform_train = PCA(n_components=number_of_PCs).fit(feature_array_train)
    reduced_features_train = pca_transform_train.transform(feature_array_train)
    
    reduced_features_test = pca_transform_train.transform(feature_array_test)
    
    return reduced_features_train, reduced_features_test


'''
    Computes the difference image y_a0d = y_0 - y_a0
    
    Params: rotated_image = upsampled LR image that is rotated by rotation matrix
            image_a0 = upsampled LR image that is convolved with the rotated kernel
            
    Return: difference_image_patches = patches of size 4x4xrotated_image.shape[2] from the difference image y_a0d
'''
def get_difference_image_patches(rotated_image, image_a0):
    
    difference_image = rotated_image - image_a0
    x = int(difference_image.shape[0]/4)
    y = int(difference_image.shape[1]/4)
    difference_image_patches = np.zeros(shape = (x*y, 4*4*difference_image.shape[2]))
    k = 0
    
    for i in range(0, difference_image.shape[1], 4):
        for j in range(0, difference_image.shape[0], 4):
            difference_image_patches[k, :] = difference_image[j:(j+4), i:(i+4), :].flatten()

            k += 1
    
    return difference_image_patches


'''
    Get feature clusters using K-Means: original paper uses K-SVD but this implementation is quite complicated
    
    Params: reduced_features_train = matrix of PCA-reduced feature vectors for the training image
            reduced_features_test = matrix of PCA-reduced feature vectors for the test image
            number_of_clusters = number of clusters we want to use - we use 128 since that is what the paper uses
            
    Return: labels_train = array for which cluster each training feature is matched to 
            labels_test = array for which cluster each test feature is matched to
'''
def cluster_with_kmeans(reduced_features_train, reduced_features_test, number_of_clusters):
        
    kmeans = KMeans(init='k-means++', n_clusters=number_of_clusters, random_state=124)
    kmeans.fit(reduced_features_train)
    
    labels_train = kmeans.predict(reduced_features_train)
    labels_test = kmeans.predict(reduced_features_test)

    return labels_train, labels_test


'''
    Compute projection matrices P_k for each cluster k that minimizes 
    the least squares distance P_k*feature_vec = difference_image_patches
    
    Params: reduced_features = matrix of PCA-reduced feature vectors
            differenced_image_patches = patches of the differenced image
            labels = K-means or K-SVD generated groupings
            regularization = value for lambda parameter
    
    Return: projection_matrices = numpy array of size 128 x 
'''
def compute_projection_matrices_per_cluster(reduced_features, differenced_image_patches, labels, regularization):
    indices_per_group = {i: np.where(labels == i)[0] for i in range(128)}
    projection_matrices = np.zeros(shape = (128, reduced_features.shape[1], differenced_image_patches.shape[1]))
    
    for i in range(len(indices_per_group)):
        features_of_group = reduced_features[indices_per_group[i], :]
        differenced_img_patches_of_group = differenced_image_patches[indices_per_group[i], :]
        
        regularization_matrix = regularization * np.identity(features_of_group.shape[1])
        projection_1 = np.linalg.inv(np.matmul(np.transpose(features_of_group), features_of_group) + regularization_matrix)
        projection_2 = np.matmul(np.transpose(features_of_group), differenced_img_patches_of_group)
        projection_matrices[i, :, :] = np.matmul(projection_1, projection_2)
        
    return projection_matrices


'''
    Project test set using the learned projection matrices
    
    Params: compressed_features_test = the features of the test image y_s0
            projection_matrices = learned projection matrices from the training images
            labels_test = labels of the test features
            original_image_shape = shape of rotated(y0)
    
    Return: new_image = the test image that is projected on the learned matrices
'''
def projection_test_set_features(compressed_features_test, projection_matrices, 
                                 labels_test, original_image_shape):
    
    new_image = np.zeros(shape = original_image_shape)
    j = 0
    k = 0
    for i in range(compressed_features_test.shape[0]):
        
        if j > 0 and j % original_image_shape[0] == 0:
            j = 0
            k += 4
        
        group = labels_test[i]
        projection = np.dot(compressed_features_test[i,:], projection_matrices[group])
        projection = np.reshape(projection, (4,4,38))
        new_image[j:(j+4),k:(k+4), :] = projection
        j += 4

    new_image = (new_image - np.min(new_image)) * 255.0 / (np.max(new_image) - np.min(new_image))
    return new_image


'''
    This function wraps all the above functions together to generate the rotation images
    
    Params: original_img = original image that is interpolated to target resolution
            kernel = slice selection kernel h0
            rotation_degree = specified rotation degree 
            regularization = parameter for ridge regression  
            number_of_PCs = number of principal components to project on
            
    Return: projection + y_s0 = newly generated image
            projection = the projection of the image y_s0 on the learned projection matrices
'''
def generate_images(original_img, kernel, rotation_degree, regularization, number_of_PCs):
    rotated_y0, y_a0, y_s0 = generate_rotated_images(original_img, kernel, rotation_degree)
    print('Rotated Images Generated')
    
    Sobel_x_train,Sobel_y_train,Sobel_z_train,Laplace_x_train,Laplace_y_train,Laplace_z_train = compute_gradient_images(y_a0)
    Sobel_x_test, Sobel_y_test, Sobel_z_test, Laplace_x_test, Laplace_y_test, Laplace_z_test = compute_gradient_images(y_s0)
    print('Gradient Images Generated')
    
    features_train = extract_features(Sobel_x_train, Sobel_y_train, Sobel_z_train, 
                                      Laplace_x_train, Laplace_y_train, Laplace_z_train)
    features_test = extract_features(Sobel_x_test, Sobel_y_test, Sobel_z_test, 
                                     Laplace_x_test, Laplace_y_test, Laplace_z_test)
    compressed_features_train, compressed_features_test = compress_features(features_train, 
                                                                            features_test, 
                                                                            number_of_PCs)
    print('Features Extracted and Compressed')
    
    difference_image_patches = get_difference_image_patches(original_img, y_a0)
    print('Image Patches of Differenced Image are Acquired')
    
    labels_train, labels_test = cluster_with_kmeans(compressed_features_train, compressed_features_test, 128)
    print('Clusters Completed')
    
    learned_projection_matrices = compute_projection_matrices_per_cluster(compressed_features_train, 
                                                                          difference_image_patches, 
                                                                          labels_train, 
                                                                          regularization)
    print('Projection Matrices are Learned')
    
    projection = projection_test_set_features(compressed_features_test, 
                                              learned_projection_matrices, 
                                              labels_test, 
                                              y_a0.shape)
    print('Projected Image is Created')
    return projection + y_s0, projection





'''
    Computes the fourier burst accumulation (e.g. the inverse FFT of the weighted average of the FFT of each image)
    
    Params: p = non-negative parameter that scales the frequency weights
            image_array = 4-D array of the generated images
    
    Return: new_image = FBA generated image
'''
def fourier_burst_accumulation(p, image_array):
    fba_fft = np.zeros(shape = (image_array.shape[1], image_array.shape[2], image_array.shape[3]))
    fft_matrix = np.zeros(shape = image_array.shape)
    sum_array = np.zeros(shape = (image_array.shape[1], image_array.shape[2], image_array.shape[3]))
    max_val = 0
    
    for i in range(image_array.shape[0]):
        fft_matrix[i,:,:,:] = np.fft.fftn(image_array[i,:,:,:])
        sum_array = np.add(sum_array, np.power(np.absolute(fft_matrix[i,:,:,:]), p))
        
    for j in range(image_array.shape[0]): 
        weight = np.divide(np.power(np.absolute(fft_matrix[j,:,:,:]), p), sum_array)
        fba_fft += np.multiply(weight, fft_matrix[j,:,:,:]) 
        
    new_image = np.real(np.fft.ifftn(fba_fft))
    return new_image
        

    
'''
    Iterate over a range of p values in order to create new FBA generated super-resolution images then plots them
    
    Params: max_p = the maximum value of p that you want to test up to (we chose 5)
            image_array = array of images that FBA will be run on
            true_image = true HR image
            LR_interp_image = low resolution image that is upsampled via interpolation
'''
def test_for_param_p(max_p, image_array, true_image, LR_interp_image):
    
    big_fig = plt.figure(1, figsize=(15,5))
    fig = plt.subplot(121)
    fig.set_title('True, HR Image in X-Y Plane')
    fig = plt.imshow(true_image[:,:,15], cmap='gray')
    fig = plt.subplot(122)
    fig.set_title('True, HR Image in Z Plane')
    fig = viz.visualize_MRI_slice(exposure.rescale_intensity(true_image), 280, slice_direction=1)
    
    big_fig = plt.figure(1, figsize=(15,5))
    fig = plt.subplot(121)
    fig.set_title('Interpolated LR Image in X-Y Plane')
    fig = plt.imshow(LR_interp_image[:,:,15], cmap='gray')
    fig = plt.subplot(122)
    fig.set_title('Interpolated LR Image in Z Plane')
    fig = viz.visualize_MRI_slice(exposure.rescale_intensity(LR_interp_image), 280, slice_direction=1)
    
    for i in range(max_p):
        new_image = fourier_burst_accumulation(i, image_array)
        
        big_fig = plt.figure(1, figsize=(15,5))
        fig = plt.subplot(121)
        fig.set_title('Image in X-Y Plane for P =' + str(i))
        fig = plt.imshow(new_image[:,:,15], cmap='gray')
        fig = plt.subplot(122)
        fig.set_title('Image in Z Plane for P =' + str(i))
        fig = viz.visualize_MRI_slice(exposure.rescale_intensity(new_image), 
                                      280, slice_direction=1)