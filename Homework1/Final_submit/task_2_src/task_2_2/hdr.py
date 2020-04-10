# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:58:15 2020

@authors: M. Riccardo - E-A. Bunchalit
"""

import numpy as np


def linear_weight(pixel_value):
    
    # calculate weights
    z_min, z_max = 0., 255.
    if pixel_value > (z_min+z_max)/2:
        pixel_value = z_max - pixel_value
    else:
        pixel_value = pixel_value-z_min
    
    return pixel_value

# sample intensities from images
def sample_rgb_images(images):
    
    num_images = len(images)
    num_intensities = 256
    intensities = np.zeros((num_intensities, num_images), dtype=np.uint8)

    mid_img = images[num_images // 2] # as reference for picking up values

    for i in range(num_intensities):
        rows, cols = np.where(mid_img==i)
        if len(rows) > 0:
            sample = np.random.randint(0,len(rows))
            for j in range(num_images):
                sample_row, sample_col = rows[sample], cols[sample]
                intensities[i, j] = images[j][sample_row, sample_col]
    
    return intensities

# Debevec paper, chapter 2.1 and Abstract A
def gsolve(intensities, log_exposures, llambda, weighting_function): 
    
    intensity_range = 255
    num_samples = intensities.shape[0]
    num_images = len(log_exposures)

    A = np.zeros((num_images * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    b = np.zeros((A.shape[0], 1), dtype=np.float64)
    
    # data fitting equation
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensities[i, j] # only integers can be used as index, i.e.  A[k,z_ij]
            w_ij = weighting_function(z_ij)
            A[k, z_ij] = w_ij 
            A[k, num_samples+i] = -w_ij
            b[k,0] = w_ij*log_exposures[j] # in the paper is [i, j] but is defined as [j] only
            k += 1
    
    #smoothness equation
    for z_k in range(1,intensity_range):
        w_k = weighting_function(z_k)
        A[k, z_k-1] = w_k * llambda
        A[k, z_k] = -2 * w_k * llambda
        A[k, z_k+1] = w_k * llambda
        k += 1
        
    #fix curve - set middle value to 0
    A[-1, intensity_range//2 ] = 1
    
    #svd
    A_inv = np.linalg.pinv(A)
    x = np.dot(A_inv, b)

    g = x[0: intensity_range + 1]
    #lE = x[n+1: x.shape[0]]
    
    return g[:, 0]

# Debevec paper, chaper 2.2
def compute_hdr_map(images, log_exposures, response_curve, weighting_function):
    
    shapes = images[0].shape
    img_rad_map = np.zeros(shapes, dtype=np.float64)

    g = np.zeros(len(images))
    w = np.zeros(len(images))
    
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            for k in range(len(images)):
                g[k] = response_curve[images[k][i,j]]
                w[k] = weighting_function(images[k][i,j])
            Sum_w = np.sum(w)
            if Sum_w>0:
                img_rad_map[i,j] = np.sum(w*(g-log_exposures)/Sum_w)
            else:
                img_rad_map[i,j] = g[len(images)//2] - log_exposures[len(images)//2]
    
    return img_rad_map

if __name__ == '__main__':
    pass