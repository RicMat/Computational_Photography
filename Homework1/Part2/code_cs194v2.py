# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:20:06 2020

@author: darim
"""


import numpy as np
import cv2
from skimage.util import img_as_ubyte

def create_hdr_image(l):
    
    images, exposures = read_images()
    ln_dt = np.log(exposures)
    
    z_red, z_green, z_blue = sample_rgb_images(images)
    
    weights = compute_weights()
    #print(z_red.shape)
    g_red = gsolve(z_red, ln_dt, l, weights)
    g_green = gsolve(z_green, ln_dt, l, weights)
    g_blue = gsolve(z_blue, ln_dt, l, weights)
    
    hdr_map = compute_hdr_map(images, g_red, g_green, g_blue, weights, ln_dt);
    
    return hdr_map
    
    
def read_images():
    
    files = ['images/memorial00.png', 'images/memorial01.png', 'images/memorial02.png', \
          'images/memorial07.png',\
         'images/memorial08.png', 'images/memorial09.png', 'images/memorial10.png', \
         'images/memorial15.png']
    
    images = list([cv2.imread(f) for f in files])

    exposures = np.float32([1. / t for t in [0.03125, 0.0625, 0.125, 4, 8, 16, 32, 1024]])
    
    return images, exposures


def sample_rgb_images(images):
    
    num_exposures = len(images)
    num_samples = np.rint(255 / (num_exposures - 1) * 2).astype(np.int)
    img_num_pixels = images[0].shape[0]* images[0].shape[1]
    
    #mid_img = images[num_exposures // 2] # used as reference
    
    step = img_num_pixels / num_samples
    sample_indices = get_sample_index(step, img_num_pixels)
    
    z_red = np.zeros((num_samples, num_exposures))
    z_green = np.zeros((num_samples, num_exposures))
    z_blue = np.zeros((num_samples, num_exposures))
    
    for i in range(num_exposures):
        sampled_red, sampled_green, sampled_blue = sample_exposure(images[i], sample_indices)
        z_red[..., i] = sampled_red
        z_green[..., i] = sampled_green
        z_blue[..., i] = sampled_blue
   
    return z_red, z_green, z_blue


def get_sample_index(step, img_pixels): 
    
    """ Z_min = 0
    Z_max = 255
    x1 = []
    x2 = [] """
    
    indexes = np.arange(1, img_pixels, step)
    indexes = np.rint(indexes).astype(np.int)
    
    """
    for i in range(Z_min, Z_max + 1): 
        rows, cols, ch= np.where(reference == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            # https://stackoverflow.com/questions/54930176/access-elements-of-a-matrix-by-a-list-of-indices-in-python-to-apply-a-maxval-0
            x1.append(rows[idx])      
            x2.append(cols[idx])
            i += step
    
    indexes = [x1,x2]
    """
    return indexes

def sample_exposure(image, indexes):
    
    #indexes = indexes.transpose() # maybe indexes.conj()
    
    red_img = image[..., 0]
    green_img = image[..., 1]
    blue_img = image[..., 2]
    
    # https://stackoverflow.com/questions/54930176/access-elements-of-a-matrix-by-a-list-of-indices-in-python-to-apply-a-maxval-0
    red_img = red_img.flatten()
    green_img = green_img.flatten()
    blue_img = blue_img.flatten()
    
    sampled_red = red_img[indexes]
    sampled_green = green_img[indexes]
    sampled_blue = blue_img[indexes]
    
    #print(sampled_red.shape)
    
    """
    sampled_red = red_img[indexes[:,0], indexes[:,1]]
    sampled_green = green_img[indexes[:,0], indexes[:,1]]
    sampled_blue = blue_img[indexes[:,0], indexes[:,1]]
    
    sampled_red = red_img[indexes]
    sampled_green = green_img[indexes]
    sampled_blue = blue_img[indexes]
    """
    return sampled_red, sampled_green, sampled_blue
    

def compute_weights():
    
    weights = np.arange(1, 256, dtype=np.int)
    weights = np.minimum(weights, 256 - weights)
    
    return weights
    
    
def weighting_function(z):
    
    Z_min = 0.
    Z_max = 255.
    
    if z <= (Z_min + Z_max) / 2.:
        return z - Z_min
    else:
        return Z_max - z
    

def weights_total(image, weights):
    
    return np.where(image > 128, 128 - (image - 128), image)
    
    

def gsolve(images, exposures, l, weights): #review
    Z = images
    B = exposures
    
    Z_min = 0
    Z_max = 255
    
    n = 255
    
    A = np.zeros((Z.shape[0] * Z.shape[1] + n , n + Z.shape[0]), dtype=np.float64)
    b = np.zeros((A.shape[0], 1))
    
    # data fitting equation
    
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            zijp1 = (Z[i, j] + 1).astype(int)
            wij = weighting_function(zijp1)
            A[k, zijp1] = wij # only integers as index
            A[k, n+i] = -wij # n+i+1
            b[k,0] = wij * B[j] # in the pdf is [i, j] but is defined as [j] only
            k += 1
    

    #fix curve - set middle value to 0
    
    A[k, (Z_max - Z_min) // 2] = 1 # optional - since Z_min and Z_max are fixed, just place the mid value
    
    
    #smoothness equation
    
    for i in range(Z_min, Z_max):  # optional - since Z_min and Z_max are fixed, just place the two values
        w_ip1 = weighting_function(i+1)
        A[k,i] = l * w_ip1
        A[k,i+1] = -2 * l * w_ip1
        A[k,i+2] = l * w_ip1
        
    #svd
        
    A_inv = np.linalg.pinv(A)
    x = np.dot(A_inv, b)
    g = x[0: n+1]
    #lE = x[n+1: x.shape[0]]
            
    return g[:, 0]


def compute_hdr_map(images, g_red, g_green, g_blue, weights, ln_dt):
    
    num_exposures = len(images)
    height, width, num_channels = images[0].shape
    numerator = np.zeros((height, width, num_channels))
    denominator = np.zeros((height, width, num_channels))
    curr_num = np.zeros((height, width, num_channels))    
    
    for i in range(num_exposures):
        # Grab the current image we are processing and split into channels.
        #onn = np.ones((images[i].shape[0], images[i].shape[1]))
        
        curr_image = np.where(images[i] == 0, 1, images[i]) # Grab the current image.  Add 1 to get rid of zeros.
        
        curr_red = curr_image[..., 0]
        curr_green = curr_image[..., 1]
        curr_blue = curr_image[..., 2]

        # Compute the numerator and denominator for this exposure.  Add to cumulative total.
        #          sum_{j=1}^{P} (w(Z_ij)[g(Z_ij) - ln dt_j])
        # ln E_i = ------------------------------------------
        #                  sum_{j=1}^{P} (w(Z_ij))
        
        
        curr_weight = weights_total(curr_image, weights)
        
        #if i == 5:
            #print(ln_dt[i])
        
        curr_num[..., 0] = curr_weight[..., 0] * (g_red[curr_red] - ln_dt[i])
        curr_num[..., 1] = curr_weight[..., 1] * (g_green[curr_green] - ln_dt[i])
        curr_num[..., 2] = curr_weight[..., 2] * (g_blue[curr_blue] - ln_dt[i])
        
        # Sum into the numerator and denominator.
        numerator = numerator + curr_num        
        denominator = denominator + curr_weight

    ln_hdr_map =  numerator / denominator
    print(curr_num.shape)
    print(curr_weight)
    
    hdr_map = np.exp(ln_hdr_map)
    #print(hdr_map)
    
    return hdr_map
    

def apply_linear_tonemap(hdr):
    
    luminance_map = compute_luminance_map(hdr)
    display_luminance = luminance_map / np.max(luminance_map)

    
    result = apply_tonemap(hdr, luminance_map, display_luminance)
    
    return result


def compute_luminance_map(hdr):
    
    red_channel = hdr[:,:,0]
    green_channel = hdr[:,:,1]
    blue_channel = hdr[:,:,2]
    luminance_map = 0.2126 * red_channel + 0.7152 * green_channel + 0.0722 * blue_channel
    
    return luminance_map


def apply_tonemap(hdr, luminance, display):
    
    height, width, num_channels = hdr.shape
    result = np.zeros((height, width, num_channels))
    for ch in range(num_channels):
        result[:,:,ch] = ((hdr[:,:,ch] / luminance)) * display
        
    return result


if __name__ == "__main__":
    
    hdr = create_hdr_image(50)
    
    #out = apply_linear_tonemap(hdr * 255)
    
    #durand = cv2.createTonemapDurand(gamma=2.5)
    #ldr = durand.process(hdr)
    
    #print(hdr*255)
    #new = img_as_ubyte(ldr)
    
    #cv2.imwrite(im2uint8(linear_result), ['output/', directory, '_linear_hdr.jpg'], 'jpg')
    #asdf = np.zeros(shape=hdr.shape, dtype=hdr.dtype)
    #cv2.normalize(hdr, asdf, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #print(asdf)
    cv2.imwrite('durand_image_cs194.jpg', hdr)
    