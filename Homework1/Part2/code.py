# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:43:22 2020

@author: darim
"""

import numpy as np
import cv2
import random

def film_response_recovery(Z, B, l, w):
    # i is number of samples
    i = 0
    return i
    
# 2.1 and Abstract A
#
# Z(i,j) pixel value of pixel i in image j - i is number of samples - j is number of images
# B(j) is log delta t for image j
# l is lambda
# w(z) is the weighting function - defined outside
    
def gsolve(images, exposures): 
    Z = sample_images(images)
    B = exposures
    l = 100.
    
    Z_min = 0
    Z_max = 255
    
    n = 255
    
    A = np.zeros((Z.shape[0] * Z.shape[1] + n , n + Z.shape[0]), dtype=np.float64)
    b = np.zeros((A.shape[0], 1))
    
    # data fitting equation
    
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            zijp1 = Z[i, j] + 1
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
    
def sample_images(images):
    Z_min = 0
    Z_max = 255
    n = 256
    
    num_images = len(images)
    intensity = np.zeros((n, num_images), dtype=np.uint8)

    mid_img = images[num_images // 2] # optional - since Z_min and Z_max are fixed, just place the mid value

    for i in range(Z_min, Z_max + 1): # optional - since Z_min and Z_max are fixed, just place the two values
        rows, cols= np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(num_images):
                intensity[i, j] = images[j][rows[idx], cols[idx]]
    return intensity
    
def weighting_function(z):
    Z_min = 0.
    Z_max = 255.
    
    if z <= (Z_min + Z_max) / 2.:
        return z - Z_min
    else:
        return Z_max - z
    
def merger(images, exposures, response): #2.2
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)
    
    num_images = len(images)
    
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            g = np.array([response[images[k][i, j]] for k in range(num_images)])
            w = np.array([weighting_function(images[k][i, j]) for k in range(num_images)])
            SumW = np.sum(w)
            if SumW > 0:
                img_rad_map[i, j] = np.sum(w * (g - exposures) / SumW)
            else:
                img_rad_map[i, j] = g[num_images // 2] - exposures[num_images // 2]
    return img_rad_map


def globalToneMapping(image, gamma):
    
    image_corrected = cv2.pow(image/255., 1.0/gamma)
    return image_corrected


def intensityAdjustment(image, template):
    
    m, n, channel = image.shape
    output = np.zeros((m, n, channel))
    for ch in range(channel):
        image_avg, template_avg = np.average(image[:, :, ch]), np.average(template[:, :, ch])
        output[..., ch] = image[..., ch] * (template_avg / image_avg)

    return output

# Read all the files with OpenCV
files = ['images/memorial00.png', 'images/memorial01.png', 'images/memorial02.png', 'images/memorial03.png',\
         'images/memorial04.png', 'images/memorial05.png', 'images/memorial06.png', 'images/memorial07.png',\
         'images/memorial08.png', 'images/memorial09.png', 'images/memorial10.png', 'images/memorial11.png',\
         'images/memorial12.png', 'images/memorial13.png', 'images/memorial14.png', 'images/memorial15.png']
images = list([cv2.imread(f) for f in files])
# Compute the exposure times in seconds
exposures = np.float32([1. / t for t in [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]])

# Compute the response curve

num_channels = images[0].shape[2]
hdr_image = np.zeros(images[0].shape, dtype=np.float64)

for channel in range(num_channels):  
    print("start {}     ".format(channel))
    layer_stack = [img[:, :, channel] for img in images]
    response = gsolve(layer_stack, exposures)
    print("mid1 {}     ".format(channel))
    # Compute the HDR image

    hdr = merger(layer_stack, exposures, response)
    print("mid2 {}     ".format(channel))
    hdr_image[..., channel] = cv2.normalize(hdr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print("end {}     ".format(channel))
    
    

"""

# Global tone mapping
image_mapped = globalToneMapping(hdr_image, gamma=0.6)

    # Adjust image intensity based on the middle image from image stack
template = images[len(images)//2]
image_tuned = intensityAdjustment(image_mapped, template)

    # Output image
output = cv2.normalize(image_tuned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imwrite('durand_image.png', output)

    
"""   
template = images[len(images)//2]
image_tuned = intensityAdjustment(hdr_image, template)

durand = cv2.createTonemapDurand(gamma=0.6)
ldr = durand.process(image_tuned)



output = cv2.normalize(image_tuned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imwrite('durand_image.png', output)


#"""