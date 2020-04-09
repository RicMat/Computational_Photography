# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:20:06 2020

@authors: M. Riccardo - E-A. Bunchalit
"""

import cv2
import numpy as np
import os
from os import path
import hdr as hdr


SRC_FOLDER = "images"
EXPOSURE_TIMES = np.float64([1. / t for t in [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]])

OUT_FOLDER = ""

EXTENSIONS = set(["jpg", "png"])

def compute_hdr(images, log_exposure_times, smoothing_lambda=100.):
    
    images = [np.atleast_3d(i) for i in images]
    num_channels = images[0].shape[2]

    hdr_image = np.zeros(images[0].shape, dtype=np.float64)
    hdr_image_01 = np.zeros(images[0].shape, dtype=np.float64)

    for channel in range(num_channels):

        layer_stack = [img[:, :, channel] for img in images]

        intensity_samples = hdr.sample_rgb_images(layer_stack)

        response_curve = hdr.gsolve(intensity_samples, log_exposure_times, smoothing_lambda, hdr.linearWeight)
        
        hdr_map = hdr.compute_hdr_map(layer_stack, log_exposure_times, response_curve, hdr.linearWeight)

        max_val_Ei = np.max(hdr_map)
        max_for_01_perc = max_val_Ei * 0.001
        hdr_01_perc = np.where(hdr_map > max_for_01_perc, 0, hdr_map)
        
        hdr_map = np.exp(hdr_map)
        hdr_map_01 = np.exp(hdr_01_perc)
        
        out = np.zeros(shape=hdr_map.shape, dtype=hdr_map.dtype)
        out_01 = np.zeros(shape=hdr_map_01.shape, dtype=hdr_map_01.dtype)
        
        cv2.normalize(hdr_map, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hdr_map_01, out_01, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        hdr_image[..., channel] = out
        hdr_image_01[..., channel] = out_01
        
        if channel == 0:
            print("\nChannel 'red' computed")
        if channel == 1:
            print("Channel 'green' computed")
        if channel == 2:
            print("Channel 'blue' computed\n")
        
    return hdr_image, hdr_image_01


def main(image_files, output_folder, exposure_times):

    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    log_exposure_times = np.log(exposure_times)
    hdr_image, hdr_image_01 = compute_hdr(img_stack, log_exposure_times)
    
    cv2.imwrite(path.join(output_folder, "output.png"), hdr_image)
    cv2.imwrite(path.join(output_folder, "output_01.png"), hdr_image_01)
    
    print("Completed")

if __name__ == "__main__":
    
    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = next(src_contents)

    image_dir = os.path.split(dirpath)[-1]
    print("Input images from: 'current folder/" + image_dir + "'")
    output_dir = OUT_FOLDER
    print("Output images to: 'current folder/" + output_dir + "'")
    
    image_files = sorted([os.path.join(dirpath, name) for name in fnames
                          if not name.startswith(".")])

    main(image_files, output_dir, EXPOSURE_TIMES)
    
    
# optional read images
#   
#   def read_images():
#    
#   files = ['images/memorial00.png', 'images/memorial01.png', 'images/memorial02.png', \
#         'images/memorial07.png',\
#        'images/memorial08.png', 'images/memorial09.png', 'images/memorial10.png', \
#        'images/memorial15.png']
#   
#   images = list([cv2.imread(f) for f in files])
#    
#   exposures = np.float32([1. / t for t in [0.03125, 0.0625, 0.125, 4, 8, 16, 32, 1024]])
#   
#   return images, exposures