#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:16:19 2020

@authors: M. Riccardo - E-A. Bunchalit
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:43:22 2020

@author: darim
"""

import numpy as np
import cv2


# Read all the files with OpenCV
files = ['images/memorial00.png', 'images/memorial01.png', 'images/memorial02.png', 'images/memorial03.png',\
         'images/memorial04.png', 'images/memorial05.png', 'images/memorial06.png', 'images/memorial07.png',\
         'images/memorial08.png', 'images/memorial09.png', 'images/memorial10.png', 'images/memorial11.png',\
         'images/memorial12.png', 'images/memorial13.png', 'images/memorial14.png', 'images/memorial15.png']
images = list([cv2.imread(f) for f in files])
# Compute the exposure times in seconds
exposures = np.float32([1. / t for t in [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]])

# Compute the response curve
calibration = cv2.createCalibrateDebevec()
response = calibration.process(images, exposures)

# Compute the HDR image
merge = cv2.createMergeDebevec()
hdr = merge.process(images, exposures, response)

# Save it to disk
cv2.imwrite('hdr22_image.hdr', hdr)

durand = cv2.createTonemapDurand(gamma=2.5)
ldr = durand.process(hdr)

# Tonemap operators create floating point images with values in the 0..1 range
# This is why we multiply the image with 255 before saving

cv2.imwrite('durand_image.png', ldr * 255)