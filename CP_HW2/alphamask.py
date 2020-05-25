#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:47:47 2020

@author: Riccardo M. and Bunchalit E. 
"""

import cv2
import numpy as np


foreground = cv2.imread("GT05.png")
background = cv2.imread("drink-on-beach.jpg")
alpha = cv2.imread("GT05M.png")


foreground = foreground.astype(float)
background = background.astype(float)

alpha = alpha.astype(float)/255


foreground = cv2.multiply(alpha, foreground)

one = np.ones_like(alpha)
oma = one - alpha
background = cv2.multiply(oma, background)

outImage = cv2.add(foreground, background)

cv2.imwrite('alphamask_blen.png', outImage)