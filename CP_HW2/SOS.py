#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:46:23 2020

@author: Riccardo M. and Bunchalit E. 
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


path = "/Users/ebunchalit/Documents/Proj2/building12/"
# Read image 
img1 = cv2.imread(path +'q11.jpg')
img2 = cv2.imread(path + 'q22.jpg') 

# SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Convering to Gray
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#keypoints with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# no.of keypoints 
print(len(des1))
print(len(des2))

# BFMatcher 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Create images with keypoints
img1=cv2.drawKeypoints(img1,kp1,img1)
cv2.imwrite(path + 'sift_keypoints1.jpg',img1)

img2=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite(path + 'sift_keypoints2.jpg',img2)

# calculate ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print("matches sift ")
print(len(good))

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imwrite(path+'matching.jpg', img3)




#ORB
orb = cv2.ORB_create()

# Convering to Gray
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# keypoints with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# no.of keypoints 
print(len(des1))
print(len(des2))

# BFMatcher 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Create images with keypoints
img1=cv2.drawKeypoints(img1,kp1,img1)
cv2.imwrite(path +'2orb_keypoints.jpg',img1)

img2=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite(path +'3orb_keypoints.jpg',img2)

#calculate ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print("matches orb")
print(len(good))



# SURF 
surf = cv2.xfeatures2d.SURF_create()

# Convering to Gray
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# keypoints  with SURF
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

# no.of keypoints 
print(len(des1))
print(len(des2))

# BFMatcher 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Create images with keypoints
img1=cv2.drawKeypoints(img1,kp1,img1)
cv2.imwrite(path +'2surf_keypoints.jpg',img1)

img2=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite(path +'3surf_keypoints.jpg',img2)

# claculate ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print("matches SURF ")
print(len(good))