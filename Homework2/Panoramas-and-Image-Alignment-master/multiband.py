#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:03:15 2020

@author: ebunchalit
"""

#!/usr/bin/python
import numpy as np
import cv2
import sys

flag_half = True
img1 = cv2.imread('r.jpg')
img2 = cv2.imread('l.jpg')
overlap_w = 200
leveln = 9


def preprocess(img1, img2, overlap_w, flag_half):
    if img1.shape[0] != img2.shape[0]:
        print ("error: image dimension error")
        sys.exit()
    if overlap_w > img1.shape[1] or overlap_w > img2.shape[1]:
        print ("error: overlapped area too large")
        sys.exit()

    w1 = img1.shape[1]
    w2 = img2.shape[1]
    img1 = img1.astype('int')

    if flag_half:
        shape = np.array(img1.shape)
        shape[1] = w1 / 2 + w2 / 2

        subA = np.zeros(shape)
        subA[:, :int(w1 / 2 + overlap_w / 2)] = img1[:, :int(w1 / 2 + overlap_w / 2)]
        subB = np.zeros(shape)
        subB[:, int(w1 / 2 - overlap_w / 2):] = img2[:, w2 - int(w2 / 2 + overlap_w / 2):]
        mask = np.zeros(shape)
        mask[:, :int(w1 / 2)] = 1
    else:
        shape = np.array(img1.shape)
        shape[1] = w1 + w2 - overlap_w

        subA = np.zeros(shape)
        subA[:, :w1] = img1
        subB = np.zeros(shape)
        subB[:, w1 - overlap_w:] = img2
        mask = np.zeros(shape)
        mask[:, :int(w1 - overlap_w / 2)] = 1

    return subA, subB, mask


def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP


def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        LP.append(img - cv2.pyrUp(next_img, img.shape[1::-1]))
        img = next_img
    LP.append(img)
    return LP


def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended


def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, lev_img.shape[1::-1])
        img += lev_img
    return img


def multi_band_blending(img1, img2, overlap_w, leveln=None, flag_half=False):
    if overlap_w < 0:
        print ("error: overlap_w should be a positive integer")
        sys.exit()

    subA, subB, mask = preprocess(img1, img2, overlap_w, flag_half)

    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1],
                                          img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print ("warning: inappropriate number of leveln")
        leveln = max_leveln

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result





result = multi_band_blending(img1, img2, overlap_w, leveln, flag_half)
cv2.imwrite('resultRL.png', result)