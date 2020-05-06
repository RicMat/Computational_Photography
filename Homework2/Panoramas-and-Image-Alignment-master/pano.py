#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:53:48 2020

@author: ebunchalit
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


input_image1 = cv2.imread('input1.png')
input_image2 = cv2.imread('input2.png')
input_image3 = cv2.imread('input3.png') 



def Bonus_perspective_warping(img1, img2, img3):
    
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    (M, pts1, pts2, mask1) = getTransform(img3, img1,'homography')
    (M1, pts3, pts4, mask2) = getTransform(img2, img1,'homography')
        
    m = np.ones_like(img3, dtype='float32')
    m1 = np.ones_like(img2, dtype='float32')
    
    
    out1 = cv2.warpPerspective(img3, M, (img1.shape[1],img1.shape[0]))
    out2 = cv2.warpPerspective(img2, M1, (img1.shape[1],img1.shape[0]))
    out3 = cv2.warpPerspective(m, M, (img1.shape[1],img1.shape[0]))
    out4 = cv2.warpPerspective(m1, M1, (img1.shape[1],img1.shape[0]))
        
    lpb = Laplacian_blending(out1,img1,out3,4)
    
    lpb1 = Laplacian_blending(out2,lpb,out4,4)
    cv2.imwrite('output_homography_lpb.png',lpb1)
    o=cv2.imread('output_homography_lpb.png',0)
    
    output_image = o 
    output_name = "lapblen.png"
    cv2.imwrite(output_name, output_image)
    
    cv2.imwrite('o1.png', out1)
    cv2.imwrite('o2.png', out2)
    cv2.imwrite('o3.png', out3)
    cv2.imwrite('o4.png', out4)
    
    return True


def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        #M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)


def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in range(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in range(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])


def Laplacian_blending(img1,img2,mask,levels=4):
    
    G1 = img1.copy()
    G2 = img2.copy()
    GM = mask.copy()
        
    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]
    
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        GM = cv2.pyrDown(GM)
        gp1.append(np.float32(G1))
        gp2.append(np.float32(G2))
        gpM.append(np.float32(GM))

        # generate Laplacian Pyramids for A,B and masks
    lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp2  = [gp2[levels-1]]
    gpMr = [gpM[levels-1]]
    for i in range(levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        L1 = np.subtract(gp1[i-1], cv2.pyrUp(gp1[i]))
        L2 = np.subtract(gp2[i-1], cv2.pyrUp(gp2[i]))
        lp1.append(L1)
        lp2.append(L2)
        gpMr.append(gpM[i-1]) # also reverse the masks

        # Now blend images according to mask in each level
    LS = []
    for l1,l2,gm in zip(lp1,lp2,gpMr):
        ls = l1 * gm + l2 * (1.0 - gm)
        LS.append(ls)

        # now reconstruct
    ls_ = LS[0]
    for i in range(1,levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

def Linear_blending(img1, img2, weight):

    output = np.zeros(img1.shape)
    x = img1.shape[0]
    y = img1.shape[1]
    z = img1.shape[2]
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if img1[i][j][k]!=0 and img2[i][j][k] != 0:
                    output[i][j][k] = img1[i][j][k] * (1 - weight) + img2[i][j][k] * weight
                else:
                    output[i][j][k] = img1[i][j][k] + img2[i][j][k]    
    return output


def Bonus_perspective_warping_linear(img1, img2, img3):
    
    
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    (M, pts1, pts2, mask1) = getTransform(img3, img1,'homography')
    (M1, pts3, pts4, mask2) = getTransform(img2, img1,'homography')
    
    out1 = cv2.warpPerspective(img3, M, (img1.shape[1],img1.shape[0]))
    out2 = cv2.warpPerspective(img2, M1, (img1.shape[1],img1.shape[0]))
    
    
    #result = Linear_blending(img1, out1, weight = 0.5)
    result_full = Linear_blending(Linear_blending(img1, out1, weight = .5), out2, weight = .5)
    #cv2.imwrite('linearblen.png',result)
    cv2.imwrite('linearblen_full.png',result_full)
    
    return True



Bonus_perspective_warping(input_image1, input_image2, input_image3)

Bonus_perspective_warping_linear(input_image1, input_image2, input_image3)







