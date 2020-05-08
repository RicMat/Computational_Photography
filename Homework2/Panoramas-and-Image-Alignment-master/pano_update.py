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


img1 = cv2.imread('one1.jpeg')
img2 = cv2.imread('two2.jpeg')
img3 = cv2.imread('tre3.jpeg') 

# =============================================================================
# dog = cv2.imread('dog.png')
# mask =cv2.imread('mask.png')
# moon =cv2.imread('moon.png')
# =============================================================================





def Bonus_perspective_warping(img1, img2, img3): #central,  right left
    
    img1_b = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    (M, pts1, pts2, mask1) = getTransform(img3, img1_b,'homography')
    (M1, pts3, pts4, mask2) = getTransform(img2, img1_b,'homography')
        
    m = np.ones_like(img3, dtype='float32')
    #m = m*255
    m0 = np.zeros_like(img3, dtype='float32')
    m1 = np.ones_like(img2, dtype='float32')
    
# =============================================================================
#     mw = np.ones_like(img1, dtype='float32')
#     mw = mw*255
#     img1_wb = cv2.copyMakeBorder(mw,200,200,500,500, cv2.BORDER_CONSTANT)
#     
#     dummymask = np.ones_like(img1_wb, dtype='float32')
# =============================================================================
    
        
    
    out1 = cv2.warpPerspective(img3, M, (img1_b.shape[1],img1_b.shape[0]))   
    #cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) 
    out2 = cv2.warpPerspective(img2, M1, (img1_b.shape[1],img1_b.shape[0]))
    out3 = cv2.warpPerspective(m, M, (img1_b.shape[1],img1_b.shape[0]))
    out4 = cv2.warpPerspective(m1, M1, (img1_b.shape[1],img1_b.shape[0]))
    
# =============================================================================
#     for  i in range (dummymask.shape[0]):
#         for j in range (dummymask.shape[1]):
#             for k in range(3):
#                 if out3[i][j][k] ==255 or img1_wb[i][j][k]==255 :
#                     dummymask[i][j][k]= 255 
# 
#     cv2.imwrite('dummymask.png', dummymask)
# =============================================================================
    lpb = Laplacian_blending(out1,img1_b,out3,4)
    
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
    cv2.imwrite('m.png', m)
    cv2.imwrite('m0.png', m0)
    cv2.imwrite('lap.png', lpb)
    cv2.imwrite('img1.png', img1)
    #cv2.imwrite('dummymask.png', dummymask)
    
    return True


def getTransform(src, dst, method='homography'):
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


def Laplacian_blending(img1,img2,mask,levels=5):
    
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
        gp1.append(np.float64(G1))
        gp2.append(np.float64(G2))
        gpM.append(np.float64(GM))
# =============================================================================
#     for i in range(levels):
#         G1 = cv2.pyrDown(gp1[i])
#         G2 = cv2.pyrDown(G2)
#         GM = cv2.pyrDown(GM)
#         gp1.append(G1)
#         gp2.append(np.float32(G2))
#         gpM.append(np.float32(GM))
# =============================================================================
        # generate Laplacian Pyramids for A,B and masks
    lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp2  = [gp2[levels-1]]
    gpMr = [gpM[levels-1]]
    for i in range(levels-1,0,-1):
        print(i)
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        #L1 = np.subtract(gp1[i], cv2.pyrUp(gp1[i]))
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


def Bonus_perspective_warping_seamless(img1, img2, img3): #c r l    lcr312 f2
    
    #img1_b = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    (M, pts1, pts2, mask1) = getTransform(img3, img1,'homography')
    #(M1, pts3, pts4, mask2) = getTransform(img2, img1,'homography')
    
# =============================================================================
#     out1 = cv2.warpPerspective(img3, M, (img1_b.shape[1],img1_b.shape[0]))
#     out2 = cv2.warpPerspective(img2, M1, (img1_b.shape[1],img1_b.shape[0]))
# =============================================================================
    
    height_img1 = img3.shape[0]
    width_img1 = img3.shape[1]
    width_img2 = img1.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2
    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img3,img1,version='left_image')
    panorama1[0:img3.shape[0], 0:img3.shape[1], :] = img3
    panorama1 *= mask1
    mask2 = create_mask(img3,img1,version='right_image')
    panorama2 = cv2.warpPerspective(img1, M, (width_panorama, height_panorama))*mask2
    
    result=panorama1+panorama2
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    cv2.imwrite('img31.jpeg',final_result)
    
    img31 = cv2.imread('img31.jpeg')
    
    (M1, pts3, pts4, mask2) = getTransform(img31, img2,'homography')
    
    height_img1 = img31.shape[0]
    width_img1 = img31.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2
    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img31,img2,version='left_image')
    panorama1[0:img31.shape[0], 0:img31.shape[1], :] = img31
    panorama1 *= mask1
    mask2 = create_mask(img31,img2,version='right_image')
    panorama2 = cv2.warpPerspective(img2, M1, (width_panorama, height_panorama))*mask2
    
    result2=panorama1+panorama2
    rows, cols = np.where(result2[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result2 = result2[min_row:max_row, min_col:max_col, :]
    
    cv2.imwrite('seamless.png',final_result2)
    
    return True


def create_mask(img1,img2,version):
    smoothing_window_size = 800
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2
    offset = int(smoothing_window_size / 2)
    barrier = img1.shape[1] - int(smoothing_window_size / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version== 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    
    return cv2.merge([mask, mask, mask])


#Bonus_perspective_warping(img1, img2,img3)
Bonus_perspective_warping_seamless(img1, img2, img3)



#Bonus_perspective_warping_linear(input_image1, input_image2, input_image3)







