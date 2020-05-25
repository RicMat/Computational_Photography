#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:55:44 2020

@author: Riccardo M. and Bunchalit E. 
"""

import cv2
import numpy as np
import sys

# set up the path 
path = "/Users/ebunchalit/Documents/CP_HW2/moving/"

# reading image
img1 = cv2.imread(path+ 'pic2.jpeg') #L
img2 = cv2.imread(path + 'pic1.jpeg') #C
#img3 = cv2.imread(path+ 'righ.jpeg') #R

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        #self.sift=cv2.xfeatures2d.SIFT_create()
        self.sift=cv2.xfeatures2d.SURF_create()
        self.smoothing_window_size=800

    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        
       
        img11 = img1.copy()
        kp11 = kp1.copy()
        img_point1 =cv2.drawKeypoints(img11,kp11,img11)
        
        
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, masks = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
       
        return H , masks

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        path = "/Users/ebunchalit/Documents/proj2f/ployy/"
        H , masks= self.registration(img1,img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        result=panorama1+panorama2
        
        cv2.imwrite(path + 'panorama1B.jpeg',panorama1)
        cv2.imwrite(path +'panorama2B.jpeg',panorama2)

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        
        return final_result
    
    
    def laplace(self,img1,img2):

        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        H , masks= self.registration(img1,img2) #LR
        
       # m = np.ones_like(img1, dtype='float32')
        m=np.ones((img1.shape[0], img1.shape[1],3),dtype='float32' )
        # to create white mask for visualization
        #m=m*255

        print (m.shape)
        
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))
        mask_left= cv2.warpPerspective(m, H, (width_panorama, height_panorama))
       
        #create overlap between mask and image
        mask_left[5:,5:,:] = mask_left[:-5,:-5,:]
        #cv2.imwrite('maskleft.jpeg', mask_left)
        
        lpb = self.Laplacian_blending(panorama2,panorama1,mask_left,4) 
                
        return lpb
    
    
    def Laplacian_blending(self,img1,img2,mask,levels=4):
        
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

        # generate Laplacian Pyramids for A,B and masks
        lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
        lp2  = [gp2[levels-1]]
        gpMr = [gpM[levels-1]]
        for i in range(levels-1,0,-1):
            print(i)
        
        #get the high frequencies
      
            L1 = np.subtract(gp1[i-1], cv2.pyrUp(gp1[i]))
            L2 = np.subtract(gp2[i-1], cv2.pyrUp(gp2[i]))
            lp1.append(L1)
            lp2.append(L2)
            gpMr.append(gpM[i-1]) # also reverse the masks

        #blend each level images 
        LS = []
        for l1,l2,gm in zip(lp1,lp2,gpMr):
            ls = l1 * gm + l2 * (1.0 - gm)
            LS.append(ls)

        # reconstruct
        ls_ = LS[0]
        for i in range(1,levels):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])
        
        return ls_


    def linear_blending(self,img1, img2, weight = 0.5):
        
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        H , masks= self.registration(img1,img2) #LR
        
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))

        output = np.zeros((height_panorama,width_panorama,3))
        x = output.shape[0]
        y = output.shape[1]
        z = output.shape[2]
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if panorama1[i][j][k]!=0 and panorama2[i][j][k] != 0:
                        output[i][j][k] = panorama1[i][j][k] * (1 - weight) + panorama2[i][j][k] * weight
                    else:
                        output[i][j][k] = panorama1[i][j][k] + panorama2[i][j][k]    
        
        return output


#Alpha blending
middle=Image_Stitching().blending(img1,img2)
cv2.imwrite(path+ 'alpha.jpeg',middle)


# =============================================================================
#for 3 images
# midimg = cv2.imread(path +'alpha.jpeg')
# final = Image_Stitching().blending(midimg,img3)
# cv2.imwrite(path+ 'panorama_alpha.jpeg', final)
# =============================================================================


#Laplace blending
middle_lpb = Image_Stitching().laplace(img1,img2)
cv2.imwrite(path+ 'laplace.jpeg', middle_lpb)

# =============================================================================
# midimg_lpb = cv2.imread(path+ 'middle_lpb.jpeg')
# final_lpb = Image_Stitching().laplace(midimg_lpb,img3)
# cv2.imwrite(path+ 'panorama_lpb.jpeg', final_lpb)
# =============================================================================


path = "/Users/ebunchalit/Documents/CP_HW2/linearpic/"
img1 = cv2.imread(path+ 'r22.jpeg.') #L
img2 = cv2.imread(path + 'r11.jpeg') #C

#Linear average
middle_lin = Image_Stitching().linear_blending(img1,img2,0.5)
cv2.imwrite(path+ 'simplelinr.jpeg', middle_lin)






