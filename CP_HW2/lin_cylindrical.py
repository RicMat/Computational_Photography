# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:28:06 2020

@author: Riccardo M. and Bunchalit E. 
"""

import cv2
import numpy as np
import sys

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        self.sift=cv2.xfeatures2d.SIFT_create()
        self.smoothing_window_size=400

    def registration(self,img1,img2):
        if (img1.shape[1] == 608):
            kp1, des1 = self.sift.detectAndCompute(img1, None)
            kp2, des2 = self.sift.detectAndCompute(img2, None)
        else:
            image8bit1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            image8bit2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            kp1, des1 = self.sift.detectAndCompute(image8bit1, None)
            kp2, des2 = self.sift.detectAndCompute(image8bit2, None)
            
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
        return H

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        smoothing_window_size = self.smoothing_window_size
        if (img1.shape[1] > 608):
            smoothing_window_size = 400
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

    def blending(self,img1,img2):
        
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        
        width_panorama = width_img1 +width_img2
        
        print("{}     {}      {}".format(height_img1, width_img1, width_img2))
        
        H = self.registration(img1,img2)
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        cv2.imwrite("pan1.jpg",panorama1)
        cv2.imwrite("pan2.jpg",panorama2)
        print("{}".format(panorama2.shape))
            
        result=panorama1+panorama2
        print("{}".format(result.shape))
        
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        print("{}".format(final_result.shape))
        return final_result
   
def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
  

"""
    
def main(img1,img2):
    final=Image_Stitching().blending(img1,img2)
    cv2.imwrite('panorama.jpg', final)

if __name__ == '__main__':
    input_image1 = cv2.imread('q11.jpg')
    input_image2 = cv2.imread('q22.jpg')
    
    main(input_image1,input_image2)
    
"""

"""
    
if __name__ == '__main__':
    input_image1 = cv2.imread('input1.png')
    input_image2 = cv2.imread('input2.png')
    
    main(input_image1,input_image2)
    
"""
    
#"""
    
def boia(img1,img2):
    final=Image_Stitching().blending(img1,img2)
    #cv2.imwrite('panorama.jpg', final)
    return final

if __name__ == '__main__':
    
    img1 = cv2.imread('input1.png')#center
    img2 = cv2.imread('input2.png')#right
    img3 = cv2.imread('input3.png')#left
    
    h, w = img1.shape[:2]
    K = np.array([[800,0,w/2],[0,800,h/2],[0,0,1]]) # mock intrinsics
    img_cy1 = cylindricalWarp(img1, K)
    cv2.imwrite("image_cyl1.png", img_cy1)
    img_cy2 = cylindricalWarp(img2, K)
    cv2.imwrite("image_cyl2.png", img_cy2)
    img_cy3 = cylindricalWarp(img3, K)
    cv2.imwrite("image_cyl3.png", img_cy3)
    
    img_cy3=cv2.imread('image_cyl3.png')
    img_cy1=cv2.imread('image_cyl1.png')
    img_cy2=cv2.imread('image_cyl2.png')
    left = boia(img_cy3,img_cy1)
    cv2.imwrite('leftt.jpg', left)
    right = boia(img_cy1,img_cy2)
    cv2.imwrite('rightt.jpg', right)
    left = cv2.imread('leftt.jpg')
    right =cv2.imread('rightt.jpg')
    final = boia(left,right)
    cv2.imwrite('final.jpg', final)
    
#"""