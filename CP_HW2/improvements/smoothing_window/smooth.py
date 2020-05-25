import cv2
import numpy as np

lion = cv2.imread('1.jpg',3)
taj = cv2.imread('2.jpg',3)
alpha = cv2.imread('angryimg_20.png',0).astype(np.float32)


a_B, a_G, a_R = cv2.split(lion)
b_B, b_G, b_R = cv2.split(taj)

b = (a_B * (alpha/255.0)) + (b_B * (1.0 - (alpha/255.0)))
g = (a_G * (alpha/255.0)) + (b_G * (1.0 - (alpha/255.0)))
r = (a_R * (alpha/255.0)) + (b_R * (1.0 - (alpha/255.0)))
output = cv2.merge((b,g,r))


cv2.imwrite("output.png", output)