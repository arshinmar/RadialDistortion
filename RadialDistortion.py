import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

'''# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
fname='calib_radial.jpg'
img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
print(ret,corners)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (7,6), corners2, ret)
cv2.imwrite('output.png',img)
fname='calib_radial.jpg'
img=cv2.imread(fname)
h,w = img.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(dist)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst=cv2.undistort(img,mtx,dist,None,newcameramtx)
cv2.imwrite('final.png',dst)'''

img=cv2.imread('test_image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height,width=img.shape
width_center=width//2
height_center=height//2
k1_value=-0.09
k2_value=-0.04
k3_value=0.0649
max_possible_r_value=(((width_center)**2)+((height_center)**2))**0.5
max_possible_height=(height)*(1+k1_value*(max_possible_r_value**2)+k2_value*(max_possible_r_value**4)+k3_value*(max_possible_r_value**6))
max_possible_width=(width)*(1+k1_value*(max_possible_r_value**2)+k2_value*(max_possible_r_value**4)+k3_value*(max_possible_r_value**6))
blank_image = np.zeros(shape=[round(max_possible_height),round(max_possible_width)], dtype=np.uint8)
#Blank black image which will be used later.
print(height,width)
for height_pixel in range(0,height+1,1):
    for width_pixel in range(0,width+1,1):
        r_value=(((width_pixel-width_center)**2)+((height_pixel-height_center)**2))**0.5

        x_distorted=(width_pixel)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
        y_distorted=(height_pixel)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
        blank_image[round(y_distorted)-1,round(x_distorted)-1]=img[round(height_pixel)-1,round(width_pixel)-1]
blank_image=cv2.resize(blank_image,(width,height))
cv2.imwrite('output.jpeg',blank_image)
