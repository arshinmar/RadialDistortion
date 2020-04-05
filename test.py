import cv2 as cv
import numpy as np
import math
from scipy.optimize import fsolve

def bdistort(x,k1,k2,k3,r):
    return x * (1 + k1*r**2 + k2*r**4 + k3*r**6)

def radius(xc,yc,x,y):
    return math.sqrt((x-xc)**2 + (y-yc)**2)

def undistort_y(k,*data):
    x,y,xd,yd,xc,yc = data
    r = math.sqrt((xd-xc)**2 + (yd-yc)**2)
    return yd*(1 + k*r**2) - y

###############################################################################

img1 = cv.imread('distorted-checker-board.png', -1)

rows = img1.shape[0] #height
cols = img1.shape[1] #width
xc = int(cols/2)
yc = int(rows/2)

#choosing the constants k
#kconstants contains sets of [k1,k2,k3], for easy testing
kconstants = [[0,0,0],
              [-1e-5,1e-12,1e-18],
              [1e-6,-1e-12,-1e-18],
              [-0.00000000000000006335,0.00000000000000018432,0.00000000000000013009],
              [1e-6,0,0]]
kselect = kconstants[1]
k1 = kselect[0]
k2 = kselect[1]
k3 = kselect[2]

img2 = np.zeros((rows,cols,3), np.uint8) #black image, for outputting distorted image
img3 = np.zeros((rows,cols,3), np.uint8)

img_filled = np.zeros((rows,cols,1), np.uint8)
for x in range(0,cols,1):
    for y in range(0,rows,1):
        img_filled[y,x] = 255

cv.imshow('Original Image', img1)
cv.waitKey(0) #Press esc or close window to continue
cv.destroyAllWindows()

mapping = []

#(y,x) is the coordinate of each pixel, with (0,0) defined at top left corner
#we should really be defining (0,0) at (yc,xc); I'll correct that below
for x in range(0,cols,1):
    for y in range(0,rows,1):
        r = radius(0,0,x-xc,y-yc)
        xd = int(bdistort(x-xc,k1,k2,k3,r))
        yd = int(bdistort(y-yc,k1,k2,k3,r))
        if xd >= -1*xc and xd < xc and yd >= -1*yc and yd < yc:
            img2[yd+yc,xd+xc] = img1[y,x]
            img_filled[yd+yc,xd+xc] = 0
            if x == 0:
                mapping += [[x-xc,xd,y-yc,yd]]

cv.imshow('Distorted', img2)
cv.waitKey(0)
cv.destroyAllWindows()

x = mapping[0][0]
xd = mapping[0][1]
y = mapping[0][2]
yd = mapping[0][3]
print(mapping)
print("y is " + str(y) + " and yd is " + str(yd))
data = (x,y,xd,yd,0,0)
x0 = 0
k = fsolve(undistort_y, x0, args=data)
print(k)

for xd in range(0,cols,1):
    for yd in range(0,rows,1):
        r = radius(0,0,xd-xc,yd-yc)
        x = int(bdistort(xd-xc,k[0],0,0,r))
        y = int(bdistort(yd-yc,k[0],0,0,r))
        if x >= -1*xc and x < xc and y >= -1*yc and y < yc:
            img3[y+yc,x+xc] = img2[yd,xd]

cv.imshow('Undistorted', img3)
cv.waitKey(0)
cv.destroyAllWindows()
