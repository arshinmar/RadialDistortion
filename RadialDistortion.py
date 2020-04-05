import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from scipy.optimize import fsolve

img=cv2.imread('distorted-checker-board.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
height,width=img.shape[:2]
width_center=width//2
height_center=height//2
k1_value=-1e-5
k2_value=1e-10
k3_value=2e-17
blank_image = np.zeros((height,width), dtype=np.uint8)
'''max_possible_height_center=max_possible_height//2
max_possible_width_center=max_possible_width//2'''
#Blank black image which will be used later.
print(height,width)
for height_pixel in range(0,height+1,1):
    for width_pixel in range(0,width+1,1):
        r_value=(((width_pixel-width_center)**2)+((height_pixel-height_center)**2))**0.5
        x_distorted=(width_pixel-width_center)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
        y_distorted=(height_pixel-height_center)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
        try:
            blank_image[round(y_distorted)+height_center,round(x_distorted)+width_center]=img[height_pixel-1,width_pixel-1]
        except:
            '''print("Distorted Height",round(y_distorted)-2+height_center)
            print("Distorted Width",round(x_distorted)-2+width_center)
            print('Height',height)
            print('Width',width)'''
#blank_image=cv2.resize(blank_image,(width,height))
cv2.imwrite("check.png",blank_image)

def myFunction(z,x_distorted,y_distorted):
   x = z[0]
   y = z[1]
   k1=-1e-5

   F = np.empty((2))
   F[0] = (x-140)*(1+k1*((((x-140)**2)+((y-109)**2))**2)-(x_distorted-140))
   F[1] = (y-109)*(1+k1*((((x-140)**2)+((y-109)**2))**2)-(y_distorted-109))
   return F
counter=0
length_counter=0
blank_image_2=np.zeros((height,width),dtype=np.uint8)
for height_pixel in range(0,height+1,1):
    for width_pixel in range(0,width+1,1):
        zGuess = np.array([1,1])
        z = fsolve(myFunction,zGuess,args=(width_pixel,height_pixel))
        #print(round(z[1]))
        try:
            blank_image_2[int(round(z[1]))+height_center,int(round(z[0]))+width_center]=blank_image[height_pixel-1,width_pixel-1]
            counter+=1
        except:
            length_counter+=1
cv2.imwrite('check2.png',blank_image_2)
print(counter/length_counter)
