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
k1_value=-5e-6
k2_value=1e-12
k3_value=1e-15
blank_image = np.zeros((height,width), dtype=np.uint8)
mappings=[]
print(height,width)
mapping_counter=0
for height_pixel in range(0,height+1,1):
    for width_pixel in range(0,width+1,1):
        r_value=(((width_pixel-width_center)**2)+((height_pixel-height_center)**2))**0.5
        x_distorted=(width_pixel-width_center)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
        y_distorted=(height_pixel-height_center)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
        try:
            blank_image[round(y_distorted)+height_center,round(x_distorted)+width_center]=img[height_pixel,width_pixel]
            if height_pixel==0 and width_pixel>=width_center and mapping_counter<=3:
                mappings+=[[[width_pixel,height_pixel],[x_distorted+width_center,y_distorted+height_center]]]
                mapping_counter+=1
        except:
            y=1
#blank_image=cv2.resize(blank_image,(width,height))
cv2.imwrite("check4.png",blank_image)

def undistortion(blank_image,mappings,k_values,height=218,width=280,height_center=109,width_center=140):
    def myFunction(z,k_values,width_center,height_center,mappings):
        if k_values==1:
            k1=z[0]
            F = np.empty((1))
            r_value=(((mappings[0][0][0]-140)**2)+((0-109)**2))**0.5
            F[0] = (mappings[0][1][0])*(1+k1*(r_value**2))-(mappings[0][0][0])
            return F
        elif k_values==2:
            k1=z[0]
            k2=z[1]
            F=np.empty((2))
            r_value=(((mappings[0][0][0]-140)**2)+((0-109)**2))**0.5
            F[0] = (mappings[0][1][0])*(1+k1*(r_value**2)+k2*(r_value**4))-(mappings[0][0][0]-width_center)
            r_value=(((mappings[1][0][0]-140)**2)+((0-109)**2))**0.5
            F[1] = (mappings[1][1][0])*(1+k1*(r_value**2)+k2*(r_value**4))-(mappings[1][0][0]-width_center)
            return F
        elif k_values==3:
            k1=z[0]
            k2=z[1]
            k3=z[2]
            F=np.empty((3))
            r_value=(((mappings[0][0][0]-140)**2)+((0-109)**2))**0.5
            F[0] = (mappings[0][1][0])*(1+k1*(r_value**2)+k2*(r_value**4))-(mappings[0][0][0]-width_center)
            r_value=(((mappings[1][0][0]-140)**2)+((0-109)**2))**0.5
            F[1] = (mappings[1][1][0])*(1+k1*(r_value**2)+k2*(r_value**4))-(mappings[1][0][0]-width_center)
            r_value=(((mappings[2][0][0]-140)**2)+((0-109)**2))**0.5
            F[2] = (mappings[2][1][0])*(1+k1*(r_value**2)+k2*(r_value**4)+k3*(r_value**6))-(mappings[2][0][0]-width_center)
            return F
    if k_values==1:
        zGuess=np.array([1])
    z=fsolve(myFunction,zGuess,args=(k_values,width_center,height_center,mappings))
    print(z)
    k1_value=z[0]
    if k_values==1:
        k2_value=0
        k3_value=0
    counter=0
    length_counter=0
    blank_image_2=np.zeros((height,width),dtype=np.uint8)
    for height_pixel in range(0,height+1,1):
        for width_pixel in range(0,width+1,1):
            r_value=(((width_pixel-width_center)**2)+((height_pixel-height_center)**2))**0.5
            x_corrected=(width_pixel-width_center)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
            y_corrected=(height_pixel-height_center)*(1+k1_value*(r_value**2)+k2_value*(r_value**4)+k3_value*(r_value**6))
            try:
                blank_image_2[int(y_corrected)+height_center,int(x_corrected)+width_center]=blank_image[height_pixel,width_pixel]
                counter+=1
            except:
                y=1
            length_counter+=1
    cv2.imwrite('undistortion_results/'+'k_values_considered_'+str(k_values)+'.png',blank_image_2)
    print(counter/length_counter)
    return True

#Undistortion for solely k1:
blank_image_k1=undistortion(blank_image,mappings,1)
