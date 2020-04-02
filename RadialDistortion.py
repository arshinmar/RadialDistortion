import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

img=cv2.imread('IMG_1807.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
height,width=img.shape[:2]
width_center=width//2
height_center=height//2
k1_value=-1e-6
#k2_value=1e-12
k3_value=1e-18
max_possible_r_value=(((width_center)**2)+((height_center)**2))**0.5
max_possible_height=(height)*(1+k1_value*(max_possible_r_value**2))#+k2_value*(max_possible_r_value**4))#+k3_value*(max_possible_r_value**6))
max_possible_width=(width)*(1+k1_value*(max_possible_r_value**2))#+k2_value*(max_possible_r_value**4))#+k3_value*(max_possible_r_value**6))
blank_image = np.zeros((height,width), dtype=np.uint8)
max_possible_height_center=max_possible_height//2
max_possible_width_center=max_possible_width//2
#Blank black image which will be used later.
print(height,width)
for height_pixel in range(0,height+1,1):
    for width_pixel in range(0,width+1,1):
        r_value=(((width_pixel-width_center)**2)+((height_pixel-height_center)**2))**0.5
        x_distorted=(width_pixel-width_center)*(1+k1_value*(r_value**2))#+k2_value*(r_value**4))#+k3_value*(r_value**6))
        y_distorted=(height_pixel-height_center)*(1+k1_value*(r_value**2))#+k2_value*(r_value**4))#+k3_value*(r_value**6))
        try:
            blank_image[round(y_distorted)+height_center,round(x_distorted)+width_center]=img[height_pixel-1,width_pixel-1]
        except:
            '''print("Distorted Height",round(y_distorted)-2+height_center)
            print("Distorted Width",round(x_distorted)-2+width_center)
            print('Height',height)
            print('Width',width)'''
#blank_image=cv2.resize(blank_image,(width,height))
cv2.imwrite('output2.png',blank_image)
