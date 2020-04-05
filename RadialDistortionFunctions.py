import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from scipy.optimize import fsolve
import math
from random import randrange

def distort_image(img,k1,k2,k3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height,width=img.shape[:2]
    x_center=width//2
    y_center=height//2
    image = np.zeros((height,width), dtype=np.uint8)
    mappings=[]
    print(height,width)
    mapping_counter=0
    mapping_counter=0
    for y_pixel in range(0,height,1):
        for x_pixel in range(0,width,1):
            r_value=(((x_pixel-x_center)**2)+((y_pixel-y_center)**2))**0.5
            x_distorted=int((x_pixel-x_center)*(1+k1*(r_value**2)+k2*(r_value**4)+k3*(r_value**6)))
            y_distorted=int((y_pixel-y_center)*(1+k1*(r_value**2)+k2*(r_value**4)+k3*(r_value**6)))
            try:
                image[y_distorted+y_center,x_distorted+x_center]=img[y_pixel,x_pixel]
                if ((x_pixel==x_center//2) and (y_pixel==y_center-108 or y_pixel==y_center or y_pixel==y_center+108)) and mapping_counter<3:
                    #Just wanted 3 mappings.
                    mappings+=[[[x_pixel-x_center,y_pixel-y_center],[x_distorted,y_distorted]]]
                    mapping_counter+=1
            except:
                y=1
    cv2.imwrite('distortion_results/'+str(k1)+'+'+str(k2)+'+'+str(k3)+'.png',image)
    return image,mappings

def solve_for_k(k_values,mappings):
    if k_values==1:
        x0=[0]
    if k_values==2:
        x0=[0,0]
    elif k_values==3:
        x0=[0,0,0]
    data=[(k_values,mappings[0][0][0],mappings[0][0][1],mappings[0][1][0],mappings[0][1][1],0,0),
          (k_values,mappings[1][0][0],mappings[1][0][1],mappings[1][1][0],mappings[1][1][1],0,0),
          (k_values,mappings[2][0][0],mappings[2][0][1],mappings[2][1][0],mappings[2][1][1],0,0)]
    def undistort_y(k,*data):
        (k_values,x,y,xd,yd,xc,yc)=data[0][0]
        (k_values,x1,y1,xd1,yd1,xc,yc)=data[0][1]
        (k_values,x2,y2,xd2,yd2,xc,yc)=data[0][2]
        r=math.sqrt((xd-xc)**2+(yd-yc)**2)
        r1=math.sqrt((xd1-xc)**2+(yd1-yc)**2)
        r2=math.sqrt((xd2-xc)**2+(yd2-yc)**2)
        if k_values==1:
            F=np.empty((1))
            F[0]=(yd)*(1+k[0]*r**2)-(y-yd)
        elif k_values==2:
            F=np.empty((2))
            F[0]=(yd)*(1+k[0]*r**2+k[1]*r**4)-(y-yd)
            F[1]=(yd1)*(1+k[0]*r1**2+k[1]*r1**4)-(y1-yd1)
        else:
            F=np.empty((3))
            F[0]=(yd)*(1+k[0]*r**2+k[1]*r**4+k[2]*r**6)-(y-yd1)
            F[1]=(yd1)*(1+k[0]*r1**2+k[1]*r1**4+k[2]*r1**6)-(y1-yd1)
            F[2]=(yd2)*(1+k[0]*r2**2+k[1]*r2**4+k[2]*r2**6)-(y2-yd2)
        return F
    k=fsolve(undistort_y,x0,args=(data))
    return k

def undistort_image(image,k):
    if len(k)==1:
        k=[k[0],0,0]
    elif len(k)==2:
        k=[k[0],k[1],0]
    height,width=image.shape[:2]
    x_center=width//2
    y_center=height//2
    counter=0
    length_counter=0
    blank_image_2=np.zeros((height,width),dtype=np.uint8)
    for y_pixel in range(0,height,1):
        for x_pixel in range(0,width,1):
            r_value=(((x_pixel-x_center)**2)+((y_pixel-y_center)**2))**0.5
            x_corrected=(x_pixel-x_center)*(1+k[0]*(r_value**2))
            y_corrected=(y_pixel-y_center)*(1+k[0]*(r_value**2))
            try:
                blank_image_2[int(y_corrected)+y_pixel,int(x_corrected)+x_pixel]=image[y_pixel,x_pixel]
                counter+=1
            except:
                y=1
            length_counter+=1
    cv2.imwrite('undistortion_results/'+'k_values_considered_'+str(k[0])+'+'+str(k[1])+'+'+str(k[2])+'.png',blank_image_2)
    return blank_image_2,counter/length_counter

def iterate_for_best_version(image,k1,k2,k3):
    image2=image
    max=0
    max_mappings=[]
    for i in range(0,5,1):
        image,mappings=distort_image(image2,k1,k2,k3)
        k=solve_for_k(3,mappings)
        print(k)
        blank_image_2,ratio=undistort_image(image,k)
        if ratio>max:
            max=ratio
            k_values_chosen=k
            max_mappings=mappings
    print(max_mappings)
    blank_image_2,ratio=undistort_image(image,k)
    print(ratio)
    cv2.imwrite('undistortion_results/'+'k_values_considered_'+str(k[0])+'+'+str(k[1])+'+'+str(k[2])+'.png',blank_image_2)
    return True
