import cv2
import numpy as np
import os
import pandas as pd
# black blank image

img=cv2.imread('unnamed.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height,width=img.shape
width_center=width//2
height_center=height//2
blank_image = np.zeros(shape=[height,width], dtype=np.uint8)
print(height,width)
for height_pixel in range(0,height+1,1):
    for width_pixel in range(0,width+1,1):
        r_value=(((width_pixel-width_center)**2)+((height_pixel-height_center)**2))**0.5
        k1_value=0.0000014
        k2_value=0.000000000000000001
        k3_value=0.00000000000000001
        x_distorted=(width_pixel)*(k1_value*(r_value**2))#+k2_value*(r_value**4)+k3_value*(r_value**6))
        y_distorted=(height_pixel)*(k1_value*(r_value**2))#+k2_value*(r_value**4)+k3_value*(r_value**6))
        blank_image[round(y_distorted)-1,round(x_distorted)-1]=img[round(height_pixel)-1,round(width_pixel)-1]
cv2.imwrite('output.jpeg',blank_image)
