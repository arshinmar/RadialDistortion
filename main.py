from RadialDistortionFunctions import *
img=cv2.imread('distorted-checker-board.png')
k1=-1e-5
k2=1e-14#1e-12
k3=1e-15

image,mappings=distort_image(img,k1,k2,k3)
k=solve_for_k(1,mappings)
print(k)
blank_image_2,ratio=undistort_image(image,k)
print(ratio)
