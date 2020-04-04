import numpy as np
from numpy.linalg import pinv
from numpy.linalg import det
'''array_1=np.reshape(np.array([[1,0],[0,1]]),(2,2))
array_2=np.reshape(np.array([2,2]),(2,1))
print(array_1.shape)
print(array_2.shape)
array_3=np.matmul(array_1,array_2)
print(array_3)

array_4=np.matmul(inv(array_1),array_3)
print(array_4)
print(array_4[1])
print(array_4[1][0])
print(array_4.shape)
comparison=array_4==array_2
print(comparison.all())'''

#------Sample inverses
#Height: 218, Width: 280
#Undistorted Point: (109,140) (height_pixel,width_pixel)
#Distorted Point: (109,140)
k1=-1e-5
array_1=np.reshape(np.array([[140,0,109,0]]),(4,1))
array_2=np.reshape(np.array([[1,k1,0,0],[0,0,0,0],[1,k1,0,0],[0,0,0,0]]),(4,4))
print(det(array_2))
array_4=np.matmul(pinv(array_2),array_1)
print(array_4)
