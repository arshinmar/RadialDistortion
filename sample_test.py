import numpy as np
from numpy.linalg import inv
array_1=np.reshape(np.array([[1,0],[0,1]]),(2,2))
array_2=np.reshape(np.array([2,2]),(2,1))
print(array_1.shape)
print(array_2.shape)
array_3=np.matmul(array_1,array_2)
print(array_3)

array_4=np.matmul(inv(array_1),array_3)
print(array_4)
print(array_4.shape)
comparison=array_4==array_2
print(comparison.all())
