import numpy as np
from numpy.linalg import pinv
from numpy.linalg import det
from sympy import Symbol, nsolve
import sympy
import mpmath
#from scipy import opt
import numpy as np
from sympy import symbols, Eq, solve
import time
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

'''#------Sample inverses
#Height: 218, Width: 280
#Undistorted Point: (109,140) (height_pixel,width_pixel)
#Distorted Point: (109,140)
k1=-1e-5
array_1=np.reshape(np.array([[140,0,109,0]]),(4,1))
array_2=np.reshape(np.array([[1,k1,0,0],[0,0,0,0],[1,k1,0,0],[0,0,0,0]]),(4,4))
print(det(array_2))
array_4=np.matmul(pinv(array_2),array_1)
print(array_4)
'''
'''x_distorted=140
y_distorted=109
k1=-1e-5
mpmath.mp.dps=15
x=symbols('x y')
start=time.time()
function_1=Eq(x*(1+k1*((((x-140)**2)+((y-109)**2))**0.5))-x_distorted)
function_2=Eq(y*(1+k1*((((x-140)**2)+((y-109)**2))**0.5))-y_distorted)
print(solve((function_1,function_2),(x,y)))
end=time.time()
print(end-start)'''

from scipy.optimize import fsolve

x_distorted=200
y_distorted=109

def myFunction(z,x_distorted_y_distorted):
   x = z[0]
   y = z[1]
   k1=-1e-5

   F = np.empty((2))
   F[0] = (x*(1+k1*((((x-140)**2)+((y-109)**2))**0.5))-x_distorted)
   F[1] = (y*(1+k1*((((x-140)**2)+((y-109)**2))**0.5))-y_distorted)
   return F

zGuess = np.array([1,1])
z = fsolve(myFunction,zGuess,args=(y_distorted))
print(z[0])
