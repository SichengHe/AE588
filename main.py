import numpy as np
import copy
#import math as math
#import pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import BFGS as BFGS


def Matyas(x):
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def Matyas_grad(x):
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    
    dfdx1 = 0.52 * x1 - 0.48 * x2
    dfdx2 = 0.52 * x2 - 0.48 * x1
    
    return np.matrix([[dfdx1],[dfdx2]])

V0 = np.matrix(np.eye(2))         
x0 = np.matrix(np.zeros((2,1)))
x0[0, 0] = 1.0
x0[1, 0] = 0.3
dim_N = 2

xmin = -1.0
xmax = 1.0
ymin = -1.0
ymax = 1.0
Nx = 100
Ny = 100
            
p1 = BFGS.BFGS(Matyas, Matyas_grad, x0, V0, dim_N)
p1.optimize(10**-12)
p1.postProcess2D(xmin, xmax, Nx, ymin, ymax, Ny)
            
        
        
        
        
        
        
        
        
        
        
        

        

        
        
    
        
        
        
        
        
        
        
        
        
        
           
        
        

