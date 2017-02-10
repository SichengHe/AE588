import numpy as np
import copy
#import math as math
#import pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import BFGS as BFGS
import trustRegion as TR

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


def traj(x):

	"2D trajectory of optimization"

	x1 = []
	x2 = []

	for i in xrange(len(x)):

		x1.append(x[i][0,0])
		x2.append(x[i][1,0])

	return x1, x2

def postProcess2D(x_low, x_up, Nx, y_low, y_up, Ny, f):
        
    f = f
    
    x_vec = np.linspace(x_low,x_up,Nx)
    y_vec = np.linspace(y_low,y_up,Ny)
    E_mat = np.matrix(np.zeros((Nx,Ny)))

    for i in xrange(Nx):

        x_loc = x_vec[i]   

        for j in xrange(Ny):

            y_loc = y_vec[j]

            E_mat[j,i] = f(np.matrix([[x_loc],[y_loc]]))

    fig = plt.figure()
    im = plt.imshow(E_mat, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(x_low, x_up, y_low, y_up))
    levels = np.arange(0.0, 5.0, 0.01) # by experiment
    CS = plt.contour(E_mat, levels,
                     origin='lower',
                     linewidths=2,
                     extent=(x_low, x_up, y_low, y_up))
        
        
        
        

    return fig




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
            
p1 = BFGS.BFGS(Matyas, Matyas_grad, x0, V0, dim_N, 1)
x_list, n_iter_list, log_g_norm_list = p1.optimize(10**-12)

x1_LS, x2_LS = traj(x_list)     
      
        
        
        
        
        
B_0 = np.matrix(np.eye(2)) 
x0 = np.matrix(np.zeros((2,1)))
x0[0, 0] = 1.0
x0[1, 0] = 0.3           
epsilon = 1e-12
        
Delta_0 = 1.0
Delta_max = 10.0
[x_list_TR, gk_norm_list] = TR.trustRegion(Delta_0, Delta_max, epsilon, x0, B_0, Matyas, Matyas_grad, 1)     
        

x1_TR, x2_TR = traj(x_list_TR)    

  

        
   
fig = postProcess2D(xmin, xmax, Nx, ymin, ymax, Ny, Matyas)  

plt.plot(x1_TR, x2_TR, '-rx') 
plt.plot(x1_LS, x2_LS, '-yo') 

plt.show()       
        
        
        
        
        
        
        
        
        
           
        
        

