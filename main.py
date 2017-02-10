import numpy as np
import copy

from scipy.optimize import minimize

import BFGS as BFGS
import trustRegion as TR
import CG as CG

import matplotlib.pyplot as plt
import matplotlib.cm as cm

""" function """

# problem 1
def Matyas_class():

	return Matyas, Matyas_grad


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

# to make scipy happy
def Matyas_scipy(x):
    
    x1 = x[0]
    x2 = x[1]
    
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def Matyas_grad_scipy(x):
    
    x1 = x[0]
    x2 = x[1]
    
    dfdx1 = 0.52 * x1 - 0.48 * x2
    dfdx2 = 0.52 * x2 - 0.48 * x1
    
    return np.array([dfdx1, dfdx2])


# problem 2

def Rosen_class():

	return Rosen, Rosen_grad


def Rosen(x):

	n = x.shape[0]

	J = 0.0

	for i in xrange(n-1):

		x_i = x[i, 0]
		x_ip1 = x[i+1, 0]

		J += 100.0 * (x_ip1 - x_i**2)**2 + (1.0 - x_i)**2

	return J

def Rosen_grad(x):

	n = x.shape[0]

	dJdx = np.matrix( np.zeros( (n,1) ) )

	for i in xrange(n-1):

		x_i = x[i, 0]
		x_ip1 = x[i+1, 0]

		dJdxi_loc = 200.0 * (x_ip1 - x_i**2) * (-2.0 * x_i) + \
					2.0 * (1.0 - x_i) * (-1.0)
		dJdxip1_loc = 200.0 * (x_ip1 - x_i**2) * 1.0


		dJdx[i, 0] += dJdxi_loc
		dJdx[i+1, 0] += dJdxip1_loc

	return dJdx

# to make scipy happy
def Rosen_scipy(x):

	n = len(x)

	J = 0.0

	for i in xrange(n-1):

		x_i = x[i]
		x_ip1 = x[i+1]

		J += 100.0 * (x_ip1 - x_i**2)**2 + (1.0 - x_i)**2

	return J

def Rosen_grad_scipy(x):

	n = len(x)

	dJdx = np.zeros(n)

	for i in xrange(n-1):

		x_i = x[i]
		x_ip1 = x[i+1]

		dJdxi_loc = 200.0 * (x_ip1 - x_i**2) * (-2.0 * x_i) + \
					2.0 * (1.0 - x_i) * (-1.0)
		dJdxip1_loc = 200.0 * (x_ip1 - x_i**2) * 1.0


		dJdx[i] += dJdxi_loc
		dJdx[i+1] += dJdxip1_loc

	return dJdx


# x0 = np.matrix(np.zeros((2,1)))
# x0[0, 0] = 1.0
# x0[1, 0] = 0.3

x0 = [1.0, 0.3]
res = minimize(Rosen_scipy, x0, method='BFGS', jac=Rosen_grad_scipy,
	options={'gtol': 1e-6, 'disp': True})
print("res",res)


# x = np.matrix(np.zeros((2,1)))
# n = 2
# x[0, 0] = 1.0
# x[1, 0] = 2.0

# J = Rosen(x)

# x_perturb_1 = copy.deepcopy(x)
# x_perturb_1[0, 0] += 1e-8
# J_perturb_1 = Rosen(x_perturb_1)

# x_perturb_2 = copy.deepcopy(x)
# x_perturb_2[1, 0] += 1e-8
# J_perturb_2 = Rosen(x_perturb_2)

# dJ_finte = np.matrix(np.zeros((2,1)))
# dJ_finte[0, 0] = (J_perturb_1 - J)/ 1e-8
# dJ_finte[1, 0] = (J_perturb_2 - J)/ 1e-8

# print("dJ_finte", dJ_finte, "Rosen_grad", Rosen_grad(x)) 





""" post process """
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












def uncon(func, x0, max_iter, tol):

	method = "BFGS_LS"
	mode = 0  # solve mode
	#mode = 1 # analysis mode

	# get the obj function and gradient

	f, g = func()

	n = x0.shape[0]	

	if (method == "BFGS_LS"):

		V0 = np.matrix(np.eye(n))

		p = BFGS.BFGS(f, g, x0, V0, n, mode)

		if (mode == 0):

			x, J = p.optimize(tol)

		elif (mode == 1):

			x_list, n_iter_list, log_g_norm_list = p.optimize(tol)

	elif (method == "BFGS_TR"):

		B_0 = np.matrix(np.eye(n))

		Delta_0 = 1.0
		Delta_max = 10.0

		if (mode == 0):

			x, J = TR.trustRegion(Delta_0, Delta_max, tol, \
				x0, B_0, f, g, mode) 

		elif (mode == 1):

			x_list, n_iter_list, log_g_norm_list = TR.trustRegion(Delta_0, Delta_max, tol, \
				x0, B_0, f, g, mode) 


	elif (method == "CG"):

		p = CG.CG(f, g, x0, mode)   
		if (mode == 0):

			x, J = p.optimize(tol)

		elif (mode == 1):

			x_list, n_iter_list, log_g_norm_list = p.optimize(tol)



	if (mode == 0):

		return x, J

	elif (mode == 1):

		return x_list, n_iter_list, log_g_norm_list







