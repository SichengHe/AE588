import numpy as np
import copy
#import math as math
#import pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm


mu1 = 1e-4
mu2 = 0.9

def lineSearch(f, g, x0, pk, alpha_1, alpha_max):
    
    alpha = []
    alpha = [0.0, alpha_1]
    
    phi_alpha_0 = f(x0)
    dphi_dalpha_0 = np.transpose(g(x0)).dot(pk)[0,0]
    
    i = 1
    while (1 == 1):
        
        print("i:",i)
        
        if (i >= 100):
            break
        
        x = x0 + alpha[i] * pk
        
        phi_alpha_i = f(x)
        
        if (phi_alpha_i > phi_alpha_0 + mu1 * alpha[i] * dphi_dalpha_0\
           or (alpha[i]>alpha[i-1] and i>1)):
            
            print("alpha[i-1], alpha[i]", alpha[i-1], alpha[i])
            
            alpha_star = zoom(alpha[i-1], alpha[i], f, g, x0, pk)
            
            return alpha_star
        
        dphi_dalpha_i = np.transpose(g(x0 + alpha[i]*pk)).dot(pk)[0,0]
        
        if (abs(dphi_dalpha_i) <= - mu2 * dphi_dalpha_0):
            
            return alpha[i]
        
        elif (dphi_dalpha_i >= 0.0):
            
            print("alpha[i], alpha[i-1]", alpha[i], alpha[i-1])
            
            alpha_star = zoom(alpha[i], alpha[i-1], f, g, x0, pk)
            
            return alpha_star
        
        else:
            
            alpha_new = 0.5 * (alpha_max + alpha[i])
            alpha.append(alpha_new)
            
        i += 1
        
        
        
def zoom(alpha_low, alpha_high, f, g, x0, pk):
    
    j = 0
    
    alpha = []
    
    phi_0 = f(x0)
    dphi_dalpha_0 = np.transpose(g(x0)).dot(pk)[0,0]
    
    while (1 == 1):
        
        print("alpha_low, alpha_high", alpha_low, alpha_high)
        
        print("j:",j+1)
        
        if (j >= 100):
            break
        
        alpha_j = (alpha_low + alpha_high)/2.0
        alpha.append(alpha_j)
        
        phi_alpha_j = f(x0 + alpha[j]*pk)
        
        if (phi_alpha_j > phi_0 + mu1 * alpha[j] * dphi_dalpha_0\
           or phi_alpha_j > f(x0 + alpha_low*pk)):
            
            alpha_high = alpha[j]
            
        else:
            
            dphi_dalpha_j = np.transpose(g(x0 + alpha[j]*pk)).dot(pk)[0,0]

            if (abs(dphi_dalpha_j) <= -mu2*dphi_dalpha_0):
                
                alpha_star = alpha[j]
                
                return alpha_star
            
            elif (dphi_dalpha_j * (alpha_high - alpha_low) >= 0):
                
                alpha_high = alpha_low
                
            alpha_low = alpha[j]
            
        j += 1


class BFGS(object):
    
    """ main function for hw2 for unconstrained optimization. 
    Implementation of BFGS method.  """

    def __init__(self, f, g, x0, V0, dim_N):
        
        """Initialization: 
            f: obj
            g: d obj/ dx
            x0: initial pt
            V0: initial Hessian (inv)
            dim_N: problem size
            """
        
        # stores the history of x (helps solve for sk and postprocess)
        self.x_list = [x0]
       
        # recored the convergence history 
        self.g_norm_list = []

        # inverse of Hessian 
        self.V_k = V0
        
        # gradient of obj of current and next step
        self.g_k = np.matrix(np.zeros((dim_N, 1)))
        self.g_kp1 = np.matrix(np.zeros((dim_N, 1)))
        
        # obj and gradient obj
        self.f = f
        self.g = g
        
        # dimension
        self.dim_N = dim_N
        
    def gradDescentDir(self):
        
        """gradient descent direction (for 1st iter)"""
        
        x0 = self.x_list[0]

        g = self.g

        g_k = g(x0)
        
        self.g_k = g_k
        
        return g_k
        
    def gradDescent(self):
        
        """main function for gradient descent. 
        it returns the next pt and the gradient there"""
        
        f = self.f
        g = self.g
        
        x_k = self.x_list[-1]
        
        alpha_1 = 1.0
        alpha_max = 1000.0
        
        # finding the descent direction
        g_k = self.gradDescentDir()
        
        # line search
        alpha_star = lineSearch(f, g, x_k, -g_k, alpha_1, alpha_max)
        
        # get the new point and its corresponding gradient 
        x_kp1 = x_k + alpha_star * (-g_k)
        
        g_kp1 = g(x_kp1)
        
        # record new point and its corresponding gradient 
        self.x_list.append(x_kp1)
        self.g_kp1 = g_kp1
        
        # record the gradient norm
        g_norm = np.sqrt((np.transpose(self.g_kp1).dot(self.g_kp1))[0,0])
        
        self.g_norm_list.append(g_norm)
        
        
        
    def approxInvHessian_update(self):
        
        """BFGS update refer to MDO notes 4.7.2 (4.46)"""
        
        x_k = self.x_list[-2]
        x_kp1 = self.x_list[-1]
        g_k = self.g_k
        g_kp1 = self.g_kp1
        
        V_k = self.V_k
        
        dim_N = self.dim_N
        
        
        s_k = x_kp1 - x_k
        y_k = g_kp1 - g_k
        
        I = np.matrix(np.identity(dim_N))
        
        sk_ykT = s_k.dot(np.transpose(y_k))
        yk_skT = y_k.dot(np.transpose(s_k))
        sk_skT = s_k.dot(np.transpose(s_k))
        
        skT_yk = np.transpose(s_k).dot(y_k)[0,0]

        Vk_left = I - sk_ykT * (1.0 / skT_yk)
        Vk_right = I - yk_skT * (1.0 / skT_yk)
        const = sk_skT * (1.0 / skT_yk)
        
        V_kp1 = Vk_left.dot(V_k).dot(Vk_right) + const

        self.V_k = V_kp1
        
    def newtonDir(self):
        
        """get the Newton direction"""
        
        g_k = self.g_kp1   
        V_k = self.V_k
        
        p_k = V_k.dot(-g_k)
        
        return p_k
        
    def newton(self):
        
        """main function to conduct a Newton iteration (with line search)"""
        
        alpha_1 = 1.0
        alpha_max = 1000.0
        
        p_k = self.newtonDir()
        
        f = self.f
        g = self.g
        
        x_k = self.x_list[-1]
        
        alpha_star = lineSearch(f, g, x_k, p_k, alpha_1, alpha_max)
        
        x_kp1 = x_k + alpha_star * p_k
        g_kp1 = g(x_kp1)
        
        self.g_k = copy.deepcopy(self.g_kp1)
        self.g_kp1 = copy.deepcopy(g_kp1)
        
        self.x_list.append(x_kp1)
        
        self.approxInvHessian_update()
        
        print("self.x_list",self.x_list)
        print("self.g_kp1", self.g_kp1)
        
        g_norm = np.sqrt((np.transpose(self.g_kp1).dot(self.g_kp1))[0,0])
        
        self.g_norm_list.append(g_norm)
        
        return g_norm
        
    def optimize(self, epsilon):
        
        """optimize until ||g||_2\leq \epsilon"""
        
        self.gradDescent()
        
        g2_norm = 1e6 # large number
        
        # record iter
        j = 1
        
        while (g2_norm > epsilon):
            
            g2_norm = self.newton()
            
            j += 1
          
        
        if (1 == 1):
            
            n_iter_list = list(xrange(j))
            
            log_g_norm_list = []

            for i in xrange(len(self.g_norm_list)):
                
                log_g_norm_list.append(np.log10(self.g_norm_list[i]))
            
            plt.plot(n_iter_list, log_g_norm_list, '-o')
            
            plt.show()
            
            
            
            
    def postProcess2D(self, x_low, x_up, Nx, y_low, y_up, Ny):
        
        f = self.f
        
        x_vec = np.linspace(x_low,x_up,Nx)
        y_vec = np.linspace(y_low,y_up,Ny)
        E_mat = np.matrix(np.zeros((Nx,Ny)))

        for i in xrange(Nx):

            x_loc = x_vec[i]   
    
            for j in xrange(Ny):

                y_loc = y_vec[j]

                E_mat[j,i] = f(np.matrix([[x_loc],[y_loc]]))

        print(E_mat)
    
        plt.figure()
        im = plt.imshow(E_mat, interpolation='bilinear', origin='lower',
                        cmap=cm.gray, extent=(x_low, x_up, y_low, y_up))
        levels = np.arange(0.0, 5.0, 0.01) # by experiment
        CS = plt.contour(E_mat, levels,
                         origin='lower',
                         linewidths=2,
                         extent=(x_low, x_up, y_low, y_up))
        
        
        x_list = copy.deepcopy(self.x_list)
        
        xx_list = []
        xy_list = []
        for i in xrange(len(x_list)):
            
            xx_list.append(x_list[i][0,0])
            xy_list.append(x_list[i][1,0])
            
        plt.plot(xx_list, xy_list, '-yo')
        
        plt.show()
        
        
        
        
        
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
            
p1 = BFGS(Matyas, Matyas_grad, x0, V0, dim_N)
p1.optimize(10**-12)
p1.postProcess2D(xmin, xmax, Nx, ymin, ymax, Ny)
            
        
        
        
        
        
        
        
        
        
        
        

        

        
        
    
        
        
        
        
        
        
        
        
        
        
           
        
        

