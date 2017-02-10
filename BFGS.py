import numpy as np
import copy
#import math as math
#import pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import linesearch as LS

class BFGS(object):
    
    """ main function for hw2 for unconstrained optimization. 
    Implementation of BFGS method.  """

    def __init__(self, f, g, x0, V0, dim_N, mode):
        
        """Initialization: 
            f: obj
            g: d obj/ dx
            x0: initial pt
            V0: initial Hessian (inv)
            dim_N: problem size
            mode: 0: output x, J; 1: output optimization history and convegrence
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

        # mode
        self.mode = mode
        
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
        alpha_star = LS.lineSearch(f, g, x_k, -g_k, alpha_1, alpha_max)
        
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
        
        alpha_star = LS.lineSearch(f, g, x_k, p_k, alpha_1, alpha_max)
        
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

          
        
        if (self.mode == 1):
            
            self.n_iter_list = list(xrange(j))
            
            self.log_g_norm_list = []

            for i in xrange(len(self.g_norm_list)):
                
                self.log_g_norm_list.append(np.log10(self.g_norm_list[i]))

            return self.x_list, self.n_iter_list, self.log_g_norm_list

        if (self.mode == 0):

            f = self.f

            x_star = self.x_list[-1]
            J = f(x_star)

            return x_star, J
            

            
            
            
            
    




