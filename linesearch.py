import numpy as np

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