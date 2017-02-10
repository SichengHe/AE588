import numpy as np
import linesearch as LS

class CG(object):

	def __init__(self, f, g, x0, mode):

		self.f = f
		self.g = g

		self.x_list = [x0]

		self.g_norm_list = []

		self.mode = mode

	def gradDescentDir(self):

		""" gradient descent direction (for 1st iter) """
		x0 = self.x_list[0]

		g = self.g

		g_k = g(x0)

		self.g_k = g_k

		# record the gradient norm
		g_norm = np.sqrt((np.transpose(self.g_k).dot(self.g_k))[0,0])

		self.g_norm_list.append(g_norm)

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
		#p_k = -g_k / self.g_norm_list[-1]
		p_k = -g_k
		self.p_k = p_k

		# line search
		alpha_star = LS.lineSearch(f, g, x_k, p_k, alpha_1, alpha_max)

		# get the new point and its corresponding gradient 
		x_kp1 = x_k + alpha_star * p_k
		g_kp1 = g(x_kp1)

		# record new point and its corresponding gradient 
		self.x_list.append(x_kp1)
		self.g_kp1 = g_kp1

		# record the gradient norm
		g_norm = np.sqrt((np.transpose(self.g_kp1).dot(self.g_kp1))[0,0])

		self.g_norm_list.append(g_norm)

	def CG_direction(self):

		g_k = self.g_k
		g_kp1 = self.g_kp1

		p_k = self.p_k

		gkT_gk = (np.transpose(g_k).dot(g_k))[0, 0]
		gkp1T_gkp1 = (np.transpose(g_kp1).dot(g_kp1))[0, 0]

		beta = gkp1T_gkp1 / gkT_gk

		print("beta", beta)

		#p_kp1 = -g_kp1 /  self.g_norm_list[-1] + beta * p_k
		p_kp1 = -g_kp1  + beta * p_k

		print("p_kp1", p_kp1)

		return p_kp1

	def CG(self):

		alpha_1 = 1.0
		alpha_max = 1000.0

		f = self.f
		g = self.g

		p_k = self.CG_direction()
		self.p_k = p_k

		x_k = self.x_list[-1]

		alpha_star = LS.lineSearch(f, g, x_k, p_k, alpha_1, alpha_max)

		x_kp1 = x_k + alpha_star * p_k

		# updates
		g_kp1 = g(x_kp1)

		self.g_k = self.g_kp1
		self.g_kp1 = g_kp1

		print("self.g_k", self.g_k,"self.g_kp1",self.g_kp1)

		self.x_list.append(x_kp1)

		g_norm = np.sqrt((np.transpose(g_kp1).dot(g_kp1))[0,0])

		self.g_norm_list.append(g_norm)

	def optimize(self, epsilon):

		self.gradDescent()

		g_norm = self.g_norm_list[-1]

		j = 2

		while (g_norm > epsilon):

			self.CG()

			g_norm = self.g_norm_list[-1]

			j += 1

		if (self.mode == 1):

			n_iter_list = list(xrange(j))

			log_g_norm_list = []

			for i in xrange(len(self.g_norm_list)):

				log_g_norm_list.append(np.log10(self.g_norm_list[i]))

			return self.x_list, n_iter_list, log_g_norm_list

		if (self.mode == 0):

			f = self.f

			x_star = self.x_list[-1]
			J = f(x_star)

			return x_star, J

			



    		













