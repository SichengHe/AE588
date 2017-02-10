import numpy as np
import matplotlib.pyplot as plt



def CauchyPt(g_k, B_k, Delta_k):

	""" solve for the Cauchy point based on numerical optimization pp 4.1"""

	gT_g = np.transpose(g_k).dot(g_k)[0, 0] 
	g_norm = np.sqrt(gT_g)

	# searching direction
	pk = (- Delta_k / g_norm) * g_k

	Bg = B_k.dot(g_k)

	gT_Bg = np.transpose(g_k).dot(Bg)[0, 0]

	if (gT_Bg <= 0.0):

		tau = 1.0

	else:

		tau = min( (g_norm**3 / (Delta_k * gT_Bg)), 1.0)

	pkc = tau * pk

	return pkc


def model(f_k, g_k, B_k, s):

	"""model function m(s)=fk + gk^T s + 1/2 s^T Bk s"""

	ms = f_k + np.transpose(g_k).dot(s)[0, 0] + 0.5 * np.transpose(s).dot(B_k).dot(s)[0, 0]

	return ms

def BFGS(s_k, y_k, B_k):

	Bk_sk = B_k.dot(s_k)
	yk_ykT = y_k.dot(np.transpose(y_k))
	ykT_sk = np.transpose(y_k).dot(s_k)[0, 0]

	Bksk_skTBk = Bk_sk.dot(np.transpose(Bk_sk))
	skT_Bksk = np.transpose(s_k).dot(Bk_sk)[0, 0]

	yk_ykT = y_k.dot(np.transpose(y_k))

	B_kp1 = B_k - Bksk_skTBk / skT_Bksk + yk_ykT / ykT_sk

	return B_kp1


def trustRegion(Delta_0, Delta_max, epsilon, x_0, B_0, f, g, mode):

	""" trust region method:
	Inputs: 
	       Delta_0:   initial region size 
	       Delta_max: maximum regoin size
	       epsilon:   gradient convergence criterion
	       H:         Hessian (approx)
	       f :        objective function 
		   mode:      0: x, J as output; 1: path and convergence hist as output
	       """

	Delta_k = Delta_0

	x_k = x_0
	f_k = f(x_0)
	g_k = g(x_0)
	B_k = B_0

	gk_norm = np.sqrt(np.transpose(g_k).dot(g_k)[0, 0])

	i  = 1

	if (mode == 1):

		gk_norm_list = [gk_norm]
		x_list = [x_k]


	while (gk_norm > epsilon):


		if (i > 100):

			break

		print("%d th trust region iter", i)


		flag_update = True

		# obtain the update
		s_k = CauchyPt(g_k, B_k, Delta_k)
		sk_norm = np.sqrt(np.transpose(s_k).dot(s_k)[0,0])

		# x for next iter
		x_kp1 = x_k + s_k
		f_kp1 = f(x_kp1)

		# model
		m0 = f_k
		msk = model(f_k, g_k, B_k, s_k)

		# check whether model ok
		rk = (f_k - f_kp1) / (m0 - msk)



		if (rk < 0.25):

			Delta_k = 0.25 * Delta_k

		elif (rk > 0.75 and sk_norm == Delta_k):

			Delta_k = min(2.0 * Delta_k, Delta_max)



		if (rk < 0.0):

			# reject step no update for xk

			x_kp1 = x_k

			flag_update = False



		if (flag_update):

			g_kp1 = g(x_kp1)

			y_k = g_kp1 - g_k

			B_kp1 = BFGS(s_k, y_k, B_k)


			x_k = x_kp1
			f_k = f_kp1
			g_k = g_kp1
			B_k = B_kp1

			gk_norm = np.sqrt(np.transpose(g_k).dot(g_k)[0, 0])

		if (mode == 1):

			gk_norm_list.append(gk_norm)

			x_list.append(x_k)

			i += 1

			



	if (mode == 1):

		n_iter_list = list(xrange(i))

		log_g_norm_list = []

        for ii in xrange(len(gk_norm_list)):
            
            log_g_norm_list.append(np.log10(gk_norm_list[ii]))
        
        #plt.plot(n_iter_list, log_g_norm_list, '-o')
        
        #plt.show()

        return x_list, gk_norm_list

	if (mode == 0):

		return x_k, f_k












