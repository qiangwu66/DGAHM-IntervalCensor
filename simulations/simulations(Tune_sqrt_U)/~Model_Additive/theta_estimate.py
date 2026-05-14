# 估计theta=(beta,gamma)
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
def theta_est(De1, De2, De3, Z, U, V, C, m, g_X, nodevec):
    def BF(*args):
        b = args[0]
        Iu = I_U(m, U * np.exp(Z * b[0] + g_X[:,0]), nodevec)
        Iv = I_U(m, V * np.exp(Z * b[0] + g_X[:,0]), nodevec)
        Ezg = np.exp(Z * b[1] + g_X[:,1])
        Loss_F = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-5) + De2 * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-5) - De3 * np.dot(Iv, C) * Ezg)
        return Loss_F
    result = spo.minimize(BF,np.zeros(2),method='SLSQP')
    return result['x']
