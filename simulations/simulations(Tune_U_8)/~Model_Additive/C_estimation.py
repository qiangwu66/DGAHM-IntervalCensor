
import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def C_est(m, U, V, De1, De2, De3, Z, theta, g_X, nodevec):
    Iu = I_U(m, U * np.exp(Z*theta[0] + g_X[:,0]), nodevec)
    Iv = I_U(m, V * np.exp(Z*theta[0] + g_X[:,0]), nodevec)
    def LF(*args):
        a = args[0]
        Ezg = np.exp(Z * theta[1] + g_X[:,1])
        Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu,a) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu,a) * Ezg) - np.exp(- np.dot(Iv,a) * Ezg) + 1e-4) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = [(0, 100)] * (m+3)
    result = spo.minimize(LF, np.ones(m+3),method='SLSQP',bounds=bnds)
    return result['x']


