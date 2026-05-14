import numpy as np
import scipy.optimize as spo
from I_spline import I_U


def C_est(m, U, V, De1, De2, De3, Z, theta, g_X, nodevec):
    c = Z.shape[1]
    Iu = I_U(m, np.clip(U * np.exp(np.dot(Z, theta[0:c]) + g_X[:,0]), 0, 50), nodevec)
    Iv = I_U(m, np.clip(V * np.exp(np.dot(Z, theta[0:c]) + g_X[:,0]), 0, 50), nodevec)
    def LF(*args):
        a = args[0]
        Ezg = np.exp(np.dot(Z, theta[c:(2*c)]) + g_X[:,1])
        Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu,a) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu,a) * Ezg) - np.exp(- np.dot(Iv,a) * Ezg) + 1e-4) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = [(0, 10)] * (m+3)
    result = spo.minimize(LF, np.ones(m+3),method='SLSQP',bounds=bnds)
    return result['x']


