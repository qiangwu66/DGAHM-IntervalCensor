
import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def C_est(m, U, V, De1, De2, De3, g_X, nodevec, C0):
    Iu = I_U(m, np.clip(U * np.exp(g_X[:,0]), 0, 70), nodevec)
    Iv = I_U(m, np.clip(V * np.exp(g_X[:,0]), 0, 70), nodevec)
    def LF(*args):
        a = args[0]
        Ezg = np.exp(g_X[:,1])
        Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.maximum(np.dot(Iu,a) * Ezg, 0)) + 1e-4) + De2 * np.log(np.exp(- np.maximum(np.dot(Iu,a) * Ezg, 0)) - np.exp(- np.maximum(np.dot(Iv,a) * Ezg, 0)) + 1e-4) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = [(0, 100)] * (m+3)
    result = spo.minimize(LF, C0, method='SLSQP', bounds=bnds)
    return result['x']


