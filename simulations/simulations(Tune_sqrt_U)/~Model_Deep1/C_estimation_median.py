
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
from B_spline2 import B_S2


def C_est_median(m, U, V, De1, De2, De3, Z, theta, g_X, nodevec):
    # Iu = I_U(m, np.clip(U * np.exp(Z * theta[0] + g_X[:,0]), 0, 100), nodevec)
    Iv = I_U(m, np.clip(V * np.exp(Z * theta[0] + g_X[:,0]), 0, 50), nodevec)
    Iu_2 = I_U(m, np.clip(U * np.exp(Z * theta[0] + g_X[:,0]) / 2, 0, 50), nodevec)
    Iuv_2 = I_U(m, np.clip((U + V) * np.exp(Z * theta[0] + g_X[:,0]) / 2, 0, 50), nodevec)
    B_U_2 = B_S2(m, np.clip(U * np.exp(Z * theta[0] + g_X[:,0]) / 2, 0, 50), nodevec)
    B_U_V_2 = B_S2(m, np.clip((U + V) * np.exp(Z * theta[0] + g_X[:,0]) / 2, 0, 50), nodevec)

    def LF(*args):
        a = args[0]
        Ezg = np.exp(Z * theta[1] + g_X[:,1])
        Loss_F1 = - np.mean(De1 * (np.log(np.dot(B_U_2, a) + 1e-4) + Z * theta[0] + g_X[:,0] + Z * theta[1] + g_X[:,1] - np.dot(Iu_2,a) * Ezg) + De2 * (np.log(np.dot(B_U_V_2, a) + 1e-4) + Z * theta[0] + g_X[:,0] + Z * theta[1] + g_X[:,1] - np.dot(Iuv_2,a) * Ezg) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = [(0, 100)] * (m+3)
    result = spo.minimize(LF, np.ones(m+3), method='SLSQP', bounds=bnds)
    return result['x']


