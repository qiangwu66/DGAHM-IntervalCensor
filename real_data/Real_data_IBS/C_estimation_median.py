
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
from B_spline2 import B_S2


def C_est_median(m, U, V, De1, De2, De3, g_X, nodevec, C0):
    Iv = I_U(m, np.clip(V * np.exp(g_X[:,0]), 0, 20), nodevec)
    Iu_2 = I_U(m, np.clip(U * np.exp(g_X[:,0]) / 2, 0, 20), nodevec)
    Iuv_2 = I_U(m, np.clip((U + V) * np.exp(g_X[:,0]) / 2, 0, 20), nodevec)
    B_U_2 = B_S2(m, np.clip(U * np.exp(g_X[:,0]) / 2, 0, 20), nodevec)
    B_U_V_2 = B_S2(m, np.clip((U + V) * np.exp(g_X[:,0]) / 2, 0, 20), nodevec)

    def LF(*args):
        a = args[0]
        Ezg = np.exp(g_X[:,1])
        Loss_F1 = - np.mean(De1 * (np.log(np.dot(B_U_2, a) + 1e-4) + g_X[:,0] + g_X[:,1] - np.dot(Iu_2,a) * Ezg) + De2 * (np.log(np.dot(B_U_V_2, a) + 1e-4) + g_X[:,0] + g_X[:,1] - np.dot(Iuv_2,a) * Ezg) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = [(0, 100)] * (m+3)
    result = spo.minimize(LF, C0, method='SLSQP', bounds=bnds)
    return result['x']
