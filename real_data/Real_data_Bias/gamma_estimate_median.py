
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
from B_spline2 import B_S2

def gamma_est_median(beta, De1, De2, De3, Z, U, V, C, m, g_X, nodevec):
    def BF(*args):
        b = args[0]
        Iv = I_U(m, np.clip(V * np.exp(np.dot(Z, beta) + g_X[:,0]), 0, 50), nodevec)
        Iu_2 = I_U(m, np.clip(U * np.exp(np.dot(Z, beta) + g_X[:,0]) / 2, 0, 50), nodevec)
        Iuv_2 = I_U(m, np.clip((U + V) * np.exp(np.dot(Z, beta) + g_X[:,0]) / 2, 0, 50), nodevec)
        B_U_2 = B_S2(m, np.clip(U * np.exp(np.dot(Z, beta) + g_X[:,0]) / 2, 0, 50), nodevec)
        B_U_V_2 = B_S2(m, np.clip((U + V) * np.exp(np.dot(Z, beta) + g_X[:,0]) / 2, 0, 50), nodevec)

        Iv = I_U(m, V * np.exp(np.dot(Z, b) + g_X[:,0]), nodevec)
        Ezg = np.exp(np.dot(Z, b) + g_X[:,1])
        Loss_F = - np.mean(De1 * (np.log(np.dot(B_U_2, C) + 1e-4) + np.dot(Z, b) + g_X[:,0] + np.dot(Z, b) + g_X[:,1] - np.dot(Iu_2, C) * Ezg) + De2 * (np.log(np.dot(B_U_V_2, C) + 1e-4) + np.dot(Z, b) + g_X[:,0] + np.dot(Z, b) + g_X[:,1] - np.dot(Iuv_2, C) * Ezg) - De3 * np.dot(Iv, C) * Ezg)
        return Loss_F
    result = spo.minimize(BF, np.zeros(Z.shape[1]), method='SLSQP')
    return result['x']



