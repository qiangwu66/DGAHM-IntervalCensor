import numpy as np
import scipy.optimize as spo
from I_spline import I_U


def beta_est(gamma, De1, De2, De3, Z, U, V, C, m, g_X, nodevec):
    def BF(*args):
        b = args[0]
        Iu = I_U(m, np.clip(U * np.exp(np.dot(Z, b) + g_X[:,0]), 0, 50), nodevec)
        Iv = I_U(m, np.clip(V * np.exp(np.dot(Z, b) + g_X[:,0]), 0, 50), nodevec)
        Ezg = np.exp(np.dot(Z, gamma) + g_X[:,1])
        Loss_F = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-4) - De3 * np.dot(Iv, C) * Ezg)
        return Loss_F
    result = spo.minimize(BF,np.zeros(Z.shape[1]),method='SLSQP')
    return result['x']

