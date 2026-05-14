
import numpy as np
from scipy.optimize import fminbound
from I_spline import I_U

def beta_est(gamma, De1, De2, De3, Z, U, V, C, m, g_X, nodevec):
    def BF(*args):
        b = args[0]
        Iu = I_U(m, U * np.exp(Z * b + g_X[:,0]), nodevec)
        Iv = I_U(m, V * np.exp(Z * b + g_X[:,0]), nodevec)
        Ezg = np.exp(Z * gamma + g_X[:,1])
        Loss_F = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-4) - De3 * np.dot(Iv, C) * Ezg)
        return Loss_F
    result = fminbound(BF, 1.5, 2.5)
    return result

