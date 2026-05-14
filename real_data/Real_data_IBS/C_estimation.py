

import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def C_est(m, U, V, De1, De2, De3, Z, theta, g_X, nodevec, eps=1e-12):
    c = Z.shape[1]

    scale_u = np.exp(Z @ theta[0:c] + g_X[:, 0])
    scale_v = scale_u  
    Iu = I_U(m, U * scale_u, nodevec)
    Iv = I_U(m, V * scale_v, nodevec)

    Ezg = np.exp(Z @ theta[c:(2*c)] + g_X[:, 1])

    def LF(a):
        xu = np.maximum(Iu @ a, 0.0) * Ezg
        xv = np.maximum(Iv @ a, 0.0) * Ezg

        p1 = -np.expm1(-xu)
        p1 = np.clip(p1, eps, 1.0 - eps)

       
        delta = xv - xu
        p2 = np.exp(-xu) * (-np.expm1(-np.maximum(delta, 0.0))) 
        p2 = np.clip(p2, eps, 1.0 - eps)

        loss = -np.mean(
            De1 * np.log(p1) +
            De2 * np.log(p2) -
            De3 * xv
        )
        return loss

    bnds = [(0.0, 100.0)] * (m + 3)
    res = spo.minimize(LF, 0.1*np.ones(m + 3), method='SLSQP', bounds=bnds)
    return res.x
