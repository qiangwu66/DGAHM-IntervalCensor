

import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def gamma_est(beta, De1, De2, De3, Z, U, V, C, m, g_X, nodevec,
              eps=1e-8, exp_cap=50.0):

    Z = np.asarray(Z)
    g_X = np.asarray(g_X)
    U = np.asarray(U)
    V = np.asarray(V)
    C = np.asarray(C)
    De1 = np.asarray(De1)
    De2 = np.asarray(De2)
    De3 = np.asarray(De3)

    def safe_exp(x):
        return np.exp(np.clip(x, -exp_cap, exp_cap))

    def BF(b):
        Zb = Z @ b
        Zb = np.clip(Zb, -exp_cap, exp_cap)

        Zbeta = Z @ beta
        Zbeta = np.clip(Zbeta, -exp_cap, exp_cap)

        gx0 = np.clip(g_X[:, 0], -exp_cap, exp_cap)
        gx1 = np.clip(g_X[:, 1], -exp_cap, exp_cap)

        scale_uv = safe_exp(Zbeta + gx0)
        U_scaled = U * scale_uv
        V_scaled = V * scale_uv

        Iu = I_U(m, U_scaled, nodevec)   # (n, k)
        Iv = I_U(m, V_scaled, nodevec)   # (n, k)

        Ezg = safe_exp(Zb + gx1)         # (n,)

        IuC = Iu @ C
        IvC = Iv @ C

        IuC = np.clip(IuC, 0.0, None)
        IvC = np.clip(IvC, 0.0, None)

        lam_u = IuC * Ezg
        lam_v = IvC * Ezg

        lam_u = np.clip(lam_u, 0.0, np.float64(exp_cap))
        lam_v = np.clip(lam_v, 0.0, np.float64(exp_cap))

        Su = np.exp(-lam_u)
        Sv = np.exp(-lam_v)

        p1 = 1.0 - Su
        p1 = np.clip(p1, eps, 1.0)  
        term1 = De1 * np.log(p1)

        diff = Su - Sv
        diff = np.clip(diff, eps, 1.0)
        term2 = De2 * np.log(diff)

        term3 = -De3 * lam_v

        Loss_F = -(np.mean(term1 + term2 + term3))

        if not np.isfinite(Loss_F):
            return 1e50
        return Loss_F

    x0 = np.zeros(Z.shape[1], dtype=float)
    result = spo.minimize(
        BF,
        x0,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    )
    return result['x']



