import numpy as np
import scipy.optimize as spo
from Bernstein_Poly import Bern_S

def B_est(m_n, U, V, De1, De2, De3, g_Z, tau):
    def LF(args):
        a = args
        Ezg = np.exp(g_Z)
        Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Bern_S(m_n, U, 0, tau), a) * Ezg) + 1e-5) + De2 * np.log(np.exp(- np.dot(Bern_S(m_n, U, 0, tau), a) * Ezg) - np.exp(- np.dot(Bern_S(m_n, V, 0, tau), a) * Ezg) + 1e-5) - De3 * np.dot(Bern_S(m_n, V, 0, tau), a) * Ezg)
        return Loss_F1

    def constraint(args):
        a = args
        return np.diff(a) 

    bnds = [(0, 1000)] * (m_n + 1)
    cons = {'type': 'ineq', 'fun': constraint}
    result = spo.minimize(LF, np.ones(m_n + 1), method='SLSQP', bounds=bnds, constraints=cons)
    return result['x']

