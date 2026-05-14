import math
import numpy as np

def bernstein_poly(m_n, i, t, l, u):
    return math.comb(m_n, i) * (((t-l)/(u-l))**i) * ((1 - (t-l)/(u-l))**(m_n - i))


def Bern_poly_basis(m_n, t, l, u):
    Bern_p = []
    for i in range(m_n + 1):
        Bern_p.append(bernstein_poly(m_n, i, t, l, u))
    return Bern_p
    
    
def Bern_S(m_n, tt, l, u):
    B_t = np.zeros(shape=(len(tt), m_n + 1))
    for b in range(len(tt)):
        t = tt[b]
        B_t[b] = Bern_poly_basis(m_n, t, l, u)
    B_t = np.array(B_t, dtype='float32')
    return B_t
