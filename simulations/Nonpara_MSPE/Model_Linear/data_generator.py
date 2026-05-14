import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_1(n, corr, tau):
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(4)
    cov = np.identity(4) * (1-corr) + np.ones((4, 4)) * corr
    X = ndm.multivariate_normal(mean, cov, n)
    X = np.clip(X, 0, 1)
    g_1_X = Z + X[:,0] / 2 + X[:,1] / 3 + X[:,2] / 4 + X[:,3] / 5 - 0.9
    g_2_X = (Z + X[:,0] + X[:,1] + X[:,2] + X[:,3]) / 3 - 0.58
    Y = ndm.rand(n)
    T = (np.exp(- np.log(Y) * np.exp(- g_2_X)) - 1) * np.exp(- g_1_X)
    U = uniform_data(n, 0, tau / 5)
    V_0 = tau / 4 + U + ndm.exponential(1, n) * tau / 3
    V = np.clip(V_0, 0, tau)
    De1 = (T <= U)
    De2 = (U < T) * (T <= V)
    De3 = (T > V)
    Z1 = Z.reshape(n, 1)
    X = np.hstack((Z1, X))
    return {
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'V': np.array(V, dtype='float32'),
        'De1': np.array(De1, dtype='float32'),
        'De2': np.array(De2, dtype='float32'),
        'De3': np.array(De3, dtype='float32'),
        'g1_X': np.array(g_1_X, dtype='float32'),
        'g2_X': np.array(g_2_X, dtype='float32')
    }

