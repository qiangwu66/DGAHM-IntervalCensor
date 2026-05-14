import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_2(n, corr, tau):
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(4)
    cov = np.identity(4) * (1-corr) + np.ones((4, 4)) * corr
    X = ndm.multivariate_normal(mean, cov, n)
    X = np.clip(X, 0, 1)
    g_1_X = Z / 5 + np.sin(np.pi * X[:,0] / 2) + np.exp(X[:,1] ** 2) / 2 + np.log(X[:,2] ** 3 + 1) + np.cos(X[:,3]) - 2.2
    g_2_X = Z / 10 + np.exp(np.sqrt(X[:,0])) / 2 + np.cos(np.sin(np.pi * X[:,1])) / 2 + np.log(X[:,2] ** 2 + 1) + np.sqrt(X[:,3]) -1.88
    Y = ndm.rand(n)
    T = (np.exp(- np.log(Y) * np.exp(- g_2_X)) - 1) * np.exp(- g_1_X)
    U = uniform_data(n, 0, tau / 6)
    V_0 = 2 * tau / 5 + U + ndm.exponential(1, n) * tau / 3
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
