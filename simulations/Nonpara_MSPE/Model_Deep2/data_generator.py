import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_4(n, corr, tau):
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(4)
    cov = np.identity(4) * (1-corr) + np.ones((4, 4)) * corr
    X = ndm.multivariate_normal(mean, cov, n)
    X = np.clip(X, 0, 1)
    g_1_X = (Z * X[:,0] + np.cos(X[:,1]) + np.exp(X[:,2] / 3 + np.cos(np.pi * X[:,3])) / 3) ** 2 / 2 - 1.6
    g_2_X = np.sqrt(Z * (np.sin(X[:,0]) + X[:,1]) + np.exp(np.cos(np.pi * X[:,2]) + np.log(X[:,3] ** 2 + 1)) / 2) - 1.14
    Y = ndm.rand(n)
    T = (np.exp(- np.log(Y) * np.exp(- g_2_X)) - 1) * np.exp(- g_1_X)
    # Lambda(t)= t/8
    U = uniform_data(n, 0, tau / 5)
    V_0 = tau / 5 + U + ndm.exponential(1, n) * tau / 5
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
