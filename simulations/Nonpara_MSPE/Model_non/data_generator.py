import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_5(n, corr, tau):
    mean = np.zeros(20)
    cov = np.identity(20) * (1-corr) + np.ones((20, 20)) * corr
    X = ndm.multivariate_normal(mean, cov, n)
    X = np.clip(X, 0, 1)
    g_1_X = 0
    g_2_X = (np.sqrt(X[:,0] ** 2 + X[:,1] ** 2) + np.sin(X[:,2] + X[:,3])) / 5
    Y = ndm.rand(n)
    T = (np.exp(- np.log(Y) * np.exp(- g_2_X)) - 1) * np.exp(- g_1_X)
    U = uniform_data(n, 0, tau / 6)
    V_0 = tau / 8 + U + ndm.exponential(1, n) * tau / 8
    V = np.clip(V_0, 0, tau)
    De1 = (T <= U)
    De2 = (U < T) * (T <= V)
    De3 = (T > V)
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

data = generate_case_5(10000, 0.5, 5)
np.mean(data['De1']), np.mean(data['De2']), np.mean(data['De3']), np.mean(data['g2_X']), 5 * np.min(np.exp(- data['g2_X']))
