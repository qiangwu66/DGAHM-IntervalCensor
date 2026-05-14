
import numpy as np
import numpy.random as ndm


def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_2(n, corr, theta, tau):
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(4)
    cov = np.identity(4) * (1-corr) + np.ones((4, 4)) * corr
    X = ndm.multivariate_normal(mean, cov, n)
    X = np.clip(X, 0, 1)
    g_1_X = np.log(X[:,0] + 1) / 3 + np.exp(X[:,1]) / 5 + X[:,2] ** 2 / 4 + X[:,3] / 2 - 0.6
    g_2_X = X[:,0] + np.sqrt(X[:,1]) / 2 + np.log(X[:,2] + 1) / 3 + X[:,3] ** 2 / 4 - 0.65
    Y = ndm.rand(n)
    T = np.exp(- Z * theta[0] - g_1_X ) * (- 6 * np.log(Y) * np.exp(- Z * theta[1] - g_2_X)) ** (4/3)
    U = uniform_data(n, 0, tau / 7)
    V_0 = tau / 3 + U + ndm.exponential(1, n) * tau / 2
    V = np.clip(V_0, 0, tau)
    De1 = (T <= U)
    De2 = (U < T) * (T <= V)
    De3 = (T > V)
    return {
        'Z': np.array(Z, dtype='float32'),
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

