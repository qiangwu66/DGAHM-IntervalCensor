
import numpy as np
import numpy.random as ndm



def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_4(n, corr, theta, tau):
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(4)
    cov = np.identity(4) * (1-corr) + np.ones((4, 4)) * corr
    X = ndm.multivariate_normal(mean, cov, n)
    X = np.clip(X, 0, 1)
    g_1_X = (np.sqrt(X[:,0] * X[:,1]) / 2 + np.log(X[:,1] * X[:,2]+ 1) / 3 + np.exp(X[:,3]) / 4) ** 2 / 2 - 0.19
    g_2_X = (X[:,0] * X[:,1] / 2 +  X[:,2] ** 2 * X[:,3] / 3 + np.log(X[:,3] + 1)) ** 2 / 2 - 0.16
    Y = ndm.rand(n)
    T = - 8 * np.log(Y) * np.exp(- Z * theta[0] - g_1_X - Z * theta[1] - g_2_X)
    U = uniform_data(n, 0, 2 *tau / 11)
    V_0 = tau / 3 + U + ndm.exponential(1, n) * tau / 4
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

