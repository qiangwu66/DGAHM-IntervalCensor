


import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def g_L(W_train, U_train, V_train, De1_train, De2_train, De3_train, W_test, W_sort1, W_sort0, C, m, nodevec):
    d = W_train.shape[1]
    eps = 1e-12
    exp_cap = 50.0 

    def GL(*args):
        b = args[0]

        g1_X = W_train @ b[0:d] + b[d]
        g2_X = W_train @ b[(d+1):(2*d+1)] + b[2*d+1]

        eg1 = np.exp(np.clip(g1_X, -exp_cap, exp_cap))
        Ezg = np.exp(np.clip(g2_X, -exp_cap, exp_cap))

        Iu = I_U(m, U_train * eg1, nodevec)  # shape: (n, K)
        Iv = I_U(m, V_train * eg1, nodevec)  # shape: (n, K)
        IuC = Iu @ C                           # shape: (n,)
        IvC = Iv @ C

        a = np.clip(IuC * Ezg, 0.0, np.inf)
        b_ = np.clip(IvC * Ezg, 0.0, np.inf)

        p1 = -np.expm1(-a)               
        ea = np.exp(-np.clip(a, 0.0, exp_cap))
        eb = np.exp(-np.clip(b_, 0.0, exp_cap))
        p2 = ea - eb

        p1 = np.clip(p1, eps, 1.0) 
        p2 = np.clip(p2, eps, np.inf)

        term1 = De1_train * np.log(p1)
        term2 = De2_train * np.log(p2)
        term3 = -De3_train * b_

        loss_fun = -np.mean(term1 + term2 + term3)
        return loss_fun

    res = spo.minimize(GL, np.zeros(2*d + 2), method='SLSQP')
    g_linear_para = res['x']

    g1_train = W_train @ g_linear_para[0:d] + g_linear_para[d]
    g2_train = W_train @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_train = np.c_[g1_train, g2_train]

    g1_test1 = W_sort1 @ g_linear_para[0:d] + g_linear_para[d]
    g2_test1 = W_sort1 @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_test1 = np.c_[g1_test1, g2_test1]

    g1_test0 = W_sort0 @ g_linear_para[0:d] + g_linear_para[d]
    g2_test0 = W_sort0 @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_test0 = np.c_[g1_test0, g2_test0]

    g1_test = W_test @ g_linear_para[0:d] + g_linear_para[d]
    g2_test = W_test @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_test = np.c_[g1_test, g2_test]

    return {
        'linear_g': g_linear_para,
        'g_train': g_train,
        'g_test1': g_test1,
        'g_test0': g_test0,
        'g_test': g_test
    }

