# ----------import packages------------
import numpy as np
from beta_estimate import beta_est
from gamma_estimate import gamma_est
from C_estimation import C_est
from g_linear import g_L


def Est_linear(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,theta_initial,nodevec,m,C):
    d = Z_train.shape[1]
    for loop in range(20):
        print('linear_iteration time=', loop)
        g_X = g_L(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,theta_initial,C,m,nodevec)
        g_train = g_X['g_train']
        C = C_est(m,U_train,V_train,De1_train,De2_train,De3_train,Z_train,theta_initial,g_train,nodevec)
        beta = beta_est(theta_initial[d:(2*d)], De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        gamma = gamma_est(beta, De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        theta = np.array([beta, gamma]).reshape(2*d,)
        print('theta=', theta)
        print('max_error=', np.max(np.abs(theta - theta_initial)))
        if (np.max(np.abs(theta - theta_initial)) <= 0.01 and loop > 3):
            C_index = 1
            break
        theta_initial = theta
    
    return {
        'g_train': g_train,
        'C': C,
        'theta': theta
    }
