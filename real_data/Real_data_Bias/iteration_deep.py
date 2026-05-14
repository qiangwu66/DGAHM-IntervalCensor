import numpy as np
from beta_estimate import beta_est
from gamma_estimate import gamma_est
from C_estimation import C_est
from g_deep import g_D


def Est_deep(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,C):
    d = Z_train.shape[1]
    for loop in range(20):
        print('deep_iteration time=', loop)
        g_X = g_D(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,theta_initial,C,m,nodevec,n_layer,n_node,n_lr,n_epoch)
        g_train = g_X['g_train']
        C = C_est(m,U_train,V_train,De1_train,De2_train,De3_train,Z_train,theta_initial,g_train,nodevec)
        beta = beta_est(theta_initial[d:(2*d)], De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        gamma = gamma_est(beta, De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        theta = np.array([beta, gamma]).reshape(2*d,)
        print('theta=', theta)
        print('max_error=', np.max(np.abs(theta - theta_initial)))
        if (np.max(np.abs(theta - theta_initial)) <= 0.01 and loop > 3):
            break
        theta_initial = theta
    
    return {
        'g_train': g_train,
        'C': C,
        'theta': theta
    }

