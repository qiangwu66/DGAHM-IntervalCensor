import numpy as np
from beta_estimate_median import beta_est_median
from gamma_estimate_median import gamma_est_median
from C_estimation_median import C_est_median
from g_deep_median import g_D_median


def Est_deep_median(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,C):
    d = Z_train.shape[1]
    for loop in range(20):
        print('deep_median_iteration time=', loop)
        g_X = g_D_median(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,theta_initial,C,m,nodevec,n_layer,n_node,n_lr,n_epoch)
        g_train = g_X['g_train']
        C = C_est_median(m, U_train, V_train, De1_train, De2_train, De3_train, Z_train, theta_initial, g_train, nodevec)
        beta = beta_est_median(theta_initial[d:(2*d)], De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        gamma = gamma_est_median(beta, De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        theta = np.array([beta, gamma]).reshape(2*d,)
        if (np.abs(theta[0] - theta_initial[0]) <= 0.01 and np.abs(theta[1] - theta_initial[1]) <= 0.01 and loop > 3):
            break
        theta_initial = theta
    
    return {
        'g_train': g_train,
        'C': C,
        'theta': theta
    }
