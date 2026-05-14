#---------------------------------------
import numpy as np
from beta_estimate import beta_est
from gamma_estimate import gamma_est
from C_estimation import C_est
from g_deep import g_D

def Est_deep(train_data,valid_data,X_test,X_subject,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,C):
    Z_train = train_data['Z']
    U_train = train_data['U']
    V_train = train_data['V']
    De1_train = train_data['De1']
    De2_train = train_data['De2']
    De3_train = train_data['De3']
    C_index = 0
    for loop in range(50):
        print('deep_iteration time=', loop)
        g_X = g_D(train_data,valid_data,X_test,X_subject,theta_initial,C,m,nodevec,n_layer,n_node,n_lr,n_epoch)
        g_train = g_X['g_train']
        C = C_est(m, U_train, V_train, De1_train, De2_train, De3_train, Z_train, theta_initial, g_train, nodevec)
        beta = beta_est(theta_initial[1], De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        gamma = gamma_est(beta, De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        theta = np.array([beta, gamma])
        if (np.abs(theta[0] - theta_initial[0]) <= 0.01 and np.abs(theta[1] - theta_initial[1]) <= 0.01 and loop > 3):
            break
        theta_initial = theta
        print('theta', theta)
    
    return {
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'g_subject': g_X['g_subject'],
        'C': C,
        'theta': theta,
        'C_index': C_index
    }

