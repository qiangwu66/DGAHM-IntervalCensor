import numpy as np
from beta_estimate import beta_est
from gamma_estimate import gamma_est
from C_estimation import C_est
from g_deep import g_D
from I_spline import I_U


def Est_deep(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,Z_test,X_test,X_sort1,X_sort0,t_nodes,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,C0):

    d = Z_train.shape[1]
    for loop in range(20):
        print('deep_iteration time=', loop)
        g_X = g_D(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,X_test,X_sort1,X_sort0,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,C0)
        g_train = g_X['g_train']
        C = C_est(m,U_train,V_train,De1_train,De2_train,De3_train,Z_train,theta_initial,g_train,nodevec)
        beta = beta_est(theta_initial[d:(2*d)], De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        gamma = gamma_est(beta, De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        theta = np.array([beta, gamma]).reshape(2*d,)
        # print('C=', C)
        # print('theta=', theta)
        # print('max_error=', np.max(np.abs(theta - theta_initial)))
        if (np.max(np.abs(C - C0)) <= 0.01 and loop > 3):
            break
        theta_initial = theta
        C0 = C
    
    N_test = X_test.shape[0]
    S_T_W_test = np.zeros((len(t_nodes), N_test))
    scale_test = np.exp(Z_test @ beta + g_X['g_test'][:,0])
    Ezg_test = np.exp(Z_test @ gamma + g_X['g_test'][:,1])

    for k in range(len(t_nodes)):
        z = t_nodes[k] * scale_test  # (N_test,)
        It_node_k = I_U(m, z, nodevec)  # (N_test, P)
        S_T_W_test[k] = np.exp(- np.dot(It_node_k, C) * Ezg_test)
    
    return {
        'S_T_W_test': S_T_W_test,
        'g_train': g_train,
        'g_test1': g_X['g_test1'],
        'g_test0': g_X['g_test0'],
        'C': C,
        'theta': theta
    }

