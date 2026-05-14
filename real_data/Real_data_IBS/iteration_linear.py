# ----------import packages------------
import numpy as np
# from beta_estimate import beta_est
# from gamma_estimate import gamma_est
from C_estimation_non import C_est_non
from g_linear import g_L
from I_spline import I_U

def Est_linear(W_train,U_train,V_train,De1_train,De2_train,De3_train,t_nodes,W_test,W_sort1,W_sort0,nodevec,m,C0):
    for loop in range(20):
        print('linear_iteration time=', loop)
        g_X = g_L(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_test,W_sort1,W_sort0,C0,m,nodevec)
        g_train = g_X['g_train']
        C = C_est_non(m, U_train, V_train, De1_train, De2_train, De3_train, g_train, nodevec)
        # print('C', C)
    
        if (np.all(np.abs(C - C0) <= 0.01) and loop > 3):
            break
        C0 = C
    
    N_test = W_test.shape[0]
    S_T_W_test = np.zeros((len(t_nodes), N_test))
    scale_test = np.exp(g_X['g_test'][:,0])
    Ezg_test = np.exp(g_X['g_test'][:,1])

    for k in range(len(t_nodes)):
        z = t_nodes[k] * scale_test  # (N_test,)
        It_node_k = I_U(m, z, nodevec)  # (N_test, P)
        S_T_W_test[k] = np.exp(- np.dot(It_node_k, C) * Ezg_test)

    return {
        'S_T_W_test': S_T_W_test,
        'g_train': g_train,
        'g_test1': g_X['g_test1'],
        'g_test0': g_X['g_test0'],
        'C': C
    }

