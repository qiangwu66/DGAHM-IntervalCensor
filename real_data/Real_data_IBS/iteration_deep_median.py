#---------------------------------------
import numpy as np
from C_estimation_median import C_est_median
from g_deep_median import g_D_median

def Est_deep_mid(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,W_test,W_sort1,W_sort0,t_nodes,n_layer_non,n_node_non,n_lr_non,n_epoch_non,nodevec,m,C0):
    for loop in range(20):
        print('deep_median_iteration time=', loop)
        g_X = g_D_median(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,W_test,W_sort1,W_sort0,t_nodes,n_layer_non,n_node_non,n_lr_non,n_epoch_non,C0,m,nodevec)
        g_train = g_X['g_train']

        C = C_est_median(m, U_train, V_train, De1_train, De2_train, De3_train, g_train, nodevec, C0)
        # print('C', C)
    
        if (np.all(np.abs(C - C0) <= 0.01) and loop > 3):
            break
        C0 = C
    
    return {
        'S_T_W_test': g_X['S_T_W_test'],
        'g_train': g_train,
        'g_test1': g_X['g_test1'],
        'g_test0': g_X['g_test0'],
        'C': C
    }

