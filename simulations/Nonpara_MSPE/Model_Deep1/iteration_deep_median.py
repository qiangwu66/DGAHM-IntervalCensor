#---------------------------------------
import numpy as np
from C_estimation_median import C_est_median
from g_deep_median import g_D_median

def Est_deep_median(train_data,valid_data,X_test,t_nodes,n_layer,n_node,n_lr,n_epoch,nodevec,m,C0):
    U_train = train_data['U']
    V_train = train_data['V']
    De1_train = train_data['De1']
    De2_train = train_data['De2']
    De3_train = train_data['De3']
    for loop in range(50):
        g_X = g_D_median(train_data,valid_data,X_test,t_nodes,C0,m,nodevec,n_layer,n_node,n_lr,n_epoch)
        g_train = g_X['g_train']

        C = C_est_median(m, U_train, V_train, De1_train, De2_train, De3_train, g_train, nodevec, C0)
    
        if (np.all(np.abs(C - C0) <= 0.02) and loop > 2):
            break
        C0 = C
    
    return {
        'S_T_X_test': g_X['S_T_X_test'],
        'C': C
    }

