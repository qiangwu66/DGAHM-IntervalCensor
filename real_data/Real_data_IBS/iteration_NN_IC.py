
import numpy as np
from g_NN_IC import g_IC
from B_estimation import B_est
from Bernstein_Poly import Bern_S


def Est_NN_IC(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,W_R_test,W_sort1,W_sort0,L_1_penalty, N_node, N_lr, N_epoch, m_n, b0):
    # ------------------------------------------------------
    Lambda_U = np.dot(Bern_S(m_n, U_train, 0, 30), b0)
    Lambda_V = np.dot(Bern_S(m_n, V_train, 0, 30), b0)
    C_index = 0
    #----------------
    loss_validation_0 = 0
    loss_validation = 0
    for loop in range(20):
        print('NN_IC_iteration time=', loop)
        # print('loss_validation=', loss_validation)
        if (loss_validation > loss_validation_0) and (loop >= 5):
            C_index = 1
            break
        else:
            loss_validation_0 = loss_validation
        #----------------------
        g_Z = g_IC(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,W_R_test,W_sort1,W_sort0,Lambda_U,Lambda_V,N_node,N_lr,N_epoch,L_1_penalty)
        g_train = g_Z['g_train']
        g_validation = g_Z['g_validation']
        # ----------------------
        b1 = B_est(m_n, U_train, V_train, De1_train, De2_train, De3_train, g_train)
        Lambda_U = np.dot(Bern_S(m_n, U_train, 0, 30), b1)
        Lambda_V = np.dot(Bern_S(m_n, V_train, 0, 30), b1)
        Lambda_U_validation = np.dot(Bern_S(m_n, U_valid, 0, 30), b1)
        Lambda_V_validation = np.dot(Bern_S(m_n, V_valid, 0, 30), b1)
        # print('B1=', b1)
        loss_validation = - np.mean(De1_valid * np.log(1 - np.exp(- Lambda_U_validation * np.exp(g_validation)) + 1e-5) + De2_valid * np.log(np.exp(- Lambda_U_validation * np.exp(g_validation)) - np.exp(- Lambda_V_validation * np.exp(g_validation)) + 1e-5) - De3_valid * Lambda_V_validation * np.exp(g_validation))
        b0 = b1
    
    return {
        'g_train': g_train,
        'g_test': g_Z['g_test'],
        'g_test1': g_Z['g_test1'],
        'g_test0': g_Z['g_test0'],
        'B1': b1,
        'C_index': C_index,
    }
