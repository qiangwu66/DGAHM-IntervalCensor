
import numpy as np
from g_NN_IC import g_IC
from B_estimation import B_est
from Bernstein_Poly import Bern_S

def Est_NN_IC(train_data,validation_data,Z_test,L_1_penalty,N_node,N_lr,N_epoch,m_n,b0,tau):
    U_train = train_data['U']
    V_train = train_data['V']
    De1_train = train_data['De1']
    De2_train = train_data['De2']
    De3_train = train_data['De3']
    # ------------------------------------------------------
    Lambda_U = np.dot(Bern_S(m_n, U_train, 0, tau), b0)
    Lambda_V = np.dot(Bern_S(m_n, V_train, 0, tau), b0)
    C_index = 0
    #----------------
    loss_validation_0 = 0
    loss_validation = 0
    for loop in range(100):
        print('NN_IC_iteration time=', loop)
        # print('loss_validation=', loss_validation)
        if (loss_validation > loss_validation_0) and (loop >= 5):
            C_index = 1
            break
        else:
            loss_validation_0 = loss_validation
        #----------------------
        g_Z = g_IC(train_data, validation_data['X'], Z_test, Lambda_U, Lambda_V, N_node, N_lr, N_epoch, L_1_penalty)
        g_train = g_Z['g_train']
        g_validation = g_Z['g_validation']
        # ----------------------
        b1 = B_est(m_n, U_train, V_train, De1_train, De2_train, De3_train, g_train, tau)
        Lambda_U = np.dot(Bern_S(m_n, U_train, 0, tau), b1)
        Lambda_V = np.dot(Bern_S(m_n, V_train, 0, tau), b1)
        Lambda_U_validation = np.dot(Bern_S(m_n, validation_data['U'], 0, tau), b1)
        Lambda_V_validation = np.dot(Bern_S(m_n, validation_data['V'], 0, tau), b1)
        # print('B1=', b1)
        loss_validation = - np.mean(validation_data['De1'] * np.log(1 - np.exp(- Lambda_U_validation * np.exp(g_validation)) + 1e-5) + validation_data['De2'] * np.log(np.exp(- Lambda_U_validation * np.exp(g_validation)) - np.exp(- Lambda_V_validation * np.exp(g_validation)) + 1e-5) - validation_data['De3'] * Lambda_V_validation * np.exp(g_validation))
        b0 = b1
    
    return {
        'g_train': g_train,
        'g_test': g_Z['g_test'],
        'B1': b1,
        'C_index': C_index,
    }
