# %% -------------import packages--------------
import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def g_L(train_data,X_test,X_subject,theta,C,m,nodevec):
    Z_train = train_data['Z']
    X_train = train_data['X']
    U_train = train_data['U']
    V_train = train_data['V']
    De1 = train_data['De1']
    De2 = train_data['De2']
    De3 = train_data['De3']
    d = X_train.shape[1]
    def GL(*args):
        b = args[0]
        g1_X = np.dot(X_train,b[0:d]) + b[d]*np.ones(X_train.shape[0])
        g2_X = np.dot(X_train,b[(d+1):(2*d+1)]) + b[2*d+1]*np.ones(X_train.shape[0])
        Iu = I_U(m, np.clip(U_train * np.exp(Z_train * theta[0] + g1_X), 0, 90), nodevec)
        Iv = I_U(m, np.clip(V_train * np.exp(Z_train * theta[0] + g1_X), 0, 90), nodevec)
        Ezg = np.exp(Z_train * theta[1] + g2_X)
        loss_fun = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-4) - De3 * np.dot(Iv, C) * Ezg)
        return loss_fun
    g_linear_para = spo.minimize(GL,np.zeros(2*d+2),method='SLSQP')['x']
    
    g1_train = np.dot(X_train, g_linear_para[0:d]) + g_linear_para[d]*np.ones(X_train.shape[0])
    g2_train = np.dot(X_train, g_linear_para[(d+1):(2*d+1)]) + g_linear_para[2*d+1]*np.ones(X_train.shape[0])
    g_train = np.c_[g1_train, g2_train]
    
    g1_test = np.dot(X_test, g_linear_para[0:d]) + g_linear_para[d]*np.ones(X_test.shape[0])
    g2_test = np.dot(X_test, g_linear_para[(d+1):(2*d+1)]) + g_linear_para[2*d+1]*np.ones(X_test.shape[0])
    g_test = np.c_[g1_test, g2_test]

    g1_subject = np.dot(X_subject, g_linear_para[0:d]) + g_linear_para[d]*np.ones(X_subject.shape[0])
    g2_subject = np.dot(X_subject, g_linear_para[(d+1):(2*d+1)]) + g_linear_para[2*d+1]*np.ones(X_subject.shape[0])
    g_subject = np.c_[g1_subject, g2_subject]

    return {'linear_g': g_linear_para,
        'g_train': g_train,
        'g_test': g_test,
        'g_subject': g_subject
    }
