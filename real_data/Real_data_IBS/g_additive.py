import numpy as np
import scipy.optimize as spo
from B_spline3 import B_S
from I_spline import I_U
def g_A(train_data,X_test,theta,C,m,nodevec,m1,nodevec1):
    Z_train = train_data['Z']
    X_train = train_data['X']
    U_train = train_data['U']
    V_train = train_data['V']
    De1 = train_data['De1']
    De2 = train_data['De2']
    De3 = train_data['De3']
    B_0 = B_S(m1, X_train[:,0], nodevec1)
    B_1 = B_S(m1, X_train[:,1], nodevec1)
    B_2 = B_S(m1, X_train[:,2], nodevec1)
    B_3 = B_S(m1, X_train[:,3], nodevec1)
    B_4 = B_S(m1, X_train[:,4], nodevec1)
    
    def GA(*args):
        b = args[0]
        g1_X = np.dot(B_0, b[0:(m1+4)]) + np.dot(B_1, b[(m1+4):(2*(m1+4))]) + np.dot(B_2, b[(2*(m1+4)):(3*(m1+4))]) + np.dot(B_3, b[(3*(m1+4)):(4*(m1+4))]) + np.dot(B_4, b[(4*(m1+4)):(5*(m1+4))]) + b[5*(m1+4)]*np.ones(X_train.shape[0])
        g2_X = np.dot(B_0, b[(5*(m1+4)+1):(6*(m1+4)+1)]) + np.dot(B_1, b[(6*(m1+4)+1):(7*(m1+4)+1)]) + np.dot(B_2, b[(7*(m1+4)+1):(8*(m1+4)+1)]) + np.dot(B_3, b[(8*(m1+4)+1):(9*(m1+4)+1)]) + np.dot(B_4, b[(9*(m1+4)+1):(10*(m1+4)+1)]) + b[10*(m1+4)+1]*np.ones(X_train.shape[0])
        Iu = I_U(m, U_train * np.exp(Z_train * theta[0] + g1_X), nodevec)
        Iv = I_U(m, V_train * np.exp(Z_train * theta[0] + g1_X), nodevec)
        Ezg = np.exp(Z_train * theta[1] + g2_X)
        loss_fun = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-5) + De2 * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-5) - De3 * np.dot(Iv, C) * Ezg)
        return loss_fun
    b_para = spo.minimize(GA,np.zeros(10*(m1+4)+2),method='SLSQP')['x']
    print('b_para=', b_para)
    
    g1_train = np.dot(B_0, b_para[0:(m1+4)]) + np.dot(B_1, b_para[(m1+4):(2*(m1+4))]) + np.dot(B_2, b_para[(2*(m1+4)):(3*(m1+4))]) + np.dot(B_3, b_para[(3*(m1+4)):(4*(m1+4))]) + np.dot(B_4, b_para[(4*(m1+4)):(5*(m1+4))]) + b_para[5*(m1+4)]*np.ones(X_train.shape[0])
    g2_train = np.dot(B_0, b_para[(5*(m1+4)+1):(6*(m1+4)+1)]) + np.dot(B_1, b_para[(6*(m1+4)+1):(7*(m1+4)+1)]) + np.dot(B_2, b_para[(7*(m1+4)+1):(8*(m1+4)+1)]) + np.dot(B_3, b_para[(8*(m1+4)+1):(9*(m1+4)+1)]) + np.dot(B_4, b_para[(9*(m1+4)+1):(10*(m1+4)+1)]) + b_para[10*(m1+4)+1]*np.ones(X_train.shape[0])
    g_train = np.c_[g1_train, g2_train]
    
    g1_test = np.dot(B_S(m1, X_test[:,0], nodevec1), b_para[0:(m1+4)]) + np.dot(B_S(m1, X_test[:,1], nodevec1), b_para[(m1+4):(2*(m1+4))]) + np.dot(B_S(m1, X_test[:,2], nodevec1), b_para[(2*(m1+4)):(3*(m1+4))]) + np.dot(B_S(m1, X_test[:,3], nodevec1), b_para[(3*(m1+4)):(4*(m1+4))]) + np.dot(B_S(m1, X_test[:,4], nodevec1), b_para[(4*(m1+4)):(5*(m1+4))]) + b_para[5*(m1+4)]*np.ones(X_test.shape[0])
    g2_test = np.dot(B_S(m1, X_test[:,0], nodevec1), b_para[(5*(m1+4)+1):(6*(m1+4)+1)]) + np.dot(B_S(m1, X_test[:,1], nodevec1), b_para[(6*(m1+4)+1):(7*(m1+4)+1)]) + np.dot(B_S(m1, X_test[:,2], nodevec1), b_para[(7*(m1+4)+1):(8*(m1+4)+1)]) + np.dot(B_S(m1, X_test[:,3], nodevec1), b_para[(8*(m1+4)+1):(9*(m1+4)+1)]) + np.dot(B_S(m1, X_test[:,4], nodevec1), b_para[(9*(m1+4)+1):(10*(m1+4)+1)]) + b_para[10*(m1+4)+1]*np.ones(X_test.shape[0])
    g_test = np.c_[g1_test, g2_test]

    return {'b_para':  b_para,
        'g_train': g_train,
        'g_test': g_test
    }