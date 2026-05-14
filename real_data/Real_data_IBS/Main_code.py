#%% ----------------------
import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

from iteration_deep import Est_deep
from iteration_linear import Est_linear
from iteration_deep_median import Est_deep_mid
from iteration_deep_non import Est_deep_non
from Indicate_T_t import indicate_T_t_matrix
from iteration_NN_IC import Est_NN_IC
from Bernstein_Poly import *


from I_spline import I_U
from I_spline import I_S
from B_spline2 import B_S2
import os

#%% ---------- ------------- 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed) 
#%% -----------------------
set_seed(20)

results_folder = "results_folder"
os.makedirs(results_folder, exist_ok=True)

def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] < b).astype(int)
    return I_M


#%% -----------------------
p = 3 
n_layer = 3
n_node = 50
n_epoch = 1000
n_lr = 3.5e-4 

n_layer_non = 3 
n_node_non = 100
n_epoch_non = 1000
n_lr_non = 5e-4

n_layer_mid = 3 
n_node_mid = 50
n_epoch_mid = 1000
n_lr_mid = 5e-4

L_1_penalty = 0.5  
N_node = 50
N_epoch = 1000
N_lr = 0.01  
m_n = 5 

def generate_increasing_random_numbers(n):
    numbers = []
    prev_number = 0.0
    for _ in range(n):
        number = np.random.uniform(prev_number, 1)
        numbers.append(number)
        prev_number = number
    return numbers

b0 = np.array(generate_increasing_random_numbers(m_n+1), dtype="float32")



m = 12
m_L = 6
#%% Data Processing
df = pd.read_csv('data_center.csv') 

Z = np.array(df[["BMI01","GLUCOS01","HDL01","TCHSIU01"]], dtype='float32')
X = np.array(df[["SBPA21","SBPA22","RACEGRP","GENDER","V1AGE01","Cen1","Cen2","Cen3"]], dtype='float32')

U = np.array(df["U_year"], dtype='float32')
V = np.array(df["V_year"], dtype='float32')

De1 = np.array(df["De1"], dtype='float32')
De2 = np.array(df["De2"], dtype='float32')
De3 = np.array(df["De3"], dtype='float32')

W = np.hstack((Z, X))

nodevec = np.array(np.linspace(0, 200, m+2), dtype="float32")
c_initial = np.array(0.05*np.ones(m+p), dtype="float32") 
nodevec_L = np.array(np.linspace(0, 200, m_L+2), dtype="float32")
c_initial_L = np.array(0.05*np.ones(m_L+p), dtype="float32") 
theta_initial = np.array(np.zeros(2*Z.shape[1]), dtype='float32')

#%% ==============================================
A = np.arange(len(U))
np.random.shuffle(A)

Z_R = Z[A]
X_R = X[A]
U_R = U[A]
V_R = V[A]
De1_R = De1[A]
De2_R = De2[A]
De3_R = De3[A]
W_R = W[A]

# -------------------------------------------------
# ---train data 11000
Z_R_train = Z_R[np.arange(11000)]
X_train_all = X_R[np.arange(11000)]
X_R_train = X_train_all[:,0:8]
U_R_train = U_R[np.arange(11000)]
V_R_train = V_R[np.arange(11000)]
De1_R_train = De1_R[np.arange(11000)]
De2_R_train = De2_R[np.arange(11000)]
De3_R_train = De3_R[np.arange(11000)]
W_R_train = W_R[np.arange(11000)]
# ---test data 2201
Z_R_test = Z_R[np.arange(11000,len(U))]
# Z_R_test = np.delete(Z_R, np.arange(11000), axis=0)
X_R_test = X_R[np.arange(11000,len(U))][:,0:8]
U_R_test = U_R[np.arange(11000,len(U))]
V_R_test = V_R[np.arange(11000,len(U))]
De1_R_test = De1_R[np.arange(11000,len(U))]
De2_R_test = De2_R[np.arange(11000,len(U))]
De3_R_test = De3_R[np.arange(11000,len(U))]
W_R_test = W_R[np.arange(11000,len(U))]

# np.mean(De1_R_test), np.mean(De2_R_test), np.mean(De3_R_test)
# np.mean(De1_R_train), np.mean(De2_R_train), np.mean(De3_R_train)
# np.mean(De1_R), np.mean(De2_R), np.mean(De3_R)


#%% Divide the samples of the test set into two classes（delta3=1, delta3=0）
# np.mean(De3_R_test) # right censoring rate
# --------- delta3 = 1----------
U_test1 = np.array(U_R_test[De3_R_test==1])
V_test1 = np.array(V_R_test[De3_R_test==1])
Z_test1 = np.array(Z_R_test[De3_R_test==1])
X_test1 = np.array(X_R_test[De3_R_test==1])
W_test1 = np.array(W_R_test[De3_R_test==1])
# Sort De_test1 to select 25%, 50%, 75% quantile points
X_sort1 = X_test1[V_test1.argsort()]
Z_sort1 = Z_test1[V_test1.argsort()]
U_sort1 = U_test1[V_test1.argsort()]
V_sort1 = V_test1[V_test1.argsort()]
W_sort1 = W_test1[V_test1.argsort()]

n_V1 = len(V_sort1)
V1_025 = V_sort1[round(n_V1*0.25)]
V1_050 = V_sort1[round(n_V1*0.5)]
V1_075 = V_sort1[round(n_V1*0.75)]
V1 = [V1_025, V1_050, V1_075]

# --------- delta3 = 0----------
U_test0 = np.array(U_R_test[De3_R_test==0])
V_test0 = np.array(V_R_test[De3_R_test==0])
Z_test0 = np.array(Z_R_test[De3_R_test==0])
X_test0 = np.array(X_R_test[De3_R_test==0])
W_test0 = np.array(W_R_test[De3_R_test==0])
# Sort De_test0 to select 25%, 50%, 75% quantile points
X_sort0 = X_test0[V_test0.argsort()]
Z_sort0 = Z_test0[V_test0.argsort()]
U_sort0 = U_test0[V_test0.argsort()]
V_sort0 = V_test0[V_test0.argsort()]
W_sort0 = W_test0[V_test0.argsort()]

n_V0 = len(V_sort0)
V0_025 = V_sort0[round(n_V0*0.25)]
V0_050 = V_sort0[round(n_V0*0.5)]
V0_075 = V_sort0[round(n_V0*0.75)]
V0 = [V0_025, V0_050, V0_075]


# Draw horizontal coordinate of the graph with delta=1
V_value = np.array(np.linspace(0, 30, 20), dtype="float32")
t_nodes = np.array(np.linspace(0, np.max(V_R_test), 201), dtype="float32")

 
#-----------存储数据画Survival---------------
beta_g_D1 = np.zeros((5,n_V1))
gamma_g_D1 = np.zeros((5,n_V1))
beta_g_L1 = np.zeros((5,n_V1))
gamma_g_L1 = np.zeros((5,n_V1))
non_g1_D1 = np.zeros((5,n_V1))
non_g2_D1 = np.zeros((5,n_V1))
mid_g1_D1 = np.zeros((5,n_V1))
mid_g2_D1 = np.zeros((5,n_V1))
NN_IC_g1 = np.zeros((5,n_V1))



beta_g_D0 = np.zeros((5,n_V0))
gamma_g_D0 = np.zeros((5,n_V0))
beta_g_L0 = np.zeros((5,n_V0))
gamma_g_L0 = np.zeros((5,n_V0))
non_g1_D0 = np.zeros((5,n_V0))
non_g2_D0 = np.zeros((5,n_V0))
mid_g1_D0 = np.zeros((5,n_V0))
mid_g2_D0 = np.zeros((5,n_V0))
NN_IC_g0 = np.zeros((5,n_V0))


C_D = np.zeros((5, m+p))
C_L = np.zeros((5, m_L+p))
C_D_non = np.zeros((5, m+p))
C_D_mid = np.zeros((5, m+p))
NN_IC_B = np.zeros((5, m_n+1))


# ----------IBS-----------------
I_T_t = np.zeros((5, len(t_nodes), len(V_R_test))) 
S_t_W_test_DGAHM = np.zeros((5, len(t_nodes), len(V_R_test))) 
S_t_W_test_DGAHM_non = np.zeros((5, len(t_nodes), len(V_R_test))) 
S_t_W_test_DGAHM_mid = np.zeros((5, len(t_nodes), len(V_R_test))) 
S_t_W_test_EHM = np.zeros((5, len(t_nodes), len(V_R_test))) 
S_t_W_test_NN_IC = np.zeros((5, len(t_nodes), len(V_R_test))) 

# ------------------------------------------
c_n = 2200
for i in range(5):
    print('i =', i)
    Z_train = np.delete(Z_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    X_train = np.delete(X_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    U_train = np.delete(U_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    V_train = np.delete(V_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De1_train = np.delete(De1_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De2_train = np.delete(De2_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De3_train = np.delete(De3_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    W_train = np.delete(W_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)

    Z_valid = Z_R_train[np.arange(i*c_n, (i+1)*c_n)]
    X_valid = X_R_train[np.arange(i*c_n, (i+1)*c_n)]
    U_valid = U_R_train[np.arange(i*c_n, (i+1)*c_n)]
    V_valid = V_R_train[np.arange(i*c_n, (i+1)*c_n)]
    De1_valid = De1_R_train[np.arange(i*c_n, (i+1)*c_n)]
    De2_valid = De2_R_train[np.arange(i*c_n, (i+1)*c_n)]
    De3_valid = De3_R_train[np.arange(i*c_n, (i+1)*c_n)]
    W_valid = W_R_train[np.arange(i*c_n, (i+1)*c_n)]

    n = len(U_train)
    z_d = Z_train.shape[1]

    #%% DGAHM-Non
    Est_D_non = Est_deep_non(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,W_R_test,U_R_test,V_R_test,W_sort1,W_sort0,t_nodes,n_layer_non,n_node_non,n_lr_non,n_epoch_non,nodevec,m,c_initial)
    non_g1_D1[i] = Est_D_non['g_test1'][:,0]
    non_g2_D1[i] = Est_D_non['g_test1'][:,1]
    non_g1_D0[i] = Est_D_non['g_test0'][:,0]
    non_g2_D0[i] = Est_D_non['g_test0'][:,1]
    C_D_non[i] = Est_D_non['C']

    #%% DGAHM-Mid
    Est_D_mid = Est_deep_mid(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,W_R_test,W_sort1,W_sort0,t_nodes,n_layer_mid,n_node_mid,n_lr_mid,n_epoch_mid,nodevec,m,c_initial)
    mid_g1_D1[i] = Est_D_mid['g_test1'][:,0]
    mid_g2_D1[i] = Est_D_mid['g_test1'][:,1]
    mid_g1_D0[i] = Est_D_mid['g_test0'][:,0]
    mid_g2_D0[i] = Est_D_mid['g_test0'][:,1]
    C_D_mid[i] = Est_D_mid['C']

    #%% DGAHM
    Est_D = Est_deep(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,Z_R_test,X_R_test,X_sort1,X_sort0,t_nodes,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
    # theta_D
    theta_D = Est_D['theta']
    
    Ezg1_D = np.exp(np.dot(Z_train, theta_D[0:z_d]) + Est_D['g_train'][:,0])
    Ezg2_D = np.exp(np.dot(Z_train, theta_D[z_d:(2*z_d)]) + Est_D['g_train'][:,1])
    # compute \Lambda()
    Iu_D = I_U(m, U_train * Ezg1_D, nodevec)
    Iv_D = I_U(m, V_train * Ezg1_D, nodevec)
    Lamb_U_D = np.dot(Iu_D, Est_D['C'])
    Lamb_V_D = np.dot(Iv_D, Est_D['C'])
    f_U_D = Lamb_U_D * Ezg2_D
    f_V_D = Lamb_V_D * Ezg2_D
    # compute \Lambda'()
    Bu_D = B_S2(m, U_train * Ezg1_D, nodevec)
    Bv_D = B_S2(m, V_train * Ezg1_D, nodevec)
    dLamb_U_D = np.matmul(Bu_D, Est_D['C'])
    dLamb_V_D = np.matmul(Bv_D, Est_D['C'])
    
    
    beta_g_D1[i] = np.dot(Z_sort1, theta_D[0:z_d]) + Est_D['g_test1'][:,0]
    gamma_g_D1[i] = np.dot(Z_sort1, theta_D[z_d:(2*z_d)]) + Est_D['g_test1'][:,1]
    beta_g_D0[i] = np.dot(Z_sort0, theta_D[0:z_d]) + Est_D['g_test0'][:,0]
    gamma_g_D0[i] = np.dot(Z_sort0, theta_D[z_d:(2*z_d)]) + Est_D['g_test0'][:,1]
    C_D[i] = Est_D['C']

    #%% EHM
    Est_L = Est_linear(W_train,U_train,V_train,De1_train,De2_train,De3_train,t_nodes,W_R_test,W_sort1,W_sort0,nodevec_L,m_L,c_initial_L)
    
    # theta_L = Est_L['theta']
    
    Ezg1_L = np.exp(Est_L['g_train'][:,0])
    Ezg2_L = np.exp(Est_L['g_train'][:,1])
    # compute \Lambda()
    Iu_L = I_U(m_L, U_train * Ezg1_L, nodevec_L)
    Iv_L = I_U(m_L, V_train * Ezg1_L, nodevec_L)
    Lamb_U_L = np.dot(Iu_L, Est_L['C'])
    Lamb_V_L = np.dot(Iv_L, Est_L['C'])
    f_U_L = Lamb_U_L * Ezg2_L
    f_V_L = Lamb_V_L * Ezg2_L
    # compute \Lambda'()
    Bu_L = B_S2(m_L, U_train * Ezg1_L, nodevec_L)
    Bv_L = B_S2(m_L, V_train * Ezg1_L, nodevec_L)
    dLamb_U_L = np.matmul(Bu_L, Est_L['C'])
    dLamb_V_L = np.matmul(Bv_L, Est_L['C'])
    
    beta_g_L1[i] = Est_L['g_test1'][:,0]
    gamma_g_L1[i] = Est_L['g_test1'][:,1]
    beta_g_L0[i] = Est_L['g_test0'][:,0]
    gamma_g_L0[i] = Est_L['g_test0'][:,1]
    C_L[i] = Est_L['C']


    #%% NN-IC
    est_NN_IC = Est_NN_IC(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,W_R_test,W_sort1,W_sort0,L_1_penalty,N_node,N_lr,N_epoch,m_n,b0)
    Lambda_NN_IC = np.dot(Bern_S(m_n, t_nodes, 0, 30), np.array(est_NN_IC['B1']))
    S_NN_IC = np.exp(-np.dot(np.exp(np.reshape(est_NN_IC['g_test'], (len(est_NN_IC['g_test']), 1))), np.reshape(Lambda_NN_IC, (1, len(Lambda_NN_IC)))))
    NN_IC_g1[i] = est_NN_IC['g_test1']
    NN_IC_g0[i] = est_NN_IC['g_test0']
    NN_IC_B[i] = est_NN_IC['B1']





    #---------------IBS save------------------
    I_T_t[i,:,:] = indicate_T_t_matrix(t_nodes, Est_D_non['S_T_W_test'], Est_D_non['S_U_test'], Est_D_non['S_V_test'],
                        U_R_test, V_R_test, De1_R_test, De2_R_test, De3_R_test)
    S_t_W_test_DGAHM_non[i,:,:] = Est_D_non['S_T_W_test']
    S_t_W_test_DGAHM_mid[i,:,:] = Est_D_mid['S_T_W_test']
    S_t_W_test_DGAHM[i,:,:] = Est_D['S_T_W_test']
    S_t_W_test_EHM[i,:,:] = Est_L['S_T_W_test']
    S_t_W_test_NN_IC[i,:,:] = S_NN_IC.T
    

#%% ------------------------save 5-fold results--------------------------
to_save = {
    'beta_g_D1': beta_g_D1,
    'gamma_g_D1': gamma_g_D1,
    'beta_g_L1': beta_g_L1,
    'gamma_g_L1': gamma_g_L1,
    'non_g1_D1': non_g1_D1,
    'non_g2_D1': non_g2_D1,
    'mid_g1_D1': mid_g1_D1,
    'mid_g2_D1': mid_g2_D1,
    'NN_IC_g1': NN_IC_g1,
    

    'beta_g_D0': beta_g_D0,
    'gamma_g_D0': gamma_g_D0,
    'beta_g_L0': beta_g_L0,
    'gamma_g_L0': gamma_g_L0,
    'non_g1_D0': non_g1_D0,
    'non_g2_D0': non_g2_D0,
    'mid_g1_D0': mid_g1_D0,
    'mid_g2_D0': mid_g2_D0,
    'NN_IC_g0': NN_IC_g0,

    'C_D': C_D,
    'C_L': C_L,
    'C_D_non': C_D_non,
    'C_D_mid': C_D_mid,
    'NN_IC_B': NN_IC_B,


    'I_T_t': I_T_t,
    'S_t_W_test_DGAHM_non': S_t_W_test_DGAHM_non,
    'S_t_W_test_DGAHM_mid': S_t_W_test_DGAHM_mid,
    'S_t_W_test_DGAHM': S_t_W_test_DGAHM,
    'S_t_W_test_EHM': S_t_W_test_EHM,
    'S_t_W_test_NN_IC': S_t_W_test_NN_IC
}

os.makedirs(results_folder, exist_ok=True)

for name, arr in to_save.items():
    np.save(os.path.join(results_folder, f'{name}.npy'), arr)
# -------------------------------------------------------------------------








# ---------------------------------------------------------------
import os
import numpy as np

results_folder = "results_folder"

file_names = [
    'beta_g_D1.npy',
    'gamma_g_D1.npy',
    'beta_g_L1.npy',
    'gamma_g_L1.npy',
    'non_g1_D1.npy',
    'non_g2_D1.npy',
    'mid_g1_D1.npy',
    'mid_g2_D1.npy',
    'NN_IC_g1.npy',
    'beta_g_D0.npy',
    'gamma_g_D0.npy',
    'beta_g_L0.npy',
    'gamma_g_L0.npy',
    'non_g1_D0.npy',
    'non_g2_D0.npy',
    'mid_g1_D0.npy',
    'mid_g2_D0.npy',
    'NN_IC_g0.npy',
    'C_D.npy',
    'C_L.npy',
    'C_D_non.npy',
    'C_D_mid.npy',
    'NN_IC_B.npy',
    'I_T_t.npy',
    'S_t_W_test_DGAHM_non.npy',
    'S_t_W_test_DGAHM_mid.npy',
    'S_t_W_test_DGAHM.npy',
    'S_t_W_test_EHM.npy',
    'S_t_W_test_NN_IC.npy'
]

data = {}
for file_name in file_names:
    file_path = os.path.join(results_folder, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found!")

    data[file_name[:-4]] = np.load(file_path, allow_pickle=False)


beta_g_D1  = data['beta_g_D1']
gamma_g_D1 = data['gamma_g_D1']
beta_g_L1  = data['beta_g_L1']
gamma_g_L1 = data['gamma_g_L1']
non_g1_D1  = data['non_g1_D1']
non_g2_D1  = data['non_g2_D1']
mid_g1_D1  = data['mid_g1_D1']
mid_g2_D1  = data['mid_g2_D1']
mid_g2_D1  = data['mid_g2_D1']
NN_IC_g1  = data['NN_IC_g1']


beta_g_D0  = data['beta_g_D0']
gamma_g_D0 = data['gamma_g_D0']
beta_g_L0  = data['beta_g_L0']
gamma_g_L0 = data['gamma_g_L0']
non_g1_D0  = data['non_g1_D0']
non_g2_D0  = data['non_g2_D0']
mid_g1_D0  = data['mid_g1_D0']
mid_g2_D0  = data['mid_g2_D0']
NN_IC_g0  = data['NN_IC_g0']


C_D      = data['C_D']
C_L      = data['C_L']
C_D_non  = data['C_D_non']
C_D_mid  = data['C_D_mid']
NN_IC_B  = data['NN_IC_B']

I_T_t = data['I_T_t']
S_t_W_test_DGAHM_non = data['S_t_W_test_DGAHM_non']
S_t_W_test_DGAHM_mid = data['S_t_W_test_DGAHM_mid']
S_t_W_test_DGAHM = data['S_t_W_test_DGAHM']
S_t_W_test_EHM = data['S_t_W_test_EHM']
S_t_W_test_NN_IC = data['S_t_W_test_NN_IC']

#%% --------------

Non_g1_D1 = np.mean(non_g1_D1,axis=0) 
Non_g2_D1 = np.mean(non_g2_D1,axis=0) 
Non_g1_D0 = np.mean(non_g1_D0,axis=0) 
Non_g2_D0 = np.mean(non_g2_D0,axis=0)
C_D_Non = np.mean(C_D_non, axis=0) 


Mid_g1_D1 = np.mean(mid_g1_D1,axis=0) 
Mid_g2_D1 = np.mean(mid_g2_D1,axis=0) 
Mid_g1_D0 = np.mean(mid_g1_D0,axis=0) 
Mid_g2_D0 = np.mean(mid_g2_D0,axis=0)
C_D_Mid = np.mean(C_D_mid, axis=0) 


Beta_g_D1 = np.mean(beta_g_D1,axis=0) 
Gamma_g_D1 = np.mean(gamma_g_D1,axis=0) 
Beta_g_D0 = np.mean(beta_g_D0,axis=0) 
Gamma_g_D0 = np.mean(gamma_g_D0,axis=0) 
C_D = np.mean(C_D, axis=0) 
Lamd_U_D1 = I_S(m,C_D,U_sort1*np.exp(Beta_g_D1),nodevec) 
Lamd_V_D1 = I_S(m,C_D,V_sort1*np.exp(Beta_g_D1),nodevec) 
Lamd_U_D0 = I_S(m,C_D,U_sort0*np.exp(Beta_g_D0),nodevec) 
Lamd_V_D0 = I_S(m,C_D,V_sort0*np.exp(Beta_g_D0),nodevec) 

Beta_g_L1 = np.mean(beta_g_L1,axis=0) 
Gamma_g_L1 = np.mean(gamma_g_L1,axis=0) 
Beta_g_L0 = np.mean(beta_g_L0,axis=0) 
Gamma_g_L0 = np.mean(gamma_g_L0,axis=0) 
C_L = np.mean(C_L, axis=0) 
Lamd_U_L1 = I_S(m_L,C_L,U_sort1*np.exp(Beta_g_L1),nodevec_L) 
Lamd_V_L1 = I_S(m_L,C_L,V_sort1*np.exp(Beta_g_L1),nodevec_L) 
Lamd_U_L0 = I_S(m_L,C_L,U_sort0*np.exp(Beta_g_L0),nodevec_L) 
Lamd_V_L0 = I_S(m_L,C_L,V_sort0*np.exp(Beta_g_L0),nodevec_L) 

NN_IC_B = np.mean(NN_IC_B,axis=0) 
Lambda_U_NN_IC1 = np.dot(Bern_S(m_n, U_sort1, 0, 30), NN_IC_B)
Lambda_V_NN_IC1 = np.dot(Bern_S(m_n, V_sort1, 0, 30), NN_IC_B)
Lambda_U_NN_IC0 = np.dot(Bern_S(m_n, U_sort0, 0, 30), NN_IC_B)
Lambda_V_NN_IC0 = np.dot(Bern_S(m_n, V_sort0, 0, 30), NN_IC_B)
NN_IC_g1 = np.mean(NN_IC_g1, axis=0) 
NN_IC_g0 = np.mean(NN_IC_g0, axis=0) 
#%% Prediction of survival function for 6 subjects
# Calculate and draw three graphs with delta3 = 1
for k in range(3):
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(1, 1, 1)
    ax5.set_xlabel("t",fontsize=8)      
    ax5.set_ylabel(r'$\hat{S}(t)$',fontsize=8)
    ax5.tick_params(axis='both',labelsize=6)
    # Shift the position, set the origin to intersect
    ax5.xaxis.set_ticks_position('bottom')
    ax5.spines['bottom'].set_position(('data',0))
    ax5.yaxis.set_ticks_position('left')
    ax5.spines['left'].set_position(('data',0))
    ax5.grid(True)
    # Calculate S(t)
    Non_g_D1_1_k = np.exp(Non_g1_D1[round(n_V1*0.25*(k+1))])
    Non_g_D1_2_k = np.exp(Non_g2_D1[round(n_V1*0.25*(k+1))])
    Non_St_D1 = np.exp(-I_S(m,C_D_Non,V_value*Non_g_D1_1_k,nodevec) * Non_g_D1_2_k)

    Mid_g_D1_1_k = np.exp(Mid_g1_D1[round(n_V1*0.25*(k+1))])
    Mid_g_D1_2_k = np.exp(Mid_g2_D1[round(n_V1*0.25*(k+1))])
    Mid_St_D1 = np.exp(-I_S(m,C_D_Mid,V_value*Mid_g_D1_1_k,nodevec) * Mid_g_D1_2_k)


    Beta_g_D1_k = np.exp(Beta_g_D1[round(n_V1*0.25*(k+1))])
    Gamma_g_D1_k = np.exp(Gamma_g_D1[round(n_V1*0.25*(k+1))])
    St_D1 = np.exp(-I_S(m,C_D,V_value*Beta_g_D1_k,nodevec) * Gamma_g_D1_k)
    
    Beta_g_L1_k = np.exp(Beta_g_L1[round(n_V1*0.25*(k+1))])
    Gamma_g_L1_k = np.exp(Gamma_g_L1[round(n_V1*0.25*(k+1))])
    St_L1 = np.exp(-I_S(m_L,C_L,V_value*Beta_g_L1_k,nodevec_L) * Gamma_g_L1_k)



    NN_IC_g1_k = np.exp(NN_IC_g1[round(n_V1*0.25*(k+1))])
    Lambda_NN_IC_1 = np.dot(Bern_S(m_n, V_value, 0, 30), NN_IC_B)

    St_NN_IC1 = np.exp(-Lambda_NN_IC_1 * NN_IC_g1_k)
    
    # drawing 
    ax5.plot(V_value, Non_St_D1, color='green', linestyle='-')
    ax5.plot(V_value, Mid_St_D1, color='purple', linestyle='--')
    ax5.plot(V_value, St_D1, color='blue', linestyle=':')
    ax5.plot(V_value, St_L1, color='orange', linestyle='-.')
    ax5.plot(V_value, St_NN_IC1, color='black', linestyle='--')
    ax5.plot(V_value, 0.5*np.ones(len(V_value)), color='red', linestyle='--')
    ax5.legend(loc='best', fontsize=6)
    # save figures
    if (k==0):
        ax5.plot(V1_025, np.exp(-I_S(m,C_D_Non,np.array([V1_025*Non_g_D1_1_k]),nodevec) * Non_g_D1_2_k), label='DGAHM-Non', marker='v', markersize=4, ls='-', color='green')
        ax5.plot(V1_025, np.exp(-I_S(m,C_D_Mid,np.array([V1_025*Mid_g_D1_1_k]),nodevec) * Mid_g_D1_2_k), label='DGAHM-Mid', marker='s', markersize=4, ls='-', color='purple')
        ax5.plot(V1_025, np.exp(-I_S(m,C_D,np.array([V1_025*Beta_g_D1_k]),nodevec) * Gamma_g_D1_k), label='DGAHM', marker='p', markersize=4, ls='-', color='blue')
        ax5.plot(V1_025, np.exp(-I_S(m_L,C_L,np.array([V1_025*Beta_g_L1_k]),nodevec_L) * Gamma_g_L1_k), label='EHM', marker='o', markersize=4, ls='--', color='orange')
        ax5.plot(V1_025, np.exp(-np.dot(Bern_poly_basis(m_n, V1_025, 0, 30), NN_IC_B) * NN_IC_g1_k), label='NN-IC', marker='h', markersize=4, ls='-', color='black')

        ax5.plot(np.array([V1_025,V1_025]), np.array([0,np.max([np.exp(-I_S(m,C_D_Non,np.array([V1_025*Non_g_D1_1_k]),nodevec) * Non_g_D1_2_k), np.exp(-I_S(m,C_D_Mid,np.array([V1_025*Mid_g_D1_1_k]),nodevec) * Mid_g_D1_2_k), np.exp(-I_S(m,C_D,np.array([V1_025*Beta_g_D1_k]),nodevec) * Gamma_g_D1_k ), np.exp(-I_S(m_L,C_L,np.array([V1_025*Beta_g_L1_k]),nodevec_L) * Gamma_g_L1_k), np.array([np.exp(-np.dot(Bern_poly_basis(m_n, V1_025, 0, 30), NN_IC_B) * NN_IC_g1_k)])])], dtype='float32'), color='k', linestyle='--')
        ax5.legend(loc='best', fontsize=6)
        ax5.set_title(r'$\Delta_3=1, 25^{\rm{th}}$', fontsize=10) 
        fig5.savefig('fig1_25.jpeg', dpi=400, bbox_inches='tight')
    elif (k==1):
        ax5.plot(V1_050, np.exp(-I_S(m,C_D_Non,np.array([V1_050*Non_g_D1_1_k]),nodevec) * Non_g_D1_2_k), label='DGAHM-Non', marker='v', markersize=4, ls='-', color='green')
        ax5.plot(V1_050, np.exp(-I_S(m,C_D_Mid,np.array([V1_050*Mid_g_D1_1_k]),nodevec) * Mid_g_D1_2_k), label='DGAHM-Mid', marker='s', markersize=4, ls='-', color='purple')
        ax5.plot(V1_050, np.exp(-I_S(m,C_D,np.array([V1_050*Beta_g_D1_k]),nodevec) * Gamma_g_D1_k), label='DGAHM', marker='p', markersize=4, ls='-', color='blue')
        ax5.plot(V1_050, np.exp(-I_S(m_L,C_L,np.array([V1_050*Beta_g_L1_k]),nodevec_L) * Gamma_g_L1_k), label='EHM', marker='o', markersize=4, ls='--', color='orange')
        ax5.plot(V1_050, np.exp(-np.dot(Bern_poly_basis(m_n, V1_050, 0, 30), NN_IC_B) * NN_IC_g1_k), label='NN-IC', marker='h', markersize=4, ls='-', color='black')
        ax5.plot(np.array([V1_050,V1_050]), np.array([0,np.max([np.exp(-I_S(m,C_D_Non,np.array([V1_050*Non_g_D1_1_k]),nodevec) * Non_g_D1_2_k), np.exp(-I_S(m,C_D_Mid,np.array([V1_050*Mid_g_D1_1_k]),nodevec) * Mid_g_D1_2_k), np.exp(-I_S(m,C_D,np.array([V1_050*Beta_g_D1_k]),nodevec) * Gamma_g_D1_k ), np.exp(-I_S(m_L,C_L,np.array([V1_050*Beta_g_L1_k]),nodevec_L) * Gamma_g_L1_k), np.array([np.exp(-np.dot(Bern_poly_basis(m_n, V1_050, 0, 30), NN_IC_B) * NN_IC_g1_k)])])], dtype='float32'), color='k', linestyle='--')
        ax5.legend(loc='best', fontsize=6)
        ax5.set_title(r'$\Delta_3=1, 50^{\rm{th}}$', fontsize=10) 
        fig5.savefig('fig1_50.jpeg', dpi=400, bbox_inches='tight')
    else:
        ax5.plot(V1_075, np.exp(-I_S(m,C_D_Non,np.array([V1_075*Non_g_D1_1_k]),nodevec) * Non_g_D1_2_k), label='DGAHM-Non', marker='v', markersize=4, ls='-', color='green')
        ax5.plot(V1_075, np.exp(-I_S(m,C_D_Mid,np.array([V1_075*Mid_g_D1_1_k]),nodevec) * Mid_g_D1_2_k), label='DGAHM-Mid', marker='s', markersize=4, ls='-', color='purple')
        ax5.plot(V1_075, np.exp(-I_S(m,C_D,np.array([V1_075*Beta_g_D1_k]),nodevec) * Gamma_g_D1_k), label='DGAHM', marker='p', markersize=4, ls='-', color='blue')
        ax5.plot(V1_075, np.exp(-I_S(m_L,C_L,np.array([V1_075*Beta_g_L1_k]),nodevec_L) * Gamma_g_L1_k), label='EHM', marker='o', markersize=4, ls='--', color='orange')
        ax5.plot(V1_075, np.exp(-np.dot(Bern_poly_basis(m_n, V1_075, 0, 30), NN_IC_B) * NN_IC_g1_k), label='NN-IC', marker='h', markersize=4, ls='-', color='black')
        ax5.plot(np.array([V1_075,V1_075]), np.array([0,np.max([np.exp(-I_S(m,C_D_Non,np.array([V1_075*Non_g_D1_1_k]),nodevec) * Non_g_D1_2_k), np.exp(-I_S(m,C_D_Mid,np.array([V1_075*Mid_g_D1_1_k]),nodevec) * Mid_g_D1_2_k), np.exp(-I_S(m,C_D,np.array([V1_075*Beta_g_D1_k]),nodevec) * Gamma_g_D1_k ), np.exp(-I_S(m_L,C_L,np.array([V1_075*Beta_g_L1_k]),nodevec_L) * Gamma_g_L1_k), np.array([np.exp(-np.dot(Bern_poly_basis(m_n, V1_075, 0, 30), NN_IC_B) * NN_IC_g1_k)])])], dtype='float32'), color='k', linestyle='--')
        ax5.legend(loc='best', fontsize=6)
        ax5.set_title(r'$\Delta_3=1, 75^{\rm{th}}$', fontsize=10) 
        fig5.savefig('fig1_75.jpeg', dpi=400, bbox_inches='tight')

# Calculate and draw three figures with delta3=0
for k in range(3):
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(1, 1, 1)
    ax6.set_xlabel("t",fontsize=8)       
    ax6.set_ylabel(r'$\hat{S}(t)$',fontsize=8) 
    ax6.tick_params(axis='both',labelsize=6) 
    # Shift the position, set the origin to intersect
    ax6.xaxis.set_ticks_position('bottom')
    ax6.spines['bottom'].set_position(('data',0))
    ax6.yaxis.set_ticks_position('left')
    ax6.spines['left'].set_position(('data',0))
    ax6.grid(True)
    # Calculate S(t)
    Non_g_D0_1_k = np.exp(Non_g1_D0[round(n_V0*0.25*(k+1))])
    Non_g_D0_2_k = np.exp(Non_g2_D0[round(n_V0*0.25*(k+1))])
    Non_St_D0 = np.exp(-I_S(m,C_D_Non,V_value*Non_g_D0_1_k,nodevec) * Non_g_D0_2_k)

    Mid_g_D0_1_k = np.exp(Mid_g1_D0[round(n_V0*0.25*(k+1))])
    Mid_g_D0_2_k = np.exp(Mid_g2_D0[round(n_V0*0.25*(k+1))])
    Mid_St_D0 = np.exp(-I_S(m,C_D_Mid,V_value*Mid_g_D0_1_k,nodevec) * Mid_g_D0_2_k)


    Beta_g_D0_k = np.exp(Beta_g_D0[round(n_V0*0.25*(k+1))])
    Gamma_g_D0_k = np.exp(Gamma_g_D0[round(n_V0*0.25*(k+1))])
    St_D0 = np.exp(-I_S(m,C_D,V_value*Beta_g_D0_k,nodevec) * Gamma_g_D0_k)
    
    Beta_g_L0_k = np.exp(Beta_g_L0[round(n_V0*0.25*(k+1))])
    Gamma_g_L0_k = np.exp(Gamma_g_L0[round(n_V0*0.25*(k+1))])
    St_L0 = np.exp(-I_S(m_L,C_L,V_value*Beta_g_L0_k,nodevec_L) * Gamma_g_L0_k)

    NN_IC_g0_k = np.exp(NN_IC_g0[round(n_V0*0.25*(k+1))])
    Lambda_NN_IC_0 = np.dot(Bern_S(m_n, V_value, 0, 30), NN_IC_B)

    St_NN_IC0 = np.exp(-Lambda_NN_IC_0 * NN_IC_g0_k)
    
    # drawing 
    ax6.plot(V_value, Non_St_D0, color='green', linestyle='-')
    ax6.plot(V_value, Mid_St_D0, color='purple', linestyle='--')
    ax6.plot(V_value, St_D0, color='blue', linestyle=':')
    ax6.plot(V_value, St_L0, color='orange', linestyle='-.')
    ax6.plot(V_value, St_NN_IC0, color='black', linestyle='--')
    ax6.plot(V_value, 0.5*np.ones(len(V_value)), color='red', linestyle='--')
    if (k==0):
        ax6.plot(V0_025, np.exp(-I_S(m,C_D_Non,np.array([V0_025*Non_g_D0_1_k]),nodevec) * Non_g_D0_2_k), label='DGAHM-Non', marker='v', markersize=4, ls='-', color='green')
        ax6.plot(V0_025, np.exp(-I_S(m,C_D_Mid,np.array([V0_025*Mid_g_D0_1_k]),nodevec) * Mid_g_D0_2_k), label='DGAHM-Mid', marker='s', markersize=4, ls='-', color='purple')
        ax6.plot(V0_025, np.exp(-I_S(m,C_D,np.array([V0_025*Beta_g_D0_k]),nodevec) * Gamma_g_D0_k), label='DGAHM', marker='p', markersize=4, ls='-', color='blue')
        ax6.plot(V0_025, np.exp(-I_S(m_L,C_L,np.array([V0_025*Beta_g_L0_k]),nodevec_L) * Gamma_g_L0_k), label='EHM', marker='o', markersize=4, ls='--', color='orange')
        ax6.plot(V0_025, np.exp(-np.dot(Bern_poly_basis(m_n, V0_025, 0, 30), NN_IC_B) * NN_IC_g0_k), label='NN-IC', marker='h', markersize=4, ls='-', color='black')

        ax6.plot(np.array([V0_025,V0_025]), np.array([0,np.max([np.exp(-I_S(m,C_D_Non,np.array([V0_025*Non_g_D0_1_k]),nodevec) * Non_g_D0_2_k), np.exp(-I_S(m,C_D_Mid,np.array([V0_025*Mid_g_D0_1_k]),nodevec) * Mid_g_D0_2_k), np.exp(-I_S(m,C_D,np.array([V0_025*Beta_g_D0_k]),nodevec) * Gamma_g_D0_k ), np.exp(-I_S(m_L,C_L,np.array([V0_025*Beta_g_L0_k]),nodevec_L) * Gamma_g_L0_k), np.array([np.exp(-np.dot(Bern_poly_basis(m_n, V0_025, 0, 30), NN_IC_B) * NN_IC_g0_k)])])], dtype='float32'), color='k', linestyle='--')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$\Delta_3=0, 25^{\rm{th}}$', fontsize=10) 
        fig6.savefig('fig0_25.jpeg', dpi=400, bbox_inches='tight')
    elif (k==1):
        ax6.plot(V0_050, np.exp(-I_S(m,C_D_Non,np.array([V0_050*Non_g_D0_1_k]),nodevec) * Non_g_D0_2_k), label='DGAHM-Non', marker='v', markersize=4, ls='-', color='green')
        ax6.plot(V0_050, np.exp(-I_S(m,C_D_Mid,np.array([V0_050*Mid_g_D0_1_k]),nodevec) * Mid_g_D0_2_k), label='DGAHM-Mid', marker='s', markersize=4, ls='-', color='purple')
        ax6.plot(V0_050, np.exp(-I_S(m,C_D,np.array([V0_050*Beta_g_D0_k]),nodevec) * Gamma_g_D0_k), label='DGAHM', marker='p', markersize=4, ls='-', color='blue')
        ax6.plot(V0_050, np.exp(-I_S(m_L,C_L,np.array([V0_050*Beta_g_L0_k]),nodevec_L) * Gamma_g_L0_k), label='EHM', marker='o', markersize=4, ls='--', color='orange')
        ax6.plot(V0_050, np.exp(-np.dot(Bern_poly_basis(m_n, V0_050, 0, 30), NN_IC_B) * NN_IC_g0_k), label='NN-IC', marker='h', markersize=4, ls='-', color='black')

        ax6.plot(np.array([V0_050,V0_050]), np.array([0,np.max([np.exp(-I_S(m,C_D_Non,np.array([V0_050*Non_g_D0_1_k]),nodevec) * Non_g_D0_2_k), np.exp(-I_S(m,C_D_Mid,np.array([V0_050*Mid_g_D0_1_k]),nodevec) * Mid_g_D0_2_k), np.exp(-I_S(m,C_D,np.array([V0_050*Beta_g_D0_k]),nodevec) * Gamma_g_D0_k ), np.exp(-I_S(m_L,C_L,np.array([V0_050*Beta_g_L0_k]),nodevec_L) * Gamma_g_L0_k), np.array([np.exp(-np.dot(Bern_poly_basis(m_n, V0_050, 0, 30), NN_IC_B) * NN_IC_g0_k)])])], dtype='float32'), color='k', linestyle='--')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$\Delta_3=0, 50^{\rm{th}}$', fontsize=10) 
        fig6.savefig('fig0_50.jpeg', dpi=400, bbox_inches='tight')
    else:
        ax6.plot(V0_075, np.exp(-I_S(m,C_D_Non,np.array([V0_075*Non_g_D0_1_k]),nodevec) * Non_g_D0_2_k), label='DGAHM-Non', marker='v', markersize=4, ls='-', color='green')
        ax6.plot(V0_075, np.exp(-I_S(m,C_D_Mid,np.array([V0_075*Mid_g_D0_1_k]),nodevec) * Mid_g_D0_2_k), label='DGAHM-Mid', marker='s', markersize=4, ls='-', color='purple')
        ax6.plot(V0_075, np.exp(-I_S(m,C_D,np.array([V0_075*Beta_g_D0_k]),nodevec) * Gamma_g_D0_k), label='DGAHM', marker='p', markersize=4, ls='-', color='blue')
        ax6.plot(V0_075, np.exp(-I_S(m_L,C_L,np.array([V0_075*Beta_g_L0_k]),nodevec_L) * Gamma_g_L0_k), label='EHM', marker='o', markersize=4, ls='--', color='orange')
        ax6.plot(V0_075, np.exp(-np.dot(Bern_poly_basis(m_n, V0_075, 0, 30), NN_IC_B) * NN_IC_g0_k), label='NN-IC', marker='h', markersize=4, ls='-', color='black')

        ax6.plot(np.array([V0_075,V0_075]), np.array([0,np.max([np.exp(-I_S(m,C_D_Non,np.array([V0_075*Non_g_D0_1_k]),nodevec) * Non_g_D0_2_k), np.exp(-I_S(m,C_D_Mid,np.array([V0_075*Mid_g_D0_1_k]),nodevec) * Mid_g_D0_2_k), np.exp(-I_S(m,C_D,np.array([V0_075*Beta_g_D0_k]),nodevec) * Gamma_g_D0_k ), np.exp(-I_S(m_L,C_L,np.array([V0_075*Beta_g_L0_k]),nodevec_L) * Gamma_g_L0_k), np.array([np.exp(-np.dot(Bern_poly_basis(m_n, V0_075, 0, 30), NN_IC_B) * NN_IC_g0_k)])])], dtype='float32'), color='k', linestyle='--')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$\Delta_3=0, 75^{\rm{th}}$', fontsize=10) 
        fig6.savefig('fig0_75.jpeg', dpi=400, bbox_inches='tight')


#%%-------------IBS----------------
IBS_DGAHM_non = np.mean((I_T_t - S_t_W_test_DGAHM_non) ** 2)
IBS_DGAHM_mid = np.mean((I_T_t - S_t_W_test_DGAHM_mid) ** 2)
IBS_DGAHM = np.mean((I_T_t - S_t_W_test_DGAHM) ** 2)
IBS_EHM = np.mean((I_T_t - S_t_W_test_EHM) ** 2)
IBS_NN_IC = np.mean((I_T_t - S_t_W_test_NN_IC) ** 2)


data_ibs = {
    'DGAHM_non': [IBS_DGAHM_non],
    'DGAHM_mid': [IBS_DGAHM_mid],
    'DGAHM': [IBS_DGAHM],
    'EHM': [IBS_EHM],
    'NN_IC': [IBS_NN_IC]
}
df = pd.DataFrame(data_ibs)
df.to_csv('ibs_results.csv', index=False)
