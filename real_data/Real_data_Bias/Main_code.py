#%% ----------------------
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from iteration_deep import Est_deep
from iteration_deep_median import Est_deep_median
from iteration_linear import Est_linear
from Least_gamma import LFD_gamma
from I_spline import I_U
from B_spline2 import B_S2

#%% ---------- -------------
def set_seed(seed):
    np.random.seed(seed)  
    torch.manual_seed(seed) 
#%% ---------------------
set_seed(20)
#%% ----------- ------------
p = 3 
n_layer = 3 
n_node = 100
n_epoch = 1000
n_lr = 3e-4 
m = 12

#%% Data Processing
df = pd.read_csv('data_center.csv') 

Z = np.array(df[["BMI01","GLUCOS01","HDL01","TCHSIU01"]], dtype='float32')
X = np.array(df[["V1AGE01","SBPA21","SBPA22","RACEGRP","GENDER","Cen1","Cen2","Cen3"]], dtype='float32')
U = np.array(df["U_year"], dtype='float32')
V = np.array(df["V_year"], dtype='float32')
De1 = np.array(df["De1"], dtype='float32')
De2 = np.array(df["De2"], dtype='float32')
De3 = np.array(df["De3"], dtype='float32')


nodevec = np.array(np.linspace(0, 200, m+2), dtype="float32")
c_initial = np.array(0.5*np.ones(m+p), dtype="float32") 
theta_initial = np.array(np.zeros(2*Z.shape[1]), dtype='float32')

#%% =========================== 
A = np.arange(len(U))
np.random.shuffle(A)

Z_R = Z[A]
X_R = X[A]
U_R = U[A]
V_R = V[A]
De1_R = De1[A]
De2_R = De2[A]
De3_R = De3[A]

# ----------------------
# ---train data 11000
Z_train = Z_R[np.arange(11000)]
X_train = X_R[np.arange(11000)]
U_train = U_R[np.arange(11000)]
V_train = V_R[np.arange(11000)]
De1_train = De1_R[np.arange(11000)]
De2_train = De2_R[np.arange(11000)]
De3_train = De3_R[np.arange(11000)]
# ---test data 2204
Z_valid = Z_R[np.arange(11000,len(U))]
X_valid = X_R[np.arange(11000,len(U))]
U_valid = U_R[np.arange(11000,len(U))]
V_valid = V_R[np.arange(11000,len(U))]
De1_valid = De1_R[np.arange(11000,len(U))]
De2_valid = De2_R[np.arange(11000,len(U))]
De3_valid = De3_R[np.arange(11000,len(U))]


n = len(U_train)
z_d = Z_train.shape[1]



#%% ----------------------DGAHM----------------------------------
Est_D = Est_deep(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
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
# compute the least favorable direction for gamma1
abc_gamma_D1 = LFD_gamma(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D1 = De1_train * Z_train[:,0] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,0] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,0] * f_V_D
    
L_g1_gamma_D1 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D1[:,0]
    
L_g2_gamma_D1 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D1[:,1] 
    
L_lambda_gamma_D1 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D1[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D1[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D1[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D1[:,3]
    
I_1_D = L_gamma_D1 - L_g1_gamma_D1 - L_g2_gamma_D1 - L_lambda_gamma_D1
    
# compute the least favorable direction for gamma2
abc_gamma_D2 = LFD_gamma(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D2 = De1_train * Z_train[:,1] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,1] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,1] * f_V_D
    
L_g1_gamma_D2 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D2[:,0]
    
L_g2_gamma_D2 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D2[:,1] 
    
L_lambda_gamma_D2 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D2[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D2[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D2[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D2[:,3]
    
I_2_D = L_gamma_D2 - L_g1_gamma_D2 - L_g2_gamma_D2 - L_lambda_gamma_D2
    
# compute the least favorable direction for gamma3
abc_gamma_D3 = LFD_gamma(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D3 = De1_train * Z_train[:,2] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,2] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,2] * f_V_D
    
L_g1_gamma_D3 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D3[:,0]
    
L_g2_gamma_D3 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D3[:,1] 
    
L_lambda_gamma_D3 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D3[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D3[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D3[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D3[:,3]
    
I_3_D = L_gamma_D3 - L_g1_gamma_D3 - L_g2_gamma_D3 - L_lambda_gamma_D3
    
# compute the least favorable direction for gamma4
abc_gamma_D4 = LFD_gamma(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D4 = De1_train * Z_train[:,3] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,3] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,3] * f_V_D
    
L_g1_gamma_D4 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D4[:,0]
    
L_g2_gamma_D4 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D4[:,1] 
    
L_lambda_gamma_D4 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D4[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D4[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D4[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D4[:,3]
    
I_4_D = L_gamma_D4 - L_g1_gamma_D4 - L_g2_gamma_D4 - L_lambda_gamma_D4
    
    
Info_D = np.zeros((4,4))
Info_D[0,0] = np.mean(I_1_D**2)
Info_D[1,1] = np.mean(I_2_D**2)
Info_D[2,2] = np.mean(I_3_D**2)
Info_D[3,3] = np.mean(I_4_D**2)
Info_D[0,1] = np.mean(I_1_D*I_2_D)
Info_D[1,0] = Info_D[0,1]
Info_D[0,2] = np.mean(I_1_D*I_3_D)
Info_D[2,0] = Info_D[0,2]
Info_D[0,3] = np.mean(I_1_D*I_4_D)
Info_D[3,0] = Info_D[0,3]
Info_D[1,2] = np.mean(I_2_D*I_3_D)
Info_D[2,1] = Info_D[1,2]
Info_D[1,3] = np.mean(I_2_D*I_4_D)
Info_D[3,1] = Info_D[1,3]
Info_D[2,3] = np.mean(I_3_D*I_4_D)
Info_D[3,2] = Info_D[2,3]
Sigma_D = np.linalg.inv(Info_D)/n


gamma_D = theta_D[z_d:(2*z_d)]
sd_D = np.sqrt(np.diag(Sigma_D))



#%% ----------------------DGAHM-Mid--------------------------------
Est_D_M = Est_deep_median(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
# theta_D_M
theta_D_M = Est_D_M['theta']

Ezg1_D_M = np.exp(np.dot(Z_train, theta_D_M[0:z_d]) + Est_D_M['g_train'][:,0])
Ezg2_D_M = np.exp(np.dot(Z_train, theta_D_M[z_d:(2*z_d)]) + Est_D_M['g_train'][:,1])
# compute \Lambda()
Iu_D_M = I_U(m, U_train * Ezg1_D_M, nodevec)
Iv_D_M = I_U(m, V_train * Ezg1_D_M, nodevec)
Lamb_U_D_M = np.dot(Iu_D_M, Est_D_M['C'])
Lamb_V_D_M = np.dot(Iv_D_M, Est_D_M['C'])
f_U_D_M = Lamb_U_D_M * Ezg2_D_M
f_V_D_M = Lamb_V_D_M * Ezg2_D_M
# compute \Lambda'()
Bu_D_M = B_S2(m, U_train * Ezg1_D_M, nodevec)
Bv_D_M = B_S2(m, V_train * Ezg1_D_M, nodevec)
dLamb_U_D_M = np.matmul(Bu_D_M, Est_D_M['C'])
dLamb_V_D_M = np.matmul(Bv_D_M, Est_D_M['C'])
# compute the least favorable direction for gamma1
abc_gamma_D_M_1 = LFD_gamma(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D_M['g_train'],Est_D_M['theta'],Est_D_M['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D_M_1 = De1_train * Z_train[:,0] * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Z_train[:,0] * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * Z_train[:,0] * f_V_D_M
    
L_g1_gamma_D_M_1 = (De1_train * U_train * dLamb_U_D_M * Ezg1_D_M * Ezg2_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Ezg1_D_M * Ezg2_D_M * (V_train * dLamb_V_D_M * np.exp(-f_V_D_M) - U_train * dLamb_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * V_train * dLamb_V_D_M * Ezg1_D_M * Ezg2_D_M) * abc_gamma_D_M_1[:,0]
    
L_g2_gamma_D_M_1 = (De1_train * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M) * abc_gamma_D_M_1[:,1] 
    
L_lambda_gamma_D_M_1 = De1_train * f_U_D_M * np.exp(-f_U_D_M) * abc_gamma_D_M_1[:,2] / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (abc_gamma_D_M_1[:,3] * f_V_D_M * np.exp(-f_V_D_M)- abc_gamma_D_M_1[:,2] * f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M * abc_gamma_D_M_1[:,3]
    
I_1_D_M = L_gamma_D_M_1 - L_g1_gamma_D_M_1 - L_g2_gamma_D_M_1 - L_lambda_gamma_D_M_1
    
# compute the least favorable direction for gamma2
abc_gamma_D_M_2 = LFD_gamma(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D_M['g_train'],Est_D_M['theta'],Est_D_M['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D_M_2 = De1_train * Z_train[:,1] * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Z_train[:,1] * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * Z_train[:,1] * f_V_D_M
    
L_g1_gamma_D_M_2 = (De1_train * U_train * dLamb_U_D_M * Ezg1_D_M * Ezg2_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Ezg1_D_M * Ezg2_D_M * (V_train * dLamb_V_D_M * np.exp(-f_V_D_M) - U_train * dLamb_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * V_train * dLamb_V_D_M * Ezg1_D_M * Ezg2_D_M) * abc_gamma_D_M_2[:,0]
    
L_g2_gamma_D_M_2 = (De1_train * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M) * abc_gamma_D_M_2[:,1] 
    
L_lambda_gamma_D_M_2 = De1_train * f_U_D_M * np.exp(-f_U_D_M) * abc_gamma_D_M_2[:,2] / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (abc_gamma_D_M_2[:,3] * f_V_D_M * np.exp(-f_V_D_M)- abc_gamma_D_M_2[:,2] * f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M * abc_gamma_D_M_2[:,3]
    
I_2_D_M = L_gamma_D_M_2 - L_g1_gamma_D_M_2 - L_g2_gamma_D_M_2 - L_lambda_gamma_D_M_2
    
# compute the least favorable direction for gamma3
abc_gamma_D_M_3 = LFD_gamma(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D_M['g_train'],Est_D_M['theta'],Est_D_M['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D_M_3 = De1_train * Z_train[:,2] * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Z_train[:,2] * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * Z_train[:,2] * f_V_D_M
    
L_g1_gamma_D_M_3 = (De1_train * U_train * dLamb_U_D_M * Ezg1_D_M * Ezg2_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Ezg1_D_M * Ezg2_D_M * (V_train * dLamb_V_D_M * np.exp(-f_V_D_M) - U_train * dLamb_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * V_train * dLamb_V_D_M * Ezg1_D_M * Ezg2_D_M) * abc_gamma_D_M_3[:,0]
    
L_g2_gamma_D_M_3 = (De1_train * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M) * abc_gamma_D_M_3[:,1] 
    
L_lambda_gamma_D_M_3 = De1_train * f_U_D_M * np.exp(-f_U_D_M) * abc_gamma_D_M_3[:,2] / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (abc_gamma_D_M_3[:,3] * f_V_D_M * np.exp(-f_V_D_M)- abc_gamma_D_M_3[:,2] * f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M * abc_gamma_D_M_3[:,3]
    
I_3_D_M = L_gamma_D_M_3 - L_g1_gamma_D_M_3 - L_g2_gamma_D_M_3 - L_lambda_gamma_D_M_3
    
# compute the least favorable direction for gamma4
abc_gamma_D_M_4 = LFD_gamma(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D_M['g_train'],Est_D_M['theta'],Est_D_M['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
L_gamma_D_M_4 = De1_train * Z_train[:,3] * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Z_train[:,3] * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * Z_train[:,3] * f_V_D_M
    
L_g1_gamma_D_M_4 = (De1_train * U_train * dLamb_U_D_M * Ezg1_D_M * Ezg2_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * Ezg1_D_M * Ezg2_D_M * (V_train * dLamb_V_D_M * np.exp(-f_V_D_M) - U_train * dLamb_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * V_train * dLamb_V_D_M * Ezg1_D_M * Ezg2_D_M) * abc_gamma_D_M_4[:,0]
    
L_g2_gamma_D_M_4 = (De1_train * f_U_D_M * np.exp(-f_U_D_M) / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (f_V_D_M * np.exp(-f_V_D_M)- f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M) * abc_gamma_D_M_4[:,1] 
    
L_lambda_gamma_D_M_4 = De1_train * f_U_D_M * np.exp(-f_U_D_M) * abc_gamma_D_M_4[:,2] / (1 - np.exp(-f_U_D_M) + 1e-4) + De2_train * (abc_gamma_D_M_4[:,3] * f_V_D_M * np.exp(-f_V_D_M)- abc_gamma_D_M_4[:,2] * f_U_D_M * np.exp(-f_U_D_M)) / (np.exp(-f_U_D_M) - np.exp(-f_V_D_M) + 1e-4) - De3_train * f_V_D_M * abc_gamma_D_M_4[:,3]
    
I_4_D_M = L_gamma_D_M_4 - L_g1_gamma_D_M_4 - L_g2_gamma_D_M_4 - L_lambda_gamma_D_M_4
    
    
Info_D_M = np.zeros((4,4))
Info_D_M[0,0] = np.mean(I_1_D_M**2)
Info_D_M[1,1] = np.mean(I_2_D_M**2)
Info_D_M[2,2] = np.mean(I_3_D_M**2)
Info_D_M[3,3] = np.mean(I_4_D_M**2)
Info_D_M[0,1] = np.mean(I_1_D_M*I_2_D_M)
Info_D_M[1,0] = Info_D_M[0,1]
Info_D_M[0,2] = np.mean(I_1_D_M*I_3_D_M)
Info_D_M[2,0] = Info_D_M[0,2]
Info_D_M[0,3] = np.mean(I_1_D_M*I_4_D_M)
Info_D_M[3,0] = Info_D_M[0,3]
Info_D_M[1,2] = np.mean(I_2_D_M*I_3_D_M)
Info_D_M[2,1] = Info_D_M[1,2]
Info_D_M[1,3] = np.mean(I_2_D_M*I_4_D_M)
Info_D_M[3,1] = Info_D_M[1,3]
Info_D_M[2,3] = np.mean(I_3_D_M*I_4_D_M)
Info_D_M[3,2] = Info_D_M[2,3]
Sigma_D_M = np.linalg.inv(Info_D_M)/n

gamma_D_M = theta_D_M[z_d:(2*z_d)]
sd_D_M = np.sqrt(np.diag(Sigma_D_M))





#%% --------------------GHM-----------------------
Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train = Z_R,X_R,U_R,V_R,De1_R,De2_R,De3_R

Z_train = np.hstack((Z_train,X_train[:,0:3]))
X_train = X_train[:,3:]
theta_initial_L = np.array(np.zeros(2*Z_train.shape[1]), dtype='float32')


Est_L = Est_linear(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,theta_initial_L,nodevec,m,c_initial)

theta_L = Est_L['theta']

n_L = len(U_train)
z_d_L = Z_train.shape[1]

Ezg1_L = np.exp(np.dot(Z_train, theta_L[0:z_d_L]) + Est_L['g_train'][:,0])
Ezg2_L = np.exp(np.dot(Z_train, theta_L[z_d_L:(2*z_d_L)]) + Est_L['g_train'][:,1])
# compute \Lambda()
Iu_L = I_U(m, U_train * Ezg1_L, nodevec)
Iv_L = I_U(m, V_train * Ezg1_L, nodevec)
Lamb_U_L = np.dot(Iu_L, Est_L['C'])
Lamb_V_L = np.dot(Iv_L, Est_L['C'])
f_U_L = Lamb_U_L * Ezg2_L
f_V_L = Lamb_V_L * Ezg2_L
# compute \Lambda'()
Bu_L = B_S2(m, U_train * Ezg1_L, nodevec)
Bv_L = B_S2(m, V_train * Ezg1_L, nodevec)
dLamb_U_L = np.matmul(Bu_L, Est_L['C'])
dLamb_V_L = np.matmul(Bv_L, Est_L['C'])
# compute the least favorable direction for gamma1
abc_gamma_L1 = LFD_gamma(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)
    
L_gamma_L1 = De1_train * Z_train[:,0] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,0] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,0] * f_V_L
    
L_g1_gamma_L1 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L1[:,0]
    
L_g2_gamma_L1 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L1[:,1] 
    
L_lambda_gamma_L1 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L1[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L1[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L1[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L1[:,3]
    
I_1_L = L_gamma_L1 - L_g1_gamma_L1 - L_g2_gamma_L1 - L_lambda_gamma_L1
    
# compute the least favorable direction for gamma2
abc_gamma_L2 = LFD_gamma(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)
    
L_gamma_L2 = De1_train * Z_train[:,1] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,1] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,1] * f_V_L
    
L_g1_gamma_L2 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L2[:,0]
    
L_g2_gamma_L2 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L2[:,1] 
    
L_lambda_gamma_L2 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L2[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L2[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L2[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L2[:,3]
    
I_2_L = L_gamma_L2 - L_g1_gamma_L2 - L_g2_gamma_L2 - L_lambda_gamma_L2
    
# compute the least favorable direction for gamma3
abc_gamma_L3 = LFD_gamma(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)
    
L_gamma_L3 = De1_train * Z_train[:,2] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,2] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,2] * f_V_L
    
L_g1_gamma_L3 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L3[:,0]
    
L_g2_gamma_L3 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L3[:,1] 
    
L_lambda_gamma_L3 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L3[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L3[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L3[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L3[:,3]
    
I_3_L = L_gamma_L3 - L_g1_gamma_L3 - L_g2_gamma_L3 - L_lambda_gamma_L3
    
# compute the least favorable direction for gamma4
abc_gamma_L4 = LFD_gamma(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)
    
L_gamma_L4 = De1_train * Z_train[:,3] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,3] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,3] * f_V_L
    
L_g1_gamma_L4 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L4[:,0]
    
L_g2_gamma_L4 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L4[:,1] 
    
L_lambda_gamma_L4 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L4[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L4[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L4[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L4[:,3]
    
I_4_L = L_gamma_L4 - L_g1_gamma_L4 - L_g2_gamma_L4 - L_lambda_gamma_L4
    
    
# compute the least favorable direction for gamma5
abc_gamma_L5 = LFD_gamma(Z_train[:,4],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)
    
L_gamma_L5 = De1_train * Z_train[:,4] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,4] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,4] * f_V_L
    
L_g1_gamma_L5 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L5[:,0]
    
L_g2_gamma_L5 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L5[:,1] 
    
L_lambda_gamma_L5 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L5[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L5[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L5[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L5[:,3]
    
I_5_L = L_gamma_L5 - L_g1_gamma_L5 - L_g2_gamma_L5 - L_lambda_gamma_L5


    
# compute the least favorable direction for gamma6
abc_gamma_L6 = LFD_gamma(Z_train[:,5],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)
    
L_gamma_L6 = De1_train * Z_train[:,5] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,5] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,5] * f_V_L
    
L_g1_gamma_L6 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L6[:,0]
    
L_g2_gamma_L6 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L6[:,1] 
    
L_lambda_gamma_L6 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L6[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L6[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L6[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L6[:,3]
    
I_6_L = L_gamma_L6 - L_g1_gamma_L6 - L_g2_gamma_L6 - L_lambda_gamma_L6


# compute the least favorable direction for gamma7
abc_gamma_L7 = LFD_gamma(Z_train[:,6],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)
    
L_gamma_L7 = De1_train * Z_train[:,6] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,6] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,6] * f_V_L
    
L_g1_gamma_L7 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L7[:,0]
    
L_g2_gamma_L7 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L7[:,1] 
    
L_lambda_gamma_L7 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L7[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L7[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L7[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L7[:,3]
    
I_7_L = L_gamma_L7 - L_g1_gamma_L7 - L_g2_gamma_L7 - L_lambda_gamma_L7


I_matrix = np.column_stack([
    I_1_L, I_2_L, I_3_L, I_4_L, 
    I_5_L, I_6_L, I_7_L
])

N = len(I_1_L)
Info_L = (I_matrix.T @ I_matrix) / N

Sigma_L = np.linalg.inv(Info_L) / n_L

gamma_L = theta_L[z_d_L:(2*z_d_L)]
sd_L = np.sqrt(np.diag(Sigma_L))





# ----------------save results-------------------

dic_D = {"gamma_deep": gamma_D, "sd_deep": sd_D}
Result_deep = pd.DataFrame(dic_D,index=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
Result_deep.to_csv('Result_deep.csv')


dic_D_Mid = {"gamma_deep_mid": gamma_D_M, "sd_deep_mid": sd_D_M}
Result_deep_Mid = pd.DataFrame(dic_D_Mid,index=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
Result_deep_Mid.to_csv('Result_deep_Mid.csv')

dic_L = {"gamma_linear": gamma_L, "sd_linear": sd_L}
Result_linear = pd.DataFrame(dic_L,index=['gamma1', 'gamma2', 'gamma3', 'gamma4', 'gamma_age','gamma_systolic','gamma_diastolic'])
Result_linear.to_csv('Result_linear.csv')