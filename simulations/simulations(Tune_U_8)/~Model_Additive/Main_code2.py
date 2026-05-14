#%% ----------- import packages -----------
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed

# Assuming these modules are available in your working path
from data_generator import generate_case_2
from iteration_deep import Est_deep
from iteration_deep_median import Est_deep_median
from iteration_linear import Est_linear
from Least_beta import LFD_beta
from Least_gamma import LFD_gamma
from I_spline import I_U
from I_spline import I_S
from B_spline2 import B_S2

#%% ---------- Create Result Folder -------------
result_folder = 'result_folder'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f"Created folder: {result_folder}")
else:
    print(f"Folder already exists: {result_folder}")

#%% ---------- define seeds -------------
def set_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)

#%% ---------- set seed -------------
set_seed(1)

#%% -----------------------
tau = 5 
p = 3 
Set_n = np.array([500, 1000])
corr = 0.5
Set_layer = np.array([2, 2]) 
Set_node = np.array([40, 50]) 
n_epoch = 1000
Set_lr = np.array([3e-4, 3e-4]) 
theta0 = np.array([2, 1], dtype="float32") 

n_layer_D = 2 
beta_node_D = np.array([46, 48]) 
beta_lr_D = np.array([2e-4, 2e-4]) 
gamma_node_D = np.array([15, 12]) 
gamma_lr_D = np.array([1e-5, 1e-5])

n_layer_D_M = 2
beta_node_D_M = np.array([40, 40]) 
beta_lr_D_M = np.array([2e-4, 2e-4]) 
gamma_node_D_M = np.array([25, 25]) 
gamma_lr_D_M = np.array([3e-5, 3e-5]) 

n_layer_L = 2 
beta_node_L = np.array([50, 50]) 
beta_lr_L = np.array([4e-4, 4e-4]) 
gamma_node_L = np.array([40, 40]) 
gamma_lr_L = np.array([1e-5, 1e-5]) 




B = 200 # Simulation times
Num_Cores = 16 # Number of cores requested

#%% test data
test_data = generate_case_2(200, corr, theta0, tau)
X_test = test_data['X']
g1_true = test_data['g1_X']
g2_true = test_data['g2_X']
dim_x = X_test.shape[0] 

set_m = np.array([10, 10])

set_m_L = np.array([5, 5])

#%% X_1,X_2, Z=0,1
X_subject = np.array([[0.1,0.3,0.5,0.7],[0.8,0.5,0.2,0]], dtype='float32')
Z_subject = np.array([0, 1], dtype='float32')
g_1_X_subject_true = np.log(X_subject[:,0] + 1) / 3 + np.exp(X_subject[:,1]) / 5 + X_subject[:,2] ** 2 / 4 + X_subject[:,3] / 2 - 0.6
g_2_X_subject_true = X_subject[:,0] + np.sqrt(X_subject[:,1]) / 2 + np.log(X_subject[:,2] + 1) / 3 + X_subject[:,3] ** 2 / 4 - 0.65

t_value = np.array(np.linspace(0, tau, 30), dtype="float32") 
Lambda_Z0_X1 = t_value * np.exp(g_1_X_subject_true[0] + g_2_X_subject_true[0]) / 8 # Z=0, X=X1
Lambda_Z1_X1 = t_value * np.exp(theta0[0] + theta0[1] + g_1_X_subject_true[0] + g_2_X_subject_true[0]) / 8 # Z=1, X=X1
Lambda_Z0_X2 = t_value * np.exp(g_1_X_subject_true[1] + g_2_X_subject_true[1]) / 8 # Z=0, X=X2
Lambda_Z1_X2 = t_value * np.exp(theta0[0] + theta0[1] + g_1_X_subject_true[1] + g_2_X_subject_true[1]) / 8 # Z=1, X=X2


results_dict = {
    "t_value": t_value,
    "Lambda_Z0_X1": Lambda_Z0_X1,
    "Lambda_Z1_X1": Lambda_Z1_X1,
    "Lambda_Z0_X2": Lambda_Z0_X2,
    "Lambda_Z1_X2": Lambda_Z1_X2
}

npy_path = os.path.join(result_folder, 'True_results.npy')

np.save(npy_path, results_dict)





# ==============================================================
# Define Worker Function for Parallel Execution
# ==============================================================
def run_single_simulation(b, n, i, n_layer, n_node, n_lr, m, nodevec, m_L, nodevec_L):
    """
    Runs a single simulation iteration (b).
    Returns a dictionary of results.
    """
    set_seed(100 + b)
    c_initial = np.array(0.1*np.ones(m+p), dtype="float32") 
    c_initial_L = np.array(0.1*np.ones(m_L+p), dtype="float32")
    # theta_initial = np.array([1.85, 0.85], dtype="float32")
    
    # --- Generate Data ---
    train_data = generate_case_2(n, corr, theta0, tau)
    Z_train = train_data['Z']
    U_train = train_data['U']
    V_train = train_data['V']
    De1_train = train_data['De1']
    De2_train = train_data['De2']
    De3_train = train_data['De3']
    
    valid_data = generate_case_2(200, corr, theta0, tau)
    
    # --- Helper function for SE calculation (LFD method) ---
    def compute_se(Est, model_type):
        # Determine parameters based on model type
        if model_type == 'D':
            curr_n_layer = n_layer_D
            curr_beta_node = beta_node_D[i]
            curr_beta_lr = beta_lr_D[i]
            curr_gamma_node = gamma_node_D[i]
            curr_gamma_lr = gamma_lr_D[i]
            curr_m = m
            curr_nodevec = nodevec

        elif model_type == 'D_M':
            curr_n_layer = n_layer_D_M
            curr_beta_node = beta_node_D_M[i]
            curr_beta_lr = beta_lr_D_M[i]
            curr_gamma_node = gamma_node_D_M[i]
            curr_gamma_lr = gamma_lr_D_M[i]
            curr_m = m
            curr_nodevec = nodevec
        else: # 'L'
            curr_n_layer = n_layer_L
            curr_beta_node = beta_node_L[i]
            curr_beta_lr = beta_lr_L[i]
            curr_gamma_node = gamma_node_L[i]
            curr_gamma_lr = gamma_lr_L[i]
            curr_m = m_L
            curr_nodevec = nodevec_L

        Ezg1 = np.exp(Z_train * Est['theta'][0] + Est['g_train'][:,0])
        Ezg2 = np.exp(Z_train * Est['theta'][1] + Est['g_train'][:,1])
        
        # 使用传入的 m 和 nodevec
        Iu = I_U(curr_m, np.clip(U_train * Ezg1, 0, 95), curr_nodevec)
        Iv = I_U(curr_m, np.clip(V_train * Ezg1, 0, 95), curr_nodevec)
        Lamb_U = np.matmul(Iu, Est['C'])
        Lamb_V = np.matmul(Iv, Est['C'])
        f_U = Lamb_U * Ezg2
        f_V = Lamb_V * Ezg2
        Bu = B_S2(curr_m, np.clip(U_train * Ezg1, 0, 95), curr_nodevec)
        Bv = B_S2(curr_m, np.clip(V_train * Ezg1, 0, 95), curr_nodevec)
        dLamb_U = np.matmul(Bu, Est['C'])
        dLamb_V = np.matmul(Bv, Est['C']) 
        
        abc_beta = LFD_beta(train_data,Est['g_train'],Est['theta'],Est['C'],curr_m,curr_nodevec,curr_n_layer,n_node=curr_beta_node,n_lr=curr_beta_lr,n_epoch=200)
        
        L_beta =  De1_train * U_train * Z_train * Ezg1 * Ezg2 * np.exp(-f_U) * dLamb_U / (1 - np.exp(-f_U) + 1e-4) + De2_train * Z_train * Ezg1 * Ezg2 * (V_train * dLamb_V * np.exp(-f_V) - U_train * dLamb_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * V_train * Z_train * dLamb_V * Ezg1 * Ezg2
        
        L_g1_beta = (De1_train * U_train * dLamb_U * Ezg1 * Ezg2 * np.exp(-f_U) / (1 - np.exp(-f_U) + 1e-4) + De2_train * Ezg1 * Ezg2 * (V_train * dLamb_V * np.exp(-f_V) - U_train * dLamb_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * V_train * dLamb_V * Ezg1 * Ezg2) * abc_beta[:,0]
        
        L_g2_beta = (De1_train * f_U * np.exp(-f_U) / (1 - np.exp(-f_U) + 1e-4) + De2_train * (f_V * np.exp(-f_V)- f_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * f_V) * abc_beta[:,1] 
        
        L_lambda_beta = De1_train * f_U * np.exp(-f_U) * abc_beta[:,2] / (1 - np.exp(-f_U) + 1e-4) + De2_train * (abc_beta[:,3] * f_V * np.exp(-f_V)- abc_beta[:,2] * f_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * f_V * abc_beta[:,3]
        
        abc_gamma = LFD_gamma(train_data,Est['g_train'],Est['theta'],Est['C'],curr_m,curr_nodevec,curr_n_layer,n_node=curr_gamma_node,n_lr=curr_gamma_lr,n_epoch=200)
        
        L_gamma = De1_train * Z_train * f_U * np.exp(-f_U) / (1 - np.exp(-f_U) + 1e-4) + De2_train * Z_train * (f_V * np.exp(-f_V)- f_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * Z_train * f_V
        
        L_g1_gamma = (De1_train * U_train * dLamb_U * Ezg1 * Ezg2 * np.exp(-f_U) / (1 - np.exp(-f_U) + 1e-4) + De2_train * Ezg1 * Ezg2 * (V_train * dLamb_V * np.exp(-f_V) - U_train * dLamb_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * V_train * dLamb_V * Ezg1 * Ezg2) * abc_gamma[:,0]
        
        L_g2_gamma = (De1_train * f_U * np.exp(-f_U) / (1 - np.exp(-f_U) + 1e-4) + De2_train * (f_V * np.exp(-f_V)- f_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * f_V) * abc_gamma[:,1] 
        
        L_lambda_gamma = De1_train * f_U * np.exp(-f_U) * abc_gamma[:,2] / (1 - np.exp(-f_U) + 1e-4) + De2_train * (abc_gamma[:,3] * f_V * np.exp(-f_V)- abc_gamma[:,2] * f_U * np.exp(-f_U)) / (np.exp(-f_U) - np.exp(-f_V) + 1e-4) - De3_train * f_V * abc_gamma[:,3]
        
        I_1 = L_beta - L_g1_beta - L_g2_beta - L_lambda_beta
        I_2 = L_gamma - L_g1_gamma - L_g2_gamma - L_lambda_gamma
        
        Info_b = np.zeros((2,2))
        Info_b[0,0] = np.mean(I_1 ** 2)
        Info_b[1,1] = np.mean(I_2 ** 2)
        Info_b[0,1] = np.mean(I_1 * I_2)
        Info_b[1,0] = Info_b[0,1]
        
        try:
            Sigma_b = np.linalg.inv(Info_b)/n
            se_beta = np.sqrt(Sigma_b[0,0])
            se_gamma = np.sqrt(Sigma_b[1,1])
        except np.linalg.LinAlgError:
            se_beta = np.nan
            se_gamma = np.nan
        return se_beta, se_gamma

    # --- DGAHM ---
    Est_D = Est_deep(train_data,valid_data,X_test,X_subject,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
    g1_re_D = np.sqrt(np.mean((Est_D['g_test'][:,0]-np.mean(Est_D['g_test'][:,0])-g1_true)**2)/np.mean(g1_true**2))
    g2_re_D = np.sqrt(np.mean((Est_D['g_test'][:,1]-np.mean(Est_D['g_test'][:,1])-g2_true)**2)/np.mean(g2_true**2))
    se_beta_D, se_gamma_D = compute_se(Est_D, 'D')

    # --- DGAHM-Mid ---
    Est_D_M = Est_deep_median(train_data,valid_data,X_test,X_subject,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
    g1_re_D_M = np.sqrt(np.mean((Est_D_M['g_test'][:,0]-np.mean(Est_D_M['g_test'][:,0])-g1_true)**2)/np.mean(g1_true**2))
    g2_re_D_M = np.sqrt(np.mean((Est_D_M['g_test'][:,1]-np.mean(Est_D_M['g_test'][:,1])-g2_true)**2)/np.mean(g2_true**2))
    se_beta_D_M, se_gamma_D_M = compute_se(Est_D_M, 'D_M')

    # --- GHM ---
    Est_L = Est_linear(train_data,X_test,X_subject,theta_initial,nodevec_L,m_L,c_initial_L)
    g1_re_L = np.sqrt(np.mean((Est_L['g_test'][:,0]-np.mean(Est_L['g_test'][:,0])-g1_true)**2)/np.mean(g1_true**2))
    g2_re_L = np.sqrt(np.mean((Est_L['g_test'][:,1]-np.mean(Est_L['g_test'][:,1])-g2_true)**2)/np.mean(g2_true**2))
    se_beta_L, se_gamma_L = compute_se(Est_L, 'L')

    # Return results
    return {
        # DGAHM
        'g1_subject_D': Est_D['g_subject'][:,0], 'g2_subject_D': Est_D['g_subject'][:,1],
        'g1_test_D': Est_D['g_test'][:,0], 'g2_test_D': Est_D['g_test'][:,1],
        'g1_re_D': g1_re_D, 'g2_re_D': g2_re_D,
        'C_D': Est_D['C'],
        'beta_D': Est_D['theta'][0], 'gamma_D': Est_D['theta'][1],
        'se_beta_D': se_beta_D, 'se_gamma_D': se_gamma_D,
        
        # DGAHM-Mid
        'g1_subject_D_M': Est_D_M['g_subject'][:,0], 'g2_subject_D_M': Est_D_M['g_subject'][:,1],
        'g1_test_D_M': Est_D_M['g_test'][:,0], 'g2_test_D_M': Est_D_M['g_test'][:,1],
        'g1_re_D_M': g1_re_D_M, 'g2_re_D_M': g2_re_D_M,
        'C_D_M': Est_D_M['C'],
        'beta_D_M': Est_D_M['theta'][0], 'gamma_D_M': Est_D_M['theta'][1],
        'se_beta_D_M': se_beta_D_M, 'se_gamma_D_M': se_gamma_D_M,

        # GHM
        'g1_subject_L': Est_L['g_subject'][:,0], 'g2_subject_L': Est_L['g_subject'][:,1],
        'g1_test_L': Est_L['g_test'][:,0], 'g2_test_L': Est_L['g_test'][:,1],
        'g1_re_L': g1_re_L, 'g2_re_L': g2_re_L,
        'C_L': Est_L['C'],
        'beta_L': Est_L['theta'][0], 'gamma_L': Est_L['theta'][1],
        'se_beta_L': se_beta_L, 'se_gamma_L': se_gamma_L
    }




#%% =============================================================================
# PHASE 1: Simulation and Saving
# =============================================================================
if __name__ == "__main__": # Recommended for Windows compatibility
    for i in range(len(Set_n)):
        n = Set_n[i]
        n_layer = Set_layer[i]
        n_node = Set_node[i]
        n_lr = Set_lr[i]
        m = set_m[i]
        nodevec = np.array(np.linspace(0, 95, m+2), dtype="float32")  
        m_L = set_m_L[i]
        nodevec_L = np.array(np.linspace(0, 95, m_L+2), dtype="float32") 
        if i == 0:
            theta_initial = np.array([1.77, 0.95], dtype="float32")
        else:
            theta_initial = np.array([1.8, 0.95], dtype="float32")




        print(f"Starting simulation for n={n} with {Num_Cores} cores...")


        results = Parallel(n_jobs=Num_Cores, verbose=10)(
            delayed(run_single_simulation)(b, n, i, n_layer, n_node, n_lr, m, nodevec, m_L, nodevec_L) for b in range(B)
        )
        
        file_path = os.path.join(result_folder, f'results_n{n}.npy')
        np.save(file_path, results)
        print(f"Saved results to {file_path}")













#%% =============================================================================
# PHASE 2: Loading, Plotting, and Stats
# =============================================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from I_spline import I_S

theta0 = np.array([2, 1], dtype="float32") 

Set_n = np.array([500, 1000])

set_m = np.array([10, 10])

set_m_L = np.array([5, 5])


#--------------------Compute results---------------------------------------
result_folder = 'result_folder'
npy_path = os.path.join(result_folder, 'True_results.npy')

loaded_data = np.load(npy_path, allow_pickle=True)

results_dict_read = loaded_data.item()

t_value = results_dict_read['t_value']
Lambda_Z0_X1 = results_dict_read['Lambda_Z0_X1']
Lambda_Z1_X1 = results_dict_read['Lambda_Z1_X1']
Lambda_Z0_X2 = results_dict_read['Lambda_Z0_X2']
Lambda_Z1_X2 = results_dict_read['Lambda_Z1_X2']


# Survival function Figures Initialization (Global objects)
fig1_Survival = plt.figure()
ax1_1_Survival = fig1_Survival.add_subplot(1, 2, 1)
ax1_1_Survival.set_title(r"Case 2, n=500, $\Lambda_0(t)=t/8$", fontsize=10) 
ax1_1_Survival.set_xlabel("Time",fontsize=8) 
ax1_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax1_1_Survival.tick_params(axis='both',labelsize=6) 
ax1_1_Survival.plot(t_value, np.exp(-Lambda_Z0_X1), color='k', label='True') 
ax1_1_Survival.legend(loc='best', fontsize=6) 

ax1_2_Survival = fig1_Survival.add_subplot(1, 2, 2)
ax1_2_Survival.set_title(r"Case 2, n=1000, $\Lambda_0(t)=t/8$", fontsize=10) 
ax1_2_Survival.set_xlabel("Time",fontsize=8) 
ax1_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax1_2_Survival.tick_params(axis='both',labelsize=6) 
ax1_2_Survival.plot(t_value, np.exp(-Lambda_Z0_X1), color='k', label='True') 
ax1_2_Survival.legend(loc='best', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

fig2_Survival = plt.figure()
ax2_1_Survival = fig2_Survival.add_subplot(1, 2, 1)
ax2_1_Survival.set_title(r"Case 2, n=500, $\Lambda_0(t)=t/8$", fontsize=10) 
ax2_1_Survival.set_xlabel("Time",fontsize=8) 
ax2_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax2_1_Survival.tick_params(axis='both',labelsize=6) 
ax2_1_Survival.plot(t_value, np.exp(-Lambda_Z1_X1), color='k', label='True') 
ax2_1_Survival.legend(loc='best', fontsize=6) 

ax2_2_Survival = fig2_Survival.add_subplot(1, 2, 2)
ax2_2_Survival.set_title(r"Case 2, n=1000, $\Lambda_0(t)=t/8$", fontsize=10) 
ax2_2_Survival.set_xlabel("Time",fontsize=8) 
ax2_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax2_2_Survival.tick_params(axis='both',labelsize=6) 
ax2_2_Survival.plot(t_value, np.exp(-Lambda_Z1_X1), color='k', label='True') 
ax2_2_Survival.legend(loc='best', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

fig3_Survival = plt.figure()
ax3_1_Survival = fig3_Survival.add_subplot(1, 2, 1)
ax3_1_Survival.set_title(r"Case 2, n=500, $\Lambda_0(t)=t/8$", fontsize=10) 
ax3_1_Survival.set_xlabel("Time",fontsize=8) 
ax3_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax3_1_Survival.tick_params(axis='both',labelsize=6)
ax3_1_Survival.plot(t_value, np.exp(-Lambda_Z0_X2), color='k', label='True') 
ax3_1_Survival.legend(loc='best', fontsize=6) 

ax3_2_Survival = fig3_Survival.add_subplot(1, 2, 2)
ax3_2_Survival.set_title(r"Case 2, n=1000, $\Lambda_0(t)=t/8$", fontsize=10) 
ax3_2_Survival.set_xlabel("Time",fontsize=8) 
ax3_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax3_2_Survival.tick_params(axis='both',labelsize=6) 
ax3_2_Survival.plot(t_value, np.exp(-Lambda_Z0_X2), color='k', label='True') 
ax3_2_Survival.legend(loc='best', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

fig4_Survival = plt.figure()
ax4_1_Survival = fig4_Survival.add_subplot(1, 2, 1)
ax4_1_Survival.set_title(r"Case 2, n=500, $\Lambda_0(t)=t/8$", fontsize=10) 
ax4_1_Survival.set_xlabel("Time",fontsize=8) 
ax4_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax4_1_Survival.tick_params(axis='both',labelsize=6) 
ax4_1_Survival.plot(t_value, np.exp(-Lambda_Z1_X2), color='k', label='True') 
ax4_1_Survival.legend(loc='best', fontsize=6) 

ax4_2_Survival = fig4_Survival.add_subplot(1, 2, 2)
ax4_2_Survival.set_title(r"Case 2, n=1000, $\Lambda_0(t)=t/8$", fontsize=10) 
ax4_2_Survival.set_xlabel("Time",fontsize=8) 
ax4_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax4_2_Survival.tick_params(axis='both',labelsize=6) 
ax4_2_Survival.plot(t_value, np.exp(-Lambda_Z1_X2), color='k', label='True') 
ax4_2_Survival.legend(loc='best', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

# Initialize storage for stats
beta_Bias_D = []; beta_Sse_D = []; beta_Ese_D = []; beta_Cp_D = []
gamma_Bias_D = []; gamma_Sse_D = []; gamma_Ese_D = []; gamma_Cp_D = []

beta_Bias_D_M = []; beta_Sse_D_M = []; beta_Ese_D_M = []; beta_Cp_D_M = []
gamma_Bias_D_M = []; gamma_Sse_D_M = []; gamma_Ese_D_M = []; gamma_Cp_D_M = []

beta_Bias_L = []; beta_Sse_L = []; beta_Ese_L = []; beta_Cp_L = []
gamma_Bias_L = []; gamma_Sse_L = []; gamma_Ese_L = []; gamma_Cp_L = []

# Loop to load and process
for i in range(len(Set_n)):
    n = Set_n[i]
    m = set_m[i]
    nodevec = np.array(np.linspace(0, 95, m+2), dtype="float32") 
    m_L = set_m_L[i]
    nodevec_L = np.array(np.linspace(0, 95, m_L+2), dtype="float32") 
    file_path = os.path.join(result_folder, f'results_n{n}.npy')
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        continue
        
    print(f"Loading and processing results for n={n}...")
    # Load results
    results = np.load(file_path, allow_pickle=True)
    
    # --- Unpack Results ---
    g1_subject_D = []; g2_subject_D = []; g1_test_D = []; g2_test_D = []; g1_re_D = []; g2_re_D = []; C_D = []; beta_D = []; gamma_D = []; se_beta_list_D = []; se_gamma_list_D = []
    g1_subject_D_M = []; g2_subject_D_M = []; g1_test_D_M = []; g2_test_D_M = []; g1_re_D_M = []; g2_re_D_M = []; C_D_M = []; beta_D_M = []; gamma_D_M = []; se_beta_list_D_M = []; se_gamma_list_D_M = []
    g1_subject_L = []; g2_subject_L = []; g1_test_L = []; g2_test_L = []; g1_re_L = []; g2_re_L = []; C_L = []; beta_L = []; gamma_L = []; se_beta_list_L = []; se_gamma_list_L = []

    for res in results:
        # D
        g1_subject_D.append(res['g1_subject_D']); g2_subject_D.append(res['g2_subject_D'])
        g1_test_D.append(res['g1_test_D']); g2_test_D.append(res['g2_test_D'])
        g1_re_D.append(res['g1_re_D']); g2_re_D.append(res['g2_re_D'])
        C_D.append(res['C_D'])
        beta_D.append(res['beta_D']); gamma_D.append(res['gamma_D'])
        se_beta_list_D.append(res['se_beta_D']); se_gamma_list_D.append(res['se_gamma_D'])
        
        # D_M
        g1_subject_D_M.append(res['g1_subject_D_M']); g2_subject_D_M.append(res['g2_subject_D_M'])
        g1_test_D_M.append(res['g1_test_D_M']); g2_test_D_M.append(res['g2_test_D_M'])
        g1_re_D_M.append(res['g1_re_D_M']); g2_re_D_M.append(res['g2_re_D_M'])
        C_D_M.append(res['C_D_M'])
        beta_D_M.append(res['beta_D_M']); gamma_D_M.append(res['gamma_D_M'])
        se_beta_list_D_M.append(res['se_beta_D_M']); se_gamma_list_D_M.append(res['se_gamma_D_M'])

        # L
        g1_subject_L.append(res['g1_subject_L']); g2_subject_L.append(res['g2_subject_L'])
        g1_test_L.append(res['g1_test_L']); g2_test_L.append(res['g2_test_L'])
        g1_re_L.append(res['g1_re_L']); g2_re_L.append(res['g2_re_L'])
        C_L.append(res['C_L'])
        beta_L.append(res['beta_L']); gamma_L.append(res['gamma_L'])
        se_beta_list_L.append(res['se_beta_L']); se_gamma_list_L.append(res['se_gamma_L'])

    # --- Process Results (Plotting and Stats) ---
    # Convert lists to arrays for easier handling for plotting
    g1_subject_D = np.array(g1_subject_D); g2_subject_D = np.array(g2_subject_D)
    g1_subject_D_M = np.array(g1_subject_D_M); g2_subject_D_M = np.array(g2_subject_D_M)
    g1_subject_L = np.array(g1_subject_L); g2_subject_L = np.array(g2_subject_L)

    # 1. DGAHM Plotting
    if (i == 0):
        ax1_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(g1_subject_D[:,0])),nodevec) * np.exp(np.mean(g2_subject_D[:,0]))), label='DGAHM', linestyle=':')
        ax1_1_Survival.legend(loc='best', fontsize=6)
        ax2_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(g1_subject_D[:,0])),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(g2_subject_D[:,0]))), label='DGAHM', linestyle=':')
        ax2_1_Survival.legend(loc='best', fontsize=6)
        ax3_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(g1_subject_D[:,1])),nodevec) * np.exp(np.mean(g2_subject_D[:,1]))), label='DGAHM', linestyle=':')
        ax3_1_Survival.legend(loc='best', fontsize=6)
        ax4_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(g1_subject_D[:,1])),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(g2_subject_D[:,1]))), label='DGAHM', linestyle=':')
        ax4_1_Survival.legend(loc='best', fontsize=6)
    else:
        ax1_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(g1_subject_D[:,0])),nodevec) * np.exp(np.mean(g2_subject_D[:,0]))), label='DGAHM', linestyle=':')
        ax1_2_Survival.legend(loc='best', fontsize=6)
        ax2_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(g1_subject_D[:,0])),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(g2_subject_D[:,0]))), label='DGAHM', linestyle=':')
        ax2_2_Survival.legend(loc='best', fontsize=6)
        ax3_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(g1_subject_D[:,1])),nodevec) * np.exp(np.mean(g2_subject_D[:,1]))), label='DGAHM', linestyle=':')
        ax3_2_Survival.legend(loc='best', fontsize=6)
        ax4_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(g1_subject_D[:,1])),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(g2_subject_D[:,1]))), label='DGAHM', linestyle=':')
        ax4_2_Survival.legend(loc='best', fontsize=6)

    # DGAHM Stats
    beta_D_arr = np.array(beta_D); gamma_D_arr = np.array(gamma_D)
    se_beta_D_arr = np.array(se_beta_list_D); se_gamma_D_arr = np.array(se_gamma_list_D)
    beta_Bias_D.append(np.mean(beta_D_arr)-theta0[0]); gamma_Bias_D.append(np.mean(gamma_D_arr)-theta0[1])
    beta_Sse_D.append(np.std(beta_D_arr)); gamma_Sse_D.append(np.std(gamma_D_arr))
    beta_Ese_D.append(np.nanmean(se_beta_D_arr)); gamma_Ese_D.append(np.nanmean(se_gamma_D_arr))
    cp_beta_D = np.mean((beta_D_arr - 1.96 * se_beta_D_arr <= theta0[0]) & (theta0[0] <= beta_D_arr + 1.96 * se_beta_D_arr))
    cp_gamma_D = np.mean((gamma_D_arr - 1.96 * se_gamma_D_arr <= theta0[1]) & (theta0[1] <= gamma_D_arr + 1.96 * se_gamma_D_arr))
    beta_Cp_D.append(cp_beta_D); gamma_Cp_D.append(cp_gamma_D)

    # 2. DGAHM-Mid Plotting
    if (i == 0):
        ax1_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(g1_subject_D_M[:,0])),nodevec) * np.exp(np.mean(g2_subject_D_M[:,0]))), label='DGAHM-Mid', linestyle='-.')
        ax1_1_Survival.legend(loc='best', fontsize=6)
        ax2_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(np.array(beta_D_M)) + np.mean(g1_subject_D_M[:,0])),nodevec) * np.exp(np.mean(np.array(gamma_D_M)) + np.mean(g2_subject_D_M[:,0]))), label='DGAHM-Mid', linestyle='-.')
        ax2_1_Survival.legend(loc='best', fontsize=6)
        ax3_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(g1_subject_D_M[:,1])),nodevec) * np.exp(np.mean(g2_subject_D_M[:,1]))), label='DGAHM-Mid', linestyle='-.')
        ax3_1_Survival.legend(loc='best', fontsize=6)
        ax4_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(np.array(beta_D_M)) + np.mean(g1_subject_D_M[:,1])),nodevec) * np.exp(np.mean(np.array(gamma_D_M)) + np.mean(g2_subject_D_M[:,1]))), label='DGAHM-Mid', linestyle='-.')
        ax4_1_Survival.legend(loc='best', fontsize=6)
    else:
        ax1_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(g1_subject_D_M[:,0])),nodevec) * np.exp(np.mean(g2_subject_D_M[:,0]))), label='DGAHM-Mid', linestyle='-.')
        ax1_2_Survival.legend(loc='best', fontsize=6)
        ax2_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(np.array(beta_D_M)) + np.mean(g1_subject_D_M[:,0])),nodevec) * np.exp(np.mean(np.array(gamma_D_M)) + np.mean(g2_subject_D_M[:,0]))), label='DGAHM-Mid', linestyle='-.')
        ax2_2_Survival.legend(loc='best', fontsize=6)
        ax3_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(g1_subject_D_M[:,1])),nodevec) * np.exp(np.mean(g2_subject_D_M[:,1]))), label='DGAHM-Mid', linestyle='-.')
        ax3_2_Survival.legend(loc='best', fontsize=6)
        ax4_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D_M), axis=0), t_value*np.exp(np.mean(np.array(beta_D_M)) + np.mean(g1_subject_D_M[:,1])),nodevec) * np.exp(np.mean(np.array(gamma_D_M)) + np.mean(g2_subject_D_M[:,1]))), label='DGAHM-Mid', linestyle='-.')
        ax4_2_Survival.legend(loc='best', fontsize=6)

    # DGAHM-Mid Stats
    beta_D_M_arr = np.array(beta_D_M); gamma_D_M_arr = np.array(gamma_D_M)
    se_beta_D_M_arr = np.array(se_beta_list_D_M); se_gamma_D_M_arr = np.array(se_gamma_list_D_M)
    beta_Bias_D_M.append(np.mean(beta_D_M_arr)-theta0[0]); gamma_Bias_D_M.append(np.mean(gamma_D_M_arr)-theta0[1])
    beta_Sse_D_M.append(np.std(beta_D_M_arr)); gamma_Sse_D_M.append(np.std(gamma_D_M_arr))
    beta_Ese_D_M.append(np.nanmean(se_beta_D_M_arr)); gamma_Ese_D_M.append(np.nanmean(se_gamma_D_M_arr))
    cp_beta_D_M = np.mean((beta_D_M_arr - 1.96 * se_beta_D_M_arr <= theta0[0]) & (theta0[0] <= beta_D_M_arr + 1.96 * se_beta_D_M_arr))
    cp_gamma_D_M = np.mean((gamma_D_M_arr - 1.96 * se_gamma_D_M_arr <= theta0[1]) & (theta0[1] <= gamma_D_M_arr + 1.96 * se_gamma_D_M_arr))
    beta_Cp_D_M.append(cp_beta_D_M); gamma_Cp_D_M.append(cp_gamma_D_M)

    # 3. GHM Plotting
    if (i == 0):
        ax1_1_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(g1_subject_L[:,0])),nodevec_L) * np.exp(np.mean(g2_subject_L[:,0]))), label='GHM', linestyle='--')
        ax1_1_Survival.legend(loc='best', fontsize=6)
        ax2_1_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(g1_subject_L[:,0])),nodevec_L) * np.exp(np.mean(np.array(gamma_L)) + np.mean(g2_subject_L[:,0]))), label='GHM', linestyle='--')
        ax2_1_Survival.legend(loc='best', fontsize=6)
        ax3_1_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(g1_subject_L[:,1])),nodevec_L) * np.exp(np.mean(g2_subject_L[:,1]))), label='GHM', linestyle='--')
        ax3_1_Survival.legend(loc='best', fontsize=6)
        ax4_1_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(g1_subject_L[:,1])),nodevec_L) * np.exp(np.mean(np.array(gamma_L)) + np.mean(g2_subject_L[:,1]))), label='GHM', linestyle='--')
        ax4_1_Survival.legend(loc='best', fontsize=6)
    else:
        ax1_2_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(g1_subject_L[:,0])),nodevec_L) * np.exp(np.mean(g2_subject_L[:,0]))), label='GHM', linestyle='--')
        ax1_2_Survival.legend(loc='best', fontsize=6)
        ax2_2_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(g1_subject_L[:,0])),nodevec_L) * np.exp(np.mean(np.array(gamma_L)) + np.mean(g2_subject_L[:,0]))), label='GHM', linestyle='--')
        ax2_2_Survival.legend(loc='best', fontsize=6)
        ax3_2_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(g1_subject_L[:,1])),nodevec_L) * np.exp(np.mean(g2_subject_L[:,1]))), label='GHM', linestyle='--')
        ax3_2_Survival.legend(loc='best', fontsize=6)
        ax4_2_Survival.plot(t_value, np.exp(-I_S(m_L,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(g1_subject_L[:,1])),nodevec_L) * np.exp(np.mean(np.array(gamma_L)) + np.mean(g2_subject_L[:,1]))), label='GHM', linestyle='--')
        ax4_2_Survival.legend(loc='best', fontsize=6)
    
    # EHM Stats
    beta_L_arr = np.clip(np.array(beta_L),0,10); gamma_L_arr = np.clip(np.array(gamma_L),0,10)
    se_beta_L_arr = np.array(se_beta_list_L); se_gamma_L_arr = np.array(se_gamma_list_L)
    beta_Bias_L.append(np.mean(beta_L_arr)-theta0[0]); gamma_Bias_L.append(np.mean(gamma_L_arr)-theta0[1])
    beta_Sse_L.append(np.std(beta_L_arr)); gamma_Sse_L.append(np.std(gamma_L_arr))
    beta_Ese_L.append(np.nanmean(se_beta_L_arr)); gamma_Ese_L.append(np.nanmean(se_gamma_L_arr))
    cp_beta_L = np.mean((beta_L_arr - 1.96 * se_beta_L_arr <= theta0[0]) & (theta0[0] <= beta_L_arr + 1.96 * se_beta_L_arr))
    cp_gamma_L = np.mean((gamma_L_arr - 1.96 * se_gamma_L_arr <= theta0[1]) & (theta0[1] <= gamma_L_arr + 1.96 * se_gamma_L_arr))
    beta_Cp_L.append(cp_beta_L); gamma_Cp_L.append(cp_gamma_L)

#%% -----------Save all results------------
# ================Figures======================
fig1_Survival.savefig(os.path.join(result_folder, 'fig_Survival_Z0_X1_case2_1.jpeg'), dpi=600, bbox_inches='tight')
fig2_Survival.savefig(os.path.join(result_folder, 'fig_Survival_Z1_X1_case2_1.jpeg'), dpi=600, bbox_inches='tight')
fig3_Survival.savefig(os.path.join(result_folder, 'fig_Survival_Z0_X2_case2_1.jpeg'), dpi=600, bbox_inches='tight')
fig4_Survival.savefig(os.path.join(result_folder, 'fig_Survival_Z1_X2_case2_1.jpeg'), dpi=600, bbox_inches='tight')




# =================Tables=======================

dic_error_beta = {
    "n": Set_n, 
    "beta_Bias_D": np.array(beta_Bias_D), 
    "beta_Sse_D": np.array(beta_Sse_D), 
    "beta_Ese_D": np.array(beta_Ese_D), 
    "beta_Cp_D": np.array(beta_Cp_D), 
    "beta_Bias_D_M": np.array(beta_Bias_D_M), 
    "beta_Sse_D_M": np.array(beta_Sse_D_M), 
    "beta_Ese_D_M": np.array(beta_Ese_D_M), 
    "beta_Cp_D_M": np.array(beta_Cp_D_M), 
    "beta_Bias_L": np.array(beta_Bias_L), 
    "beta_Sse_L": np.array(beta_Sse_L), 
    "beta_Ese_L": np.array(beta_Ese_L), 
    "beta_Cp_L": np.array(beta_Cp_L),
}

dic_error_gamma = {
    "n": Set_n, 
    "gamma_Bias_D": np.array(gamma_Bias_D), 
    "gamma_Sse_D": np.array(gamma_Sse_D), 
    "gamma_Ese_D": np.array(gamma_Ese_D), 
    "gamma_Cp_D": np.array(gamma_Cp_D), 
    "gamma_Bias_D_M": np.array(gamma_Bias_D_M), 
    "gamma_Sse_D_M": np.array(gamma_Sse_D_M), 
    "gamma_Ese_D_M": np.array(gamma_Ese_D_M), 
    "gamma_Cp_D_M": np.array(gamma_Cp_D_M), 
    "gamma_Bias_L": np.array(gamma_Bias_L), 
    "gamma_Sse_L": np.array(gamma_Sse_L), 
    "gamma_Ese_L": np.array(gamma_Ese_L), 
    "gamma_Cp_L": np.array(gamma_Cp_L)
}

# ====================================

df_beta = pd.DataFrame.from_dict(dic_error_beta, orient='index').T

output_path_beta = os.path.join(result_folder, 'beta_error_case2.csv')

df_beta.to_csv(output_path_beta, index=False)

df_gamma = pd.DataFrame.from_dict(dic_error_gamma, orient='index').T

output_path_gamma = os.path.join(result_folder, 'gamma_error_case2.csv')

df_gamma.to_csv(output_path_gamma, index=False)

