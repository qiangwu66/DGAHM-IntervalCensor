#%% ----------- import packages -----------
import numpy as np
import random
import torch
import pandas as pd
from joblib import Parallel, delayed

from data_generator import generate_case_2
from iteration_deep import Est_deep
from iteration_deep_Semi import Est_deep_semi
from iteration_deep_median import Est_deep_median
from iteration_NN_IC import Est_NN_IC
from Bernstein_Poly import Bern_S

#%% ---------- define seeds -------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%% ---------- set base seed -------------
set_seed(1)

#%% -----------------------
def Indicator_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    I_M = (a[:, np.newaxis] < b).astype(int)
    return I_M

# ---------------- Config ----------------
tau = 5
p = 3
Set_n = np.array([500, 1000])
corr = 0.5
n_layer = 2
Set_node = np.array([50, 50])
n_epoch = 500
Set_lr = np.array([3e-4, 2e-4])

L_1_penalty = 0.5  
N_node = 50
N_epoch = 1000
N_lr = 0.01  
m_n = 3 

def generate_increasing_random_numbers(n):
    numbers = []
    prev_number = 0.0
    for _ in range(n):
        number = np.random.uniform(prev_number, 1)
        numbers.append(number)
        prev_number = number
    return numbers

b0 = np.array(generate_increasing_random_numbers(m_n+1), dtype="float32") # 生成[0,1]之间的单调递增随机数m_n+1个


B = 200
# test data (fixed across runs)
test_data = generate_case_2(500, corr, tau)
X_test = test_data['X']
g1_test = test_data['g1_X']
g2_test = test_data['g2_X']
T_test = test_data['T']
dim_x = X_test.shape[0]

t_nodes = np.array(np.linspace(tau/500, tau, 501), dtype="float32")
I_t_T = Indicator_matrix(t_nodes, T_test)  # 501 * 500
S0_t_X_true = np.exp(- np.log(np.outer(t_nodes, np.exp(g1_test)) + 1) * np.exp(g2_test))


# spline basis config
m = 5  # number of interior knots of integrated spline basis
nodevec = np.array(np.linspace(0, 70, m + 2), dtype="float32")
c_initial = np.array(0.1 * np.ones(m + p), dtype="float32")
theta_initial = np.array([0, 0], dtype="float32")


# ---------------- containers ----------------
MSPE_DNN = np.zeros(2)
MSPE_sd_DNN = np.zeros(2)

MSPE_DNN_semi = np.zeros(2)
MSPE_sd_DNN_semi = np.zeros(2)

MSPE_DNN_median = np.zeros(2)
MSPE_sd_DNN_median = np.zeros(2)


MSPE_NN_IC = np.zeros(2)
MSPE_sd_NN_IC = np.zeros(2)


# -------- function for one repetition b (to be parallelized) --------
def one_replication(b,
                    n,
                    corr,
                    tau,
                    X_test,
                    t_nodes,
                    n_layer,
                    n_node,
                    n_lr,
                    n_epoch,
                    nodevec,
                    m,
                    c_initial,
                    I_t_T,
                    S0_t_X_true,
                    valid_n,
                    base_seed):
    # each process sets its own seed
    set_seed(base_seed + b)

    # Generate training data
    train_data = generate_case_2(n, corr, tau)
    # valid data for early stopping/model selection
    valid_data = generate_case_2(valid_n, corr, tau)

    # DGAHM-Non
    Est_D = Est_deep(train_data, valid_data, X_test, t_nodes,
                     n_layer, n_node, n_lr, n_epoch, nodevec, m, c_initial)
    S_t_X_DNN = Est_D['S_T_X_test']  # 501 * 500
    mspe_dnn = float(np.mean(I_t_T * (S_t_X_DNN - S0_t_X_true) ** 2))

    # DGAHM-Semi
    Est_D_semi = Est_deep_semi(train_data, valid_data, X_test, t_nodes, theta_initial,
                     n_layer, n_node, n_lr, n_epoch, nodevec, m, c_initial)
    S_t_X_DNN_semi = Est_D_semi['S_T_X_test']  # 501 * 500
    mspe_dnn_semi = float(np.mean(I_t_T * (S_t_X_DNN_semi - S0_t_X_true) ** 2))

    # DGAHM-Mid (median loss)
    Est_D_median = Est_deep_median(train_data, valid_data, X_test, t_nodes,
                                   n_layer, n_node, n_lr, n_epoch, nodevec, m, c_initial)
    S_t_X_DNN_median = Est_D_median['S_T_X_test']  # 501 * 500
    mspe_dnn_median = float(np.mean(I_t_T * (S_t_X_DNN_median - S0_t_X_true) ** 2))


    # NN-IC
    est_NN_IC = Est_NN_IC(train_data, valid_data, X_test, L_1_penalty, N_node, N_lr, N_epoch, m_n, b0, tau)
    Lambda_NN_IC = np.dot(Bern_S(m_n, t_nodes, 0, tau), np.array(est_NN_IC['B1']))
    S_NN_IC = np.exp(-np.dot(np.exp(np.reshape(est_NN_IC['g_test'], (len(est_NN_IC['g_test']), 1))), np.reshape(Lambda_NN_IC, (1, len(Lambda_NN_IC))))) # 500*501
    mspe_NN_IC = float(np.mean(I_t_T * (S_NN_IC.T - S0_t_X_true) ** 2))
    
    return mspe_dnn, mspe_dnn_semi, mspe_dnn_median, mspe_NN_IC

# ---------------- main loop over n ----------------
for i in range(len(Set_n)):
    n = int(Set_n[i])
    n_node = int(Set_node[i])
    n_lr = float(Set_lr[i])

    # Run B repetitions in parallel
    results = Parallel(n_jobs=4, backend="loky", verbose=0)(
        delayed(one_replication)(
            b=b,
            n=n,
            corr=corr,
            tau=tau,
            X_test=X_test,
            t_nodes=t_nodes,
            n_layer=n_layer,
            n_node=n_node,
            n_lr=n_lr,
            n_epoch=n_epoch,
            nodevec=nodevec,
            m=m,
            c_initial=c_initial,
            I_t_T=I_t_T,
            S0_t_X_true=S0_t_X_true,
            valid_n=200,
            base_seed=12
        ) for b in range(B)
    )

    # Unpack results
    MSPE_DNN_B = [r[0] for r in results]
    MSPE_DNN_semi_B = [r[1] for r in results]
    MSPE_DNN_median_B = [r[2] for r in results]
    MSPE_NN_IC_B = [r[3] for r in results]

    # Aggregate
    MSPE_DNN[i] = np.mean(MSPE_DNN_B)
    MSPE_sd_DNN[i] = np.sqrt(np.nanvar(MSPE_DNN_B, ddof=1))
    MSPE_DNN_semi[i] = np.mean(MSPE_DNN_semi_B)
    MSPE_sd_DNN_semi[i] = np.sqrt(np.nanvar(MSPE_DNN_semi_B, ddof=1))
    MSPE_DNN_median[i] = np.mean(MSPE_DNN_median_B)
    MSPE_sd_DNN_median[i] = np.sqrt(np.nanvar(MSPE_DNN_median_B, ddof=1))
    MSPE_NN_IC[i] = np.mean(MSPE_NN_IC_B)
    MSPE_sd_NN_IC[i] = np.sqrt(np.nanvar(MSPE_NN_IC_B, ddof=1))


# =================Tables=======================
dic_MSPE = {
    "n": Set_n,
    "MSPE_DNN": MSPE_DNN,
    "MSPE_sd_DNN": MSPE_sd_DNN,
    "MSPE_DNN_semi": MSPE_DNN_semi,
    "MSPE_sd_DNN_semi": MSPE_sd_DNN_semi,
    "MSPE_DNN_median": MSPE_DNN_median,
    "MSPE_sd_DNN_median": MSPE_sd_DNN_median,
    "MSPE_NN_IC": MSPE_NN_IC,
    "MSPE_sd_NN_IC": MSPE_sd_NN_IC,
}
result_MSPE = pd.DataFrame(dic_MSPE)
result_MSPE.to_csv('MSPE_Case2.csv', index=False)