import numpy as np

def indicate_T_t_matrix(t, S_t_W, S_U_W, S_V_W,
                        U_R_test, V_R_test, De1_R_test, De2_R_test, De3_R_test):
    m = len(t)
    n = len(U_R_test)
    out = np.zeros((m, n), dtype=float)

    for j in range(m):
        res = np.zeros(n, dtype=float)
  
        for i in range(n):
            if De1_R_test[i] == 1:
                if U_R_test[i] < t[j]:
                    res[i] = 0.0
                else:
                    res[i] = (S_t_W[j, i] - S_U_W[i]) / (1.0 - S_U_W[i])
            if De2_R_test[i] == 1:
                if t[j] < U_R_test[i]:
                    res[i] = 1.0
                elif t[j] >= V_R_test[i]:
                    res[i] = 0.0
                else:
                    res[i] = (S_t_W[j, i] - S_V_W[i]) / (S_U_W[i] - S_V_W[i])
            if De3_R_test[i] == 1:
                if t[j] < V_R_test[i]:
                    res[i] = 1.0
                else:
                    res[i] = S_t_W[j, i] / S_V_W[i]
        out[j] = res
    return out
