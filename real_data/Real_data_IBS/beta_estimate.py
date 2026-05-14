

import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def beta_est(gamma, De1, De2, De3, Z, U, V, C, m, g_X, nodevec,
             eps=1e-8, exp_cap=50.0):
    """
    稳健版本：通过裁剪防止 log(0)、负数取对数与溢出。
    
    参数
    ----
    eps : float
        对数与概率下限，避免 log(0)。
    exp_cap : float
        对指数输入的截断上界，避免溢出。
    """

    # 预计算与形状检查
    Z = np.asarray(Z)
    g_X = np.asarray(g_X)
    U = np.asarray(U)
    V = np.asarray(V)
    C = np.asarray(C)
    De1 = np.asarray(De1)
    De2 = np.asarray(De2)
    De3 = np.asarray(De3)

    # 安全的 exp 函数：对输入截断到 [-exp_cap, exp_cap]
    def safe_exp(x):
        return np.exp(np.clip(x, -exp_cap, exp_cap))

    def BF(b):
        # 线性预测
        Zb = Z @ b
        Zb = np.clip(Zb, -exp_cap, exp_cap)

        # g_X 的两列
        gx0 = np.clip(g_X[:, 0], -exp_cap, exp_cap)
        gx1 = np.clip(g_X[:, 1], -exp_cap, exp_cap)

        # 缩放后的 U, V
        scale = safe_exp(Zb + gx0)
        U_scaled = U * scale
        V_scaled = V * scale

        # I-spline 评估
        Iu = I_U(m, U_scaled, nodevec)   # shape: (n, k)
        Iv = I_U(m, V_scaled, nodevec)   # shape: (n, k)

        # Ezg
        Ezg = safe_exp(Z @ gamma + gx1)  # shape: (n,)

        # 计算 lambda_u = (Iu @ C) * Ezg, lambda_v = (Iv @ C) * Ezg
        # 先计算基函数线性组合，裁剪到非负（若模型设定期望为非负）
        IuC = Iu @ C
        IvC = Iv @ C
        # 若理论上应非负，则裁剪
        IuC = np.clip(IuC, 0.0, None)
        IvC = np.clip(IvC, 0.0, None)

        lam_u = IuC * Ezg
        lam_v = IvC * Ezg

        # 为稳定性，限制 lambda 的大小，避免 exp(-lambda) 下溢出到 0
        lam_u = np.clip(lam_u, 0.0, np.float64(exp_cap))  # exp_cap 即上限
        lam_v = np.clip(lam_v, 0.0, np.float64(exp_cap))

        # 生存函数 S(t) = exp(-lambda)
        Su = np.exp(-lam_u)
        Sv = np.exp(-lam_v)

        # 三类项
        # term1: log(1 - Su)
        p1 = 1.0 - Su
        p1 = np.clip(p1, eps, 1.0)  # 防止 0
        term1 = De1 * np.log(p1)

        # term2: log(Su - Sv)
        diff = Su - Sv
        # 模型上应当 U <= V，Su >= Sv；若数值误差导致非正，裁剪到 eps
        diff = np.clip(diff, eps, 1.0)
        term2 = De2 * np.log(diff)

        # term3: -De3 * (IvC * Ezg)
        term3 = -De3 * lam_v  # 因为 lam_v = IvC * Ezg

        # 总损失（负对数似然）
        Loss_F = -(np.mean(term1 + term2 + term3))

        # 若出现非数，返回大惩罚值
        if not np.isfinite(Loss_F):
            return 1e50
        return Loss_F

    x0 = np.zeros(Z.shape[1], dtype=float)

    result = spo.minimize(
        BF,
        x0,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    )

    return result['x']
