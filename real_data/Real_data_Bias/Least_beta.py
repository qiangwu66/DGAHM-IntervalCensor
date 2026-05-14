import torch
from torch import nn
import numpy as np
from I_spline import I_U
from B_spline2 import B_S2
#%% --------------------------
def LFD_beta(Zi_train,Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,g_train,theta,C,m,nodevec,n_layer,n_node,n_lr,n_epoch):
    Zi_train = torch.Tensor(Zi_train)
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    U_train = torch.Tensor(U_train)
    V_train = torch.Tensor(V_train)
    De1_train = torch.Tensor(De1_train)
    De2_train = torch.Tensor(De2_train)
    De3_train = torch.Tensor(De3_train)
    X_U = torch.Tensor(np.c_[X_train, Z_train, U_train, V_train])
    g_train = torch.Tensor(g_train)
    theta = torch.Tensor(theta)
    C = torch.Tensor(C)
    c = Z_train.size()[1]
    d = X_U.size()[1]
    # ----------------------------
    class DNNAB(torch.nn.Module):
        def __init__(self):
            super(DNNAB, self).__init__()
            layers = []
            layers.append(nn.Linear(d, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 4))
            self.model = nn.Sequential(*layers)
            
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred


    # ---------------------------
    model = DNNAB()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def Loss(De1,De2,De3,Zi,Z,U,V,g_X,theta,C,m,nodevec,a_b_c):
        Ezg1 = torch.exp(torch.matmul(Z, theta[0:c]) + g_X[:,0])
        Ezg2 = torch.exp(torch.matmul(Z, theta[c:(2*c)]) + g_X[:,1])
        # compute \Lambda()
        Iu = torch.Tensor(I_U(m, U.detach().numpy() * Ezg1.detach().numpy(), nodevec))
        Iv = torch.Tensor(I_U(m, V.detach().numpy() * Ezg1.detach().numpy(), nodevec))
        Lamb_U = torch.matmul(Iu,C)
        Lamb_V = torch.matmul(Iv,C)
        f_U = Lamb_U * Ezg2
        f_V = Lamb_V * Ezg2
        # compute \Lambda'()
        Bu = torch.Tensor(B_S2(m, U.detach().numpy() * Ezg1.detach().numpy(), nodevec))
        Bv = torch.Tensor(B_S2(m, V.detach().numpy() * Ezg1.detach().numpy(), nodevec))
        dLamb_U = torch.matmul(Bu,C)
        dLamb_V = torch.matmul(Bv,C) 
        #compute partial derivatives
        L_beta =  De1 * U * Zi * Ezg1 * Ezg2 * torch.exp(-f_U) * dLamb_U / (1 - torch.exp(-f_U) + 1e-4) + De2 * Zi * Ezg1 * Ezg2 * (V * dLamb_V * torch.exp(-f_V) - U * dLamb_U * torch.exp(-f_U)) / (torch.exp(-f_U) - torch.exp(-f_V) + 1e-4) - De3 * V * Zi * dLamb_V * Ezg1 * Ezg2
        
        L_g1 = (De1 * U * dLamb_U * Ezg1 * Ezg2 * torch.exp(-f_U) / (1 - torch.exp(-f_U) + 1e-4) + De2 * Ezg1 * Ezg2 * (V * dLamb_V * torch.exp(-f_V) - U * dLamb_U * torch.exp(-f_U)) / (torch.exp(-f_U) - torch.exp(-f_V) + 1e-4) - De3 * V * dLamb_V * Ezg1 * Ezg2) * a_b_c[:,0]
        
        L_g2 = (De1 * f_U * torch.exp(-f_U) / (1 - torch.exp(-f_U) + 1e-4) + De2 * (f_V * torch.exp(-f_V)- f_U * torch.exp(-f_U)) / (torch.exp(-f_U) - torch.exp(-f_V) + 1e-4) - De3 * f_V) * a_b_c[:,1] 
        
        L_lambda = De1 * f_U * torch.exp(-f_U) * a_b_c[:,2] / (1 - torch.exp(-f_U) + 1e-4) + De2 * (a_b_c[:,3] * f_V * torch.exp(-f_V)- a_b_c[:,2] * f_U * torch.exp(-f_U)) / (torch.exp(-f_U) - torch.exp(-f_V) + 1e-4) - De3 * f_V * a_b_c[:,3]
        
        Loss_f = torch.mean((L_beta - L_g1 - L_g2 - L_lambda) ** 2)
        return Loss_f


    # -----------------------------
    for epoch in range(n_epoch):
        pred_abc = model(X_U) 
        loss = Loss(De1_train,De2_train,De3_train,Zi_train,Z_train,U_train,V_train,g_train,theta,C,m,nodevec,pred_abc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#%% -----------------------
    abc_beta = model(X_U)
    abc_beta = abc_beta.detach().numpy()
    return abc_beta