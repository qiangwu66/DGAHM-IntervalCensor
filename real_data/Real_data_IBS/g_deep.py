# %% -------------import packages--------------
import torch
from torch import nn
from I_spline import I_U
import copy
import numpy as np
#%% --------------------------
def g_D(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_val,X_val,U_val,V_val,De1_val,De2_val,De3_val,X_test,X_sort1,X_sort0,theta,n_layer,n_node,n_lr,n_epoch,nodevec,m,C):
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    U_train = torch.Tensor(U_train)
    V_train = torch.Tensor(V_train)
    De1_train = torch.Tensor(De1_train)
    De2_train = torch.Tensor(De2_train)
    De3_train = torch.Tensor(De3_train)
    Z_val = torch.Tensor(Z_val)
    X_val = torch.Tensor(X_val)
    U_val = torch.Tensor(U_val)
    V_val = torch.Tensor(V_val)
    De1_val = torch.Tensor(De1_val)
    De2_val = torch.Tensor(De2_val)
    De3_val = torch.Tensor(De3_val)

    X_test = torch.Tensor(X_test)
    X_sort1 = torch.Tensor(X_sort1)
    X_sort0 = torch.Tensor(X_sort0)
    theta = torch.Tensor(theta)
    C = torch.Tensor(C)
    d = X_train.size()[1]
    # ----------------------------
    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(d, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 2))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred




    # ---------------------------
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)

    def my_loss(De1, De2, De3, Z, theta, C, m, U, V, nodevec, g_X):
        Iu = torch.Tensor(I_U(m, U.detach().numpy() * (torch.exp(torch.matmul(Z, theta[0:4]) + g_X[:,0])).detach().numpy(), nodevec))
        Iv = torch.Tensor(I_U(m, V.detach().numpy() * (torch.exp(torch.matmul(Z, theta[0:4]) + g_X[:,0])).detach().numpy(), nodevec))
        Ezg = torch.exp(torch.matmul(Z, theta[4:8]) + g_X[:,1])
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- torch.matmul(Iu,C) * Ezg) + 1e-4) + De2 * torch.log(torch.exp(- torch.matmul(Iu,C) * Ezg) - torch.exp(- torch.matmul(Iv,C) * Ezg) + 1e-4) - De3 * torch.matmul(Iv,C) * Ezg)
        return loss_fun


    patience = 10
    counter = 0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict()) 

    for epoch in range(n_epoch):
        # --- Training Step ---
        model.train() 
        pred_g_X_train = model(X_train)
        loss_train = my_loss(De1_train, De2_train, De3_train, Z_train, theta, C, m, U_train, V_train, nodevec, pred_g_X_train)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # --- Validation Step ---
        model.eval() 
        with torch.no_grad(): 
            pred_g_X_val = model(X_val)
            loss_val = my_loss(De1_val, De2_val, De3_val, Z_val, theta, C, m, U_val, V_val, nodevec, pred_g_X_val)

        # --- Early Stopping Logic ---
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model_wts = copy.deepcopy(model.state_dict()) 
            counter = 0 
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}')
                break
        

    # %% ----------------------
    model.load_state_dict(best_model_wts)
    model.eval() 

    with torch.no_grad():
        g_train = model(X_train)
        g_test1 = model(X_sort1)
        g_test0 = model(X_sort0)
        g_test = model(X_test)
        g_train = g_train.detach().numpy()
        g_test1 = g_test1.detach().numpy()
        g_test0 = g_test0.detach().numpy()
        g_test = g_test.detach().numpy()

    return {
        'g_train': g_train,
        'g_test1': g_test1,
        'g_test0': g_test0,
        'g_test': g_test
    }


