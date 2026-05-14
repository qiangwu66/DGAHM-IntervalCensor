
# %% -------------import packages--------------
import torch
import numpy as np
from torch import nn
from I_spline import I_U
import copy 

#%% --------------------------
def g_D(train_data, valid_data, X_test, X_subject, theta, C, m, nodevec, n_layer, n_node, n_lr, n_epoch):
    # --- ---
    Z_train = torch.Tensor(train_data['Z'])
    X_train = torch.Tensor(train_data['X'])
    U_train = torch.Tensor(train_data['U'])
    V_train = torch.Tensor(train_data['V'])
    De1_train = torch.Tensor(train_data['De1'])
    De2_train = torch.Tensor(train_data['De2'])
    De3_train = torch.Tensor(train_data['De3'])

    # ------
    Z_val = torch.Tensor(valid_data['Z'])
    X_val = torch.Tensor(valid_data['X'])
    U_val = torch.Tensor(valid_data['U'])
    V_val = torch.Tensor(valid_data['V'])
    De1_val = torch.Tensor(valid_data['De1'])
    De2_val = torch.Tensor(valid_data['De2'])
    De3_val = torch.Tensor(valid_data['De3'])

    # ------
    X_test = torch.Tensor(X_test)
    X_subject = torch.Tensor(X_subject)
    theta = torch.Tensor(theta)
    C = torch.Tensor(C)
    d = X_train.size()[1]

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
        Iu = torch.Tensor(I_U(m, U.detach().numpy() * (torch.exp(Z * theta[0] + g_X[:,0])).detach().numpy(), nodevec))
        Iv = torch.Tensor(I_U(m, V.detach().numpy() * (torch.exp(Z * theta[0] + g_X[:,0])).detach().numpy(), nodevec))
        Ezg = torch.exp(Z * theta[1] + g_X[:,1])
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- torch.matmul(Iu,C) * Ezg) + 1e-4) + De2 * torch.log(torch.exp(- torch.matmul(Iu,C) * Ezg) - torch.exp(- torch.matmul(Iv,C) * Ezg) + 1e-4) - De3 * torch.matmul(Iv,C) * Ezg)
        return loss_fun


    # ------
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
        g_test = model(X_test)
        g_subject = model(X_subject)
    
    g_train = g_train.detach().numpy()
    g_test = g_test.detach().numpy()
    g_subject = g_subject.detach().numpy()


    return {
        'g_train': g_train - np.mean(g_train, axis=0),
        'g_test': g_test,
        'g_subject': g_subject
    }


