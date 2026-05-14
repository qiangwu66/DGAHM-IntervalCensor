# %% -------------import packages--------------
import torch
from torch import nn
from I_spline import I_U
import numpy as np
from B_spline2 import B_S2


#%% --------------------------
def g_D_median(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Z_valid,X_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,theta,C,m,nodevec,n_layer,n_node,n_lr,n_epoch):
    device = C.device if isinstance(C, torch.Tensor) else torch.device('cpu')
    dtype = C.dtype if isinstance(C, torch.Tensor) else torch.float32
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C, dtype=dtype, device=device)

    # to tensor with device/dtype
    def to_t(x):
        return torch.as_tensor(x, dtype=dtype, device=device)
    
    Z_train = to_t(Z_train)
    X_train = to_t(X_train)
    U_train = to_t(U_train)
    V_train = to_t(V_train)
    De1_train = to_t(De1_train)
    De2_train = to_t(De2_train)
    De3_train = to_t(De3_train)
    Z_valid = to_t(Z_valid)
    X_valid = to_t(X_valid)
    U_valid = to_t(U_valid)
    V_valid = to_t(V_valid)
    De1_valid = to_t(De1_valid)
    De2_valid = to_t(De2_valid)
    De3_valid = to_t(De3_valid)
    theta = to_t(theta)
    C = to_t(C)

    d = X_train.size(1)

    class DNNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            layers.append(nn.Linear(d, n_node))
            layers.append(nn.ReLU())
            for _ in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 2))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)

    model = DNNModel().to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)

    def I_U_torch(m_, z_t, nodevec_):
        z_np = z_t.detach().cpu().numpy()
        out_np = I_U(m_, z_np, nodevec_)
        return torch.as_tensor(out_np, dtype=dtype, device=device)
    
    def B_U_torch(m_, z_t, nodevec_):
        z_np = z_t.detach().cpu().numpy()
        out_np = B_S2(m_, z_np, nodevec_) 
        return torch.as_tensor(out_np, dtype=dtype, device=device)

    def my_loss(De1, De2, De3, Z, theta, C, m, U, V, nodevec, g_X):

        g1 = torch.clamp(g_X[:, 0], min=-50, max=50)
        g2 = torch.clamp(g_X[:, 1], min=-50, max=50)

        scale = torch.exp(torch.matmul(Z, theta[0:4]) + g1) 
        Ezg  = torch.exp(torch.matmul(Z, theta[4:8]) + g2)  

        # Iu = I_U_torch(m, U * scale, nodevec)
        Iv = I_U_torch(m, V * scale, nodevec)
        Iu_2 = I_U_torch(m, U * scale / 2, nodevec) 
        Iuv_2 = I_U_torch(m, (U + V) * scale / 2, nodevec) 
        B_U_2 = B_U_torch(m, U * scale / 2, nodevec)
        B_U_V_2 = B_U_torch(m, (U + V) * scale / 2, nodevec)

        b = torch.matmul(Iv, C) * Ezg

        c1 = torch.matmul(B_U_2, C)
        d1 = torch.matmul(Iu_2, C) * Ezg

        c2 = torch.matmul(B_U_V_2, C)
        d2 = torch.matmul(Iuv_2, C) * Ezg

        eps = torch.finfo(dtype).eps 

        term1 = torch.log(torch.clamp(c1, min=eps)) + torch.matmul(Z, theta[0:4]) + torch.matmul(Z, theta[4:8]) + g1 + g2 - d1

        term2 = torch.log(torch.clamp(c2, min=eps)) + torch.matmul(Z, theta[0:4]) + torch.matmul(Z, theta[4:8]) + g1 + g2 - d2

        term3 = - b

        loss_vec = -(De1 * term1 + De2 * term2 + De3 * term3)
        return torch.mean(loss_vec)

    # ---------------------------
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epoch):
        model.train()
        pred_g_X = model(X_train)
        loss_train = my_loss(De1_train, De2_train, De3_train, Z_train, theta, C, m, U_train, V_train, nodevec, pred_g_X)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_g_X_valid = model(X_valid)
            loss_valid = my_loss(De1_valid, De2_valid, De3_valid, Z_valid, theta, C, m, U_valid, V_valid, nodevec, pred_g_X_valid)

        if loss_valid < best_val_loss:
            best_val_loss = loss_valid
            patience_counter = 0
            best_model_state = model.state_dict()
            
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print(f'Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test
    model.eval()
    with torch.no_grad():
        g_train = model(X_train)
    
    g_train = g_train.detach().cpu().numpy()

    return {
        'g_train': g_train - np.mean(g_train, axis=0)
    }




