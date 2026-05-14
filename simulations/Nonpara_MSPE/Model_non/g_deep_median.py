# %% -------------import packages--------------
import torch
from torch import nn
from I_spline import I_U
import numpy as np
from B_spline2 import B_S2

#%% --------------------------
def g_D_median(train_data, valid_data, X_test, t_nodes, C, m, nodevec, n_layer, n_node, n_lr, n_epoch):
    device = C.device if isinstance(C, torch.Tensor) else torch.device('cpu')
    dtype = C.dtype if isinstance(C, torch.Tensor) else torch.float32
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C, dtype=dtype, device=device)

    # to tensor with device/dtype
    def to_t(x):
        return torch.as_tensor(x, dtype=dtype, device=device)

    X_train = to_t(train_data['X'])
    U_train = to_t(train_data['U'])
    V_train = to_t(train_data['V'])
    De1_train = to_t(train_data['De1'])
    De2_train = to_t(train_data['De2'])
    De3_train = to_t(train_data['De3'])

    X_valid = to_t(valid_data['X'])
    U_valid = to_t(valid_data['U'])
    V_valid = to_t(valid_data['V'])
    De1_valid = to_t(valid_data['De1'])
    De2_valid = to_t(valid_data['De2'])
    De3_valid = to_t(valid_data['De3'])

    X_test = to_t(X_test)

    t_nodes = np.asarray(t_nodes, dtype=np.float32) 

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

    
    def my_loss(De1, De2, De3, C, m, U, V, nodevec, g_X):

        g1 = torch.clamp(g_X[:, 0], min=-20.0, max=20.0)
        g2 = torch.clamp(g_X[:, 1], min=-20.0, max=20.0)

        scale = torch.exp(g1) 
        Ezg  = torch.exp(g2)  

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

        term1 = torch.log(torch.clamp(c1, min=eps)) + g1 + g2 - d1

        term2 = torch.log(torch.clamp(c2, min=eps)) + g1 + g2 - d2

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
        loss_train = my_loss(De1_train, De2_train, De3_train, C, m, U_train, V_train, nodevec, pred_g_X)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_g_X_valid = model(X_valid)
            loss_valid = my_loss(De1_valid, De2_valid, De3_valid, C, m, U_valid, V_valid, nodevec, pred_g_X_valid)
            

        if loss_valid < best_val_loss:
            best_val_loss = loss_valid
            patience_counter = 0
            best_model_state = model.state_dict()
            
        else:
            patience_counter += 1

        if patience_counter >= 10:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test
    model.eval()
    with torch.no_grad():
        g_train = model(X_train)
        g_train = g_train.detach().cpu().numpy()

        g_test = model(X_test)

        g1_test = torch.clamp(g_test[:, 0], min=-20.0, max=20.0)
        g2_test = torch.clamp(g_test[:, 1], min=-20.0, max=20.0)
        scale_test = torch.exp(g1_test)
        Ezg_test = torch.exp(g2_test)

        N_test = X_test.size(0)
        S_T_X_test = torch.ones(len(t_nodes), N_test, dtype=dtype, device=device)

        for k in range(len(t_nodes)):
            z = t_nodes[k] * scale_test  # (N_test,)
            It_node_k = I_U_torch(m, z, nodevec)  # (N_test, P)
            S_T_X_test[k] = torch.exp(- torch.matmul(It_node_k, C) * Ezg_test)
            
    return {
        'S_T_X_test': S_T_X_test.detach().cpu().numpy(),
        'g_train': g_train - np.mean(g_train, axis=0)
    }
