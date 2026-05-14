# %% -------------import packages--------------
import torch
from torch import nn
from I_spline import I_U
import numpy as np

#%% --------------------------
def g_D_non(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_valid,U_valid,V_valid,De1_valid,De2_valid,De3_valid,W_test,U_test,V_test,W_sort1,W_sort0,t_nodes,n_layer,n_node,n_lr,n_epoch,C,m,nodevec):
    device = C.device if isinstance(C, torch.Tensor) else torch.device('cpu')
    dtype = C.dtype if isinstance(C, torch.Tensor) else torch.float32
    if not isinstance(C, torch.Tensor):
        C = torch.as_tensor(C, dtype=dtype, device=device)

    # to tensor with device/dtype
    def to_t(x):
        return torch.as_tensor(x, dtype=dtype, device=device)

    W_train = to_t(W_train)
    U_train = to_t(U_train)
    V_train = to_t(V_train)
    De1_train = to_t(De1_train)
    De2_train = to_t(De2_train)
    De3_train = to_t(De3_train)

    W_valid = to_t(W_valid)
    U_valid = to_t(U_valid)
    V_valid = to_t(V_valid)
    De1_valid = to_t(De1_valid)
    De2_valid = to_t(De2_valid)
    De3_valid = to_t(De3_valid)

    W_test = to_t(W_test)
    U_test = to_t(U_test)
    V_test = to_t(V_test)
    W_sort1 = to_t(W_sort1)
    W_sort0 = to_t(W_sort0)

    t_nodes = np.asarray(t_nodes, dtype=np.float32) 

    d = W_train.size(1)

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

    def my_loss(De1, De2, De3, C, m, U, V, nodevec, g_X):
        
        g1 = torch.clamp(g_X[:, 0], min=-200, max=200)
        g2 = torch.clamp(g_X[:, 1], min=-200, max=200)

        scale = torch.exp(g1)  
        Ezg  = torch.exp(g2)   

        Iu = I_U_torch(m, U * scale, nodevec)  # (N, P)
        Iv = I_U_torch(m, V * scale, nodevec)  # (N, P)

        # a, b: (N,)
        a = torch.matmul(Iu, C) * Ezg
        b = torch.matmul(Iv, C) * Ezg

        eps = torch.finfo(dtype).eps  
        x1 = 1.0 - torch.exp(-a)
        term1 = torch.log(torch.clamp(x1, min=eps))

        x2 = torch.exp(-a) - torch.exp(-b)
        term2 = torch.log(torch.clamp(x2, min=eps))

        term3 = - b  # -De3 * (Iv@C) * Ezg = De3 * term3

        loss_vec = -(De1 * term1 + De2 * term2 + De3 * term3)
        return torch.mean(loss_vec)

    # ---------------------------
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epoch):
        # print('epoch', epoch)
        model.train()
        pred_g_X = model(W_train)
        loss_train = my_loss(De1_train, De2_train, De3_train, C, m, U_train, V_train, nodevec, pred_g_X)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        # print('train_loss', loss_train)

        model.eval()
        with torch.no_grad():
            pred_g_X_valid = model(W_valid)
            loss_valid = my_loss(De1_valid, De2_valid, De3_valid, C, m, U_valid, V_valid, nodevec, pred_g_X_valid)
            # print('valid_loss', loss_valid)

        if loss_valid < best_val_loss:
            best_val_loss = loss_valid
            patience_counter = 0
            best_model_state = model.state_dict()
            # best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            # print('patience_counter', patience_counter)

        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch + 1}', 'train—loss=', loss_train.detach().numpy(), 'validation—loss=', loss_valid.detach().numpy())
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test
    model.eval()
    with torch.no_grad():
        g_train = model(W_train)
        g_train = g_train.detach().cpu().numpy()
        g_test1 = model(W_sort1)
        g_test0 = model(W_sort0)
        

        g_test = model(W_test)

        g1_test = torch.clamp(g_test[:, 0], min=-20.0, max=20.0)
        g2_test = torch.clamp(g_test[:, 1], min=-20.0, max=20.0)
        scale_test = torch.exp(g1_test)
        Ezg_test = torch.exp(g2_test)

        N_test = W_test.size(0)
        S_T_W_test = torch.ones(len(t_nodes), N_test, dtype=dtype, device=device)

        for k in range(len(t_nodes)):
            z = t_nodes[k] * scale_test  # (N_test,)
            It_node_k = I_U_torch(m, z, nodevec)  # (N_test, P)
            S_T_W_test[k] = torch.exp(- torch.matmul(It_node_k, C) * Ezg_test)

        I_U_W = I_U_torch(m, U_test*scale_test, nodevec)
        I_V_W = I_U_torch(m, V_test*scale_test, nodevec)

        S_U_test = torch.exp(- torch.matmul(I_U_W, C) * Ezg_test)
        S_V_test = torch.exp(- torch.matmul(I_V_W, C) * Ezg_test)


    return {
        'S_U_test': S_U_test.detach().cpu().numpy(),
        'S_V_test': S_V_test.detach().cpu().numpy(),
        'S_T_W_test': S_T_W_test.detach().cpu().numpy(),
        'g_train': g_train,
        'g_test1': g_test1.detach().cpu().numpy(),
        'g_test0': g_test0.detach().cpu().numpy(),
    }




