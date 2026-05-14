
# %% -------------import packages--------------
import torch
from torch import nn
from I_spline import I_U
import copy  # 必须导入 copy 模块用于深拷贝模型参数
import numpy as np

#%% --------------------------
def g_D_semi(train_data, valid_data, X_test, t_nodes, theta, C, m, nodevec, n_layer, n_node, n_lr, n_epoch):
    # --- 1. 处理训练数据 ---
    Z_train = torch.Tensor(train_data['X'][:,0])
    X_train = torch.Tensor(train_data['X'][:,1:])
    U_train = torch.Tensor(train_data['U'])
    V_train = torch.Tensor(train_data['V'])
    De1_train = torch.Tensor(train_data['De1'])
    De2_train = torch.Tensor(train_data['De2'])
    De3_train = torch.Tensor(train_data['De3'])
    # g_train_true = torch.Tensor(np.c_[train_data['g1_X'], train_data['g2_X']]) # 未使用，注释掉或保留均可

    # --- 2. 处理验证数据 (新增) ---
    Z_val = torch.Tensor(valid_data['X'][:,0])
    X_val = torch.Tensor(valid_data['X'][:,1:])
    U_val = torch.Tensor(valid_data['U'])
    V_val = torch.Tensor(valid_data['V'])
    De1_val = torch.Tensor(valid_data['De1'])
    De2_val = torch.Tensor(valid_data['De2'])
    De3_val = torch.Tensor(valid_data['De3'])

    # --- 3. 处理测试和其他数据 ---
    X_test = torch.Tensor(X_test)

    t_nodes = np.asarray(t_nodes, dtype=np.float32) 

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

    def I_U_torch(m_, z_t, nodevec_):
        z_np = z_t.detach().numpy()
        out_np = I_U(m_, z_np, nodevec_)  
        return torch.as_tensor(out_np)
    


    def my_loss(De1, De2, De3, Z, theta, C, m, U, V, nodevec, g_X):
        # 注意：这里使用了 detach()，意味着 I_U 内部的计算不反向传播梯度到 g_X
        # 只有 Ezg 部分会产生梯度
        Iu = torch.Tensor(I_U(m, U.detach().numpy() * (torch.exp(Z * theta[0] + g_X[:,0])).detach().numpy(), nodevec))
        Iv = torch.Tensor(I_U(m, V.detach().numpy() * (torch.exp(Z * theta[0] + g_X[:,0])).detach().numpy(), nodevec))
        Ezg = torch.exp(Z * theta[1] + g_X[:,1])
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- torch.matmul(Iu,C) * Ezg) + 1e-4) + De2 * torch.log(torch.exp(- torch.matmul(Iu,C) * Ezg) - torch.exp(- torch.matmul(Iv,C) * Ezg) + 1e-4) - De3 * torch.matmul(Iv,C) * Ezg)
        return loss_fun


    # --- 4. 训练循环 (加入早停机制) ---
    patience = 10
    counter = 0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict()) # 初始化最佳权重

    for epoch in range(n_epoch):
        # --- Training Step ---
        model.train() # 启用 Dropout/BatchNorm 等 (如果有)
        pred_g_X_train = model(X_train)
        loss_train = my_loss(De1_train, De2_train, De3_train, Z_train, theta, C, m, U_train, V_train, nodevec, pred_g_X_train)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # --- Validation Step ---
        model.eval() # 切换到评估模式
        with torch.no_grad(): # 不计算梯度，节省内存和计算资源
            pred_g_X_val = model(X_val)
            loss_val = my_loss(De1_val, De2_val, De3_val, Z_val, theta, C, m, U_val, V_val, nodevec, pred_g_X_val)

        # --- Early Stopping Logic ---
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model_wts = copy.deepcopy(model.state_dict()) # 保存当前最佳模型
            counter = 0 # 重置计数器
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}')
                break
        
        # 可选：打印日志
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}: Train Loss {loss_train.item():.4f}, Val Loss {loss_val.item():.4f}")

    # %% ----------------------
    # 恢复最佳模型权重
    model.load_state_dict(best_model_wts)
    model.eval() # 最终预测使用评估模式

    # 使用最佳模型进行预测
    with torch.no_grad():
        g_train = model(X_train)
        g_test = model(X_test[:, 1:])
        g1_test = torch.clamp(g_test[:, 0], min=-20.0, max=20.0)
        g2_test = torch.clamp(g_test[:, 1], min=-20.0, max=20.0)
        scale_test = torch.exp(X_test[:, 0] * theta[0] + g1_test)
        Ezg_test = torch.exp(X_test[:, 0] * theta[1] + g2_test)

        N_test = X_test.size(0)
        S_T_X_test = torch.ones(len(t_nodes), N_test)

        for k in range(len(t_nodes)):
            z = t_nodes[k] * scale_test  # (N_test,)
            It_node_k = I_U_torch(m, z, nodevec)  # (N_test, P)
            S_T_X_test[k] = torch.exp(- torch.matmul(It_node_k, C) * Ezg_test)
        
    
    g_train = g_train.detach().numpy()
    g_test = g_test.detach().numpy()
    
    return {
        'S_T_X_test': S_T_X_test.detach().numpy(),
        'g_train': g_train - np.mean(g_train, axis=0),
        'g_test': g_test
    }


