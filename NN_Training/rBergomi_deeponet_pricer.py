# %%
from ast import mod
import os

import gzip
# import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import torch.utils
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文宋体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import matplotlib.ticker as mtick


# %%
# 加载数据
f = gzip.GzipFile( 
    r"../Data/rBergomiTrainSet.txt.gz", 
    "r"
)
data = np.load(f) 
print(f"网格数据形状：{data.shape}")

# 网格定义
strikes=np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ])
maturities=np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ])

# xx: 参数
## 前 4 列代表网格所对应的参数
xx = data[:, :4]
print(f"参数形状：{xx.shape}")

# yy: 隐含波动率曲面 
# 后 88 列表示隐含波动率曲面 8 * 11 = 88
yy = data[:, 4:]
print(f"隐含波动率曲面形状：{yy.shape}")

print(f"参数上界: {np.max(xx, axis=0)}")
print(f"参数下界: {np.min(xx, axis=0)}")


# %%
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 分割数据集
x_train, x_test, y_train, y_test = train_test_split( 
    xx, yy, 
    test_size=0.15, 
    random_state=42
)

scale_x, scale_y = StandardScaler(), StandardScaler() 


# 工具函数——数据标准化
def x_transform(train_data, test_data): 
    return scale_x.fit_transform(train_data), scale_x.transform(test_data)

def x_inv_transform(x):
    return scale_x.inverse_transform(x)

def y_transform(train_data, test_data): 
    return scale_y.fit_transform(train_data), scale_y.transform(test_data)

def y_inv_transform(y):
    return scale_y.inverse_transform(y)


# 训练集的 Upper and Lower Bounds
upper_bound = np.array([0.16,4,-0.1,0.5])
lower_bound = np.array([0.01,0.3,-0.95,0.025])

def params_scaler(x): 
    return (x - (upper_bound+lower_bound) / 2 ) * 2 / (upper_bound-lower_bound)

def params_inv_scaler(x):
    return x * (upper_bound-lower_bound) / 2 + (upper_bound+lower_bound) / 2


x_train_transform = params_scaler(x_train)
x_test_transform = params_scaler(x_test)

y_train_transform, y_test_transform = y_transform(y_train, y_test)


# %%
# 查找 GPU 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(f"使用设备: {device}")


# 确定训练数据
train_dataset = torch.utils.data.TensorDataset( 
    torch.from_numpy(x_train_transform).to(device=device),
    torch.from_numpy(y_train_transform).to(device=device)
)

test_dataset = torch.utils.data.TensorDataset( 
    torch.from_numpy(x_test_transform).to(device=device),
    torch.from_numpy(y_test_transform).to(device=device)
)


train_data = (torch.from_numpy(x_train_transform).to(device=device),torch.from_numpy(y_train_transform).to(device=device))

test_data = (torch.from_numpy(x_test_transform).to(device=device),torch.from_numpy(y_test_transform).to(device=device))


data_loader = torch.utils.data.DataLoader( 
    train_dataset, batch_size=32, shuffle=True
)


# %%
def train_epoch(coords, loss_function, optimizer, model, loader, train_data, test_data, test = True):
    """
    执行 1 个 epoch 的训练
    
    参数
    ----
    loss_function : 损失函数
    optimizer : 优化器
    model : 模型
    loader : 数据加载器
    train_data : 训练数据
    test_data : 测试数据
    test : 是否测试 (默认为 True)
    """

    for i, (x, y) in enumerate(loader):
        # 清理 gradient
        optimizer.zero_grad() 
        # 前向传播
        y_pred = model.forward(x, coords)
        # batch loss
        # 在 loss.backward() 前添加验证
        # print(f"y_pred device: {y_pred.device}, y device: {y.device}")
        # print(f"y_pred shape: {y_pred.shape}, y shape: {y.shape}")
        loss = loss_function(y_pred, y)
        # 反向传播
        # 在 backward() 前清理梯度
        # optimizer.zero_grad(set_to_none=True)  # 使用 set_to_none=True 更彻底
        loss.backward()
        # 更新参数
        optimizer.step()


        if i % (len(loader)//100) == 0: 
            print(f'Batch: {i // len(loader) // 100 }%, Loss: {loss.item()}')

            if test: 
                # 测试时 禁用梯度计算（用于推理/评估）
                with torch.no_grad():
                    test_outputs = model.forward(test_data[0])
                    test_loss = loss_function(test_outputs, test_data[1])
                    print(f'Test Loss: {test_loss.item()}')

    
    with torch.no_grad(): 
        if test==True: 
            train_loss = loss_function(model(train_data[0]), train_data[1])
            test_loss = loss_function(model(test_data[0]), test_data[1])
            print(f'Train Loss: {train_loss.item()}')
            print(f'Test Loss: {test_loss.item()}')

            return train_loss, test_loss
        


def train_model(coords, loss_function, optimizer, model, loader, train_data, test_data, epochs=25, test = True):
    """ 
    训练模型

    参数
    ----
    loss_function : 损失函数
    optimizer : 优化器
    model : 模型
    loader : 数据加载器
    train_data : 训练数据
    test_data : 测试数据
    epochs : 训练轮数 (默认为 25)
    test : 是否测试 (默认为 True)
    """
    # 记录训练和测试损失
    train_loss_lst, test_loss_lst = [], []

    for epoch in range(epochs):
        print('-'*35, f'Epoch: {epoch+1}/{epochs}', '-'*35)
        
        train_loss, test_loss = train_epoch(
            coords,  
            loss_function, 
            optimizer, 
            model, 
            loader, 
            train_data, 
            test_data, 
            test = test
        )

        if test==True: 
            # 把损失从 gpu detach 出来到 cpu 上

            train_loss_lst.append( 
                train_loss.detach().cpu().item()
            )
            test_loss_lst.append( 
                test_loss.detach().cpu().item()
            )

            # 可视化
            plt.plot( 
                list( range(1, len(train_loss_lst)+1) ), 
                train_loss_lst,
                label = 'Train Loss'
            )
            plt.plot( 
                list( range(1, len(test_loss_lst)+1) ),
                test_loss_lst,
                label = 'Test Loss'
            )
            plt.xlabel('Epoch')
            plt.ylabel('Mean Loss')
            plt.legend()
            plt.show()

    return train_loss_lst, test_loss_lst



# %%
import sys
sys.path.append(r"../") 

from NN_Training.NN.nn import NN_pricing_DeepONet

# from NN_Training.NN.training import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


hyperparams = {
    'param_dim': 4,       # 你的4个参数
    'coord_dim': 2,       # T 和 K
    'branch_width': 64,   # 分支网络宽度
    'branch_depth': 3,    # 分支网络深度（与你的MLP隐藏层数一致）
    'trunk_width': 64,    # 主干网络宽度
    'trunk_depth': 3,     # 主干网络深度
    'latent_dim': 32,     # 特征向量维度（不宜过大，是关键超参数）
    'num_points': 88,     # 你的输出维度
    'use_bias': True,     # 推荐使用
    'activation': 'elu',  # 与你的MLP保持一致
    'enable_output_mlp': False  # 初始阶段建议关闭，保持经典结构
}

# NN_pricing_DeepONet
## 深度算子网络
model = NN_pricing_DeepONet(hyperparams).to(device=device, dtype=torch.float64)

loss_MSE = nn.MSELoss()
optim_Adam = torch.optim.Adam(model.parameters(), lr=0.0001)

tain_loss_lst, test_loss_lst = train_model( 
    loss_MSE, 
    optim_Adam, 
    model, 
    data_loader, 
    train_data, 
    test_data, 
    epochs=10
)

print(f"训练集损失: {tain_loss_lst[-1]}")
print(f"测试集损失: {test_loss_lst[-1]}")

torch.save(model.state_dict(), r"../Data/Models/nn_deeponet_rBergomi.pth")

print("模型已保存")



# %%
