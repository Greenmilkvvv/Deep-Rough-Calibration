# %%
from ast import mod
import os

import gzip
import pandas as pd
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


# %%
# 参数
print(f"参数上界: {np.max(xx, axis=0)}")
print(f"参数下界: {np.min(xx, axis=0)}")

# %%
# 画出 Implied Volatility
from mpl_toolkits.mplot3d import Axes3D

def ImpVol_surface_3d( 
        xx_data: np.ndarray, 
        yy_data: np.ndarray, 
        K: np.array = np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ]), 
        M: list = [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ],
        params_index: int = 10000, 
        save_path = None
): 
    """
    绘制隐含波动率曲面
    
    Args:
        xx_data (np.ndarray): 参数
        yy_data (np.ndarray): 隐含波动率曲面
        K (np.array, optional): 行权价. Defaults to np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ]).
        M (list, optional): 到期时间. Defaults to [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ].
        params_index (int, optional): 参数索引. Defaults to 10000.
    """

    model_params = xx_data[params_index,:]
    
    if yy_data.shape[1] != len(K) * len(M): return None

    vol_surf_with_params = yy_data.reshape(-1, len(M), len(K))[params_index]

    x_axis, y_axis = np.meshgrid(M, K, indexing='ij')

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf_to_plot = ax.plot_surface( 
        x_axis, 
        y_axis, 
        vol_surf_with_params, 
        cmap='viridis'
    )
    ax.set_xlabel('到期时间')
    ax.set_ylabel('行权价')
    ax.set_zlabel('隐含波动率')
    ax.set_title(f"隐含波动率曲面 (参数: {model_params.tolist()})")
    
    # color bar
    fig.colorbar(surf_to_plot, shrink=0.5, aspect=5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    
    return 

ImpVol_surface_3d(xx, yy, params_index=3999)


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

# %%
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
import sys
sys.path.append(r"../") 

from NN_Training.NN.nn import NN_pricing, NN_pricing_ResNet
from NN_Training.NN.training import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


hyperparams = { 
    "input_dim": 4,
    'hidden_dim': 64, 
    'hidden_nums': 10, 
    'output_dim': 88, 
    'block_layer_nums': 3
}

model = NN_pricing(hyperparams).to(device=device, dtype=torch.float64)

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

torch.save(model.state_dict(), r"../Data/Models/nn_rBergomi.pth")

print("模型已保存")

# %%
# NN_pricing_ResNet
model = NN_pricing_ResNet(hyperparams).to(device=device, dtype=torch.float64)

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

torch.save(model.state_dict(), r"../Data/Models/nn_resnet_rBergomi.pth")

print("模型已保存")

