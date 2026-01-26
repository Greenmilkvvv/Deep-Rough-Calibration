# %%
import os

import gzip
import numpy as np

print('game is on')

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

# 参数
print(f"参数上界: {np.max(xx, axis=0)}")
print(f"参数下界: {np.min(xx, axis=0)}")


print('='*40)


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

print(f"训练集形状：{x_train_transform.shape}") 
print(f"测试集形状：{x_test_transform.shape}")


# %%
import torch

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
import torch.nn as nn

import sys
sys.path.append(r"../") 

from NN_Training.NN.nn import NN_pricing_GRU
from NN_Training.NN.training import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


hyperparams = { 
    "input_dim": 4,
    'hidden_dim': 64, 
    'hidden_nums': 10, 
    'output_dim': 88
}





