# %%
import numpy as np
import pandas as pd

import torch 
import scipy 
import time

import sys
sys.path.append('../')

# 导入模型
from NN_Training.NN.nn import NN_pricing

# 评估使用 cpu
device = torch.device('cpu')

# 设置参数为 4 个
hyperparams = { 
    'input_dim': 4, 
    'hidden_dim': 64, 
    'hidden_nums': 10,
    'output_dim': 88,
    'block_layer_nums': 3
}

model = NN_pricing(hyperparams=hyperparams).to(device=device, dtype=torch.float64)


model_state = torch.load( 
    '../Data/Models/nn_rBergomi.pth'
)
model.load_state_dict(model_state)

# 设置为 eval mode
model.eval()
model.to(device=device, dtype=torch.float64)


# %%
# 数据集
import gzip
f = gzip.GzipFile(
    filename = r"../Data/rBergomiTrainSet.txt.gz", 
    mode = "r"
)

data = np.load(f)
xx, yy = data[:, :4], data[:, 4:]

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


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../')

# from NN_Training.rBergomi_nn_pricer import x_transform, x_inv_transform, y_transform, y_inv_transform, params_scaler, params_inv_scaler


x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.15, random_state=42)

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


train_data = (torch.from_numpy(x_train_transform).to(device=device),torch.from_numpy(y_train_transform).to(device=device))

test_data = (torch.from_numpy(x_test_transform).to(device=device),torch.from_numpy(y_test_transform).to(device=device))


print(f"训练集形状：{train_data[0].shape}")
print(f"测试集形状：{test_data[0].shape}")


# %%
import torch
import torch.nn as nn

loss_MSE = nn.MSELoss()

vol_model = y_inv_transform(model(test_data[0]).detach().numpy())
vol_real = y_test

error_real = np.abs(vol_model-vol_real)
error_relative = error_real/vol_real

np.mean(error_relative)


# %%
# 使用 LBFGS 优化器
def calibrate_with_torch_lbfgs(model, y_test_transform, device='cpu'):
    """ 
    使用 LBFGS 优化器

    Args:
        model (torch.nn.Module): 模型
        y_test_transform (torch.Tensor): 隐含波动率曲面
        device (str, optional): 设备. Defaults to 'cpu'.
    """

    model = model.to(device)
    model.eval()  # 评估模式
    
    Approx = []
    Timing = []
    
    # 转换为tensor
    y_test_tensor = torch.from_numpy(y_test_transform).float().to(device)
    
    for i in range(len(y_test_tensor)):
        print(f"{i+1}/{len(y_test_tensor)}", end="\r")
        
        # 初始化待优化参数（需要梯度）
        params = torch.zeros(4, requires_grad=True, dtype=torch.float64, device=device)
        
        # 获取当前样本的真实值
        target = y_test_tensor[i].unsqueeze(0)  # 形状: (1, ...)
        
        # 定义 LBFGS 优化器
        optimizer = torch.optim.LBFGS(
            [params],
            lr=1.0,
            max_iter=10000,
            tolerance_grad=1e-10,
            tolerance_change=1e-10,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        # 定义 closure 函数
        def closure():
            optimizer.zero_grad()
            
            # 神经网络预测（参数需要先转换格式）
            params_reshaped = params.unsqueeze(0)  # 形状: (1, 4)
            predicted = model(params_reshaped)
            
            # 计算损失（MSE）
            loss = torch.sum((predicted - target) ** 2)
            
            # 反向传播
            loss.backward()
            
            return loss
        
        # 优化
        start = time.time()
        
        # LBFGS优化循环
        max_epochs = 100
        for epoch in range(max_epochs):
            loss = optimizer.step(closure)
            
            # 提前停止条件
            if loss.item() < 1e-10:
                break
        
        end = time.time()
        
        # 记录结果
        solutions = params.detach().cpu().numpy()
        times = end - start
        
        Approx.append(solutions)
        Timing.append(times)
    
    return np.array(Approx), np.array(Timing)


# %%
Approx, Timing = calibrate_with_torch_lbfgs(model, y_test_transform, device='cpu')


