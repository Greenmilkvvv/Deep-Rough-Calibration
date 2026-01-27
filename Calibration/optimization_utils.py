# %%
import numpy as np
import torch
import time


# %%
# 基于 torch.optim.LBFGS 的优化函数
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


