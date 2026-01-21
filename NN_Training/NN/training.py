# %%
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文宋体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# %%
def train_epoch(loss_function, optimizer, model, loader, train_data, test_data, test = True):
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
        y_pred = model.forward(x)
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



# %%
def train_model(loss_function, optimizer, model, loader, train_data, test_data, epochs=25, test = True):
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
