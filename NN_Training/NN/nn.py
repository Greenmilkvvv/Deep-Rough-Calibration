# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class NN_pricing(nn.Module): 
    """ 
    This basic architecture refers to "Deep Learning Volatility" by Horvath (2019).
    """
    def __init__(self, hyperparams):
        """
        hyperparams = { 
            'input_dim':5, 
            'hidden_dim':30, 
            'hidden_nums':3, 
            'output_dim':88
        }
        """

        super().__init__()
        self.input_dim = hyperparams['input_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.hidden_nums = hyperparams['hidden_nums']
        self.output_dim = hyperparams['output_dim']

        # 架构
        self.layer_lst = [] 

        # 输入层
        ## 使用 ELU 激活, 参考 Theorem 2: Universal approximation theorem for derivatives (Hornik, Stinchcombe and White)
        
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.input_dim, self.hidden_dim), 
                nn.ELU()
            )
        )

        # 隐藏层
        for _ in range(self.hidden_nums-1): # 隐藏层数量-1
            self.layer_lst.append( 
                nn.Sequential( 
                    nn.Linear(self.hidden_dim, self.hidden_dim), 
                    nn.ELU()
                )
            )
        # 最后一个隐藏层
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        )

        self.model = nn.Sequential(*self.layer_lst)

    def forward(self, x):
        return self.model(x)
    


# %%
# 加入 Residual Block 改进
class ResNet_Block(nn.Module):
    def __init__(self, hyperparams):
        """ 
        hyperparams = {
            'hidden_dim':64,
            'block_layer_nums':3
        }
        """
        super(ResNet_Block, self ).__init__()

        self.hidden_dim = hyperparams['hidden_dim']
        self.block_layer_nums = hyperparams['block_layer_nums']


        # MLP
        self.layers = nn.ModuleList() 

        for _ in range(self.block_layer_nums):
            self.layers.append( 
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )

        # 正规化 Normalization
        self.layernorms = nn.ModuleList() 
        for _ in range(self.block_layer_nums): 
            self.layernorms.append( 
                nn.LayerNorm(self.hidden_dim)
            )


    def forward(self, x): 
        # 通过 MLP 前向通过
        out = x 
        for i in range(self.block_layer_nums): 
            out = self.layers[i](out)
            out = self.layernorms[i](out)
            out = F.relu(out)

        # 实现残差链接
        out = out + x 

        return out
        

# %%
class NN_pricing_ResNet(nn.Module):
    def __init__(self, hyperparams):
        """ 
        hyperparams = {
            'input_dim':4,
            'hidden_dim':64,
            'hidden_nums':10,
            'output_dim':88,
            'block_layer_nums':3
        }
        """
        super().__init__()
        self.input_dim = hyperparams['input_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.hidden_nums = hyperparams['hidden_nums']
        self.output_dim = hyperparams['output_dim']
        self.block_layer_nums = hyperparams['block_layer_nums']


        self.layer_lst = [] 
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU()
            )
        )

        for _ in range(self.hidden_nums-1):
            self.layer_lst.append( ResNet_Block(hyperparams) )

        self.layer_lst.append( 
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        self.model = nn.Sequential(*self.layer_lst)

    def forward(self, inputs): 
        return self.model(inputs)


# %%
# LSTM 
## 核心思路是让网络先理解参数特征 (使用上面的 MLP), 再将这些特征逐步解码成曲面 (使用 LSTM).
class NN_pricing_LSTM(nn.Module):
    def __init__(self, hyperparams):
        """
        hyperparams = { 
            "input_dim": 4,
            'hidden_dim': 64, 
            'hidden_nums': 10, 
            'output_dim': 88, 
            'seq_len': 8, 
            'feature_per_step': 11
        }
        """
        super().__init__()

        # 保持原有的 MLP
        self.input_dim = hyperparams['input_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.hidden_nums = hyperparams['hidden_nums']
        self.output_dim = hyperparams['output_dim']

        # 序列相关的参数
        self.seq_len = hyperparams['seq_len']
        self.feature_per_step = hyperparams['feature_per_step']


        # MLP
        # 架构
        self.layer_lst = [] 

        # 输入层
        ## 使用 ELU 激活
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.input_dim, self.hidden_dim), 
                nn.ELU()
            )
        )

        # 隐藏层
        for _ in range(self.hidden_nums-1): # 隐藏层数量-1
            self.layer_lst.append( 
                nn.Sequential( 
                    nn.Linear(self.hidden_dim, self.hidden_dim), 
                    nn.ELU()
                )
            )

        # 注意：这里去掉了原MLP最后的输出层，只保留到特征层
        self.mlp_encoder = nn.Sequential(*self.layer_lst)


        # LSTM decoder
        ## 输入: 每个时间步我们输入MLP提取的特征
        ## 输出: 每个时间步预测一个期限的11个IV值
        self.lstm = nn.LSTM( 
            input_size=self.hidden_dim,   # 输入特征维度 = MLP输出的hidden_dim
            hidden_size=64,               # LSTM隐藏状态维度（可调，通常大于输出）
            num_layers=2,                 # LSTM层数（可调）
            batch_first=True,             # 输入形状为 (batch, seq_len, features)
            dropout=0.1 if hyperparams.get('dropout', False) else 0  # 可选dropout
        )


        # LSTM 之后的全连接层: 将 LSTM 隐藏状态映射到每个期限的 11 个隐藏波动率
        self.fc_out = nn.Linear(64, self.feature_per_step)

        ### 如果担心初始输出范围不稳定 或许可以加入 Tanh 约束输出范围
        # sekf.activation = nn.Tahn


    def forward(self, x): 
        """
        x: 输入参数，形状 [batch_size, input_dim=4]
        返回: IV曲面, 形状 [batch_size, output_dim=88]
        """
        batch_size = x.size(0) 

        # 1. encode
        encoded = self.mlp_encoder(x)


        # 2. 处理 LSTM 输入
        ## 特征向量重复seq_len次，形成序列
        ## lstm_input: [batch_size, seq_len=8, hidden_dim]
        lstm_input = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)


        # 3. LSTM 解码
        ## lstm_out: [batch_size, seq_len=8, hidden_size=64]
        lstm_out, _ = self.lstm(lstm_input)


        # 4. 将每个时间步 (maturities) 的输出映射到11维 (strikes) IV值
        # 先 reshape 以便并行处理所有时间步
        lstm_out_flat = lstm_out.reshape(-1, lstm_out.size(-1))  # [batch*8, 64]
        output_flat = self.fc_out(lstm_out_flat)                 # [batch*8, 11]


        # 5. 重塑为最终的曲面
        # output: [batch_size, 88] = [batch_size, 8*11]
        output = output_flat.reshape(batch_size, -1)

        # 可选：应用激活函数约束输出范围（如IV通常为正）
        # output = self.activation(output)
        
        return output



# %%
# GRU
class NN_pricing_GRU(nn.Module):
    def __init__(self, hyperparams):
        """
        hyperparams = { 
            "input_dim": 4,
            'hidden_dim': 64, 
            'hidden_nums': 10, 
            'output_dim': 88, 
            'seq_len': 8, 
            'feature_per_step': 11
        }
        """
        super().__init__()

        # 保持原有的 MLP
        self.input_dim = hyperparams['input_dim']
        self.hidden_dim = hyperparams['hidden_dim']
        self.hidden_nums = hyperparams['hidden_nums']
        self.output_dim = hyperparams['output_dim']

        # 序列相关的参数
        self.seq_len = hyperparams['seq_len']
        self.feature_per_step = hyperparams['feature_per_step']


        # MLP
        # 架构
        self.layer_lst = [] 

        # 输入层
        ## 使用 ELU 激活
        self.layer_lst.append( 
            nn.Sequential( 
                nn.Linear(self.input_dim, self.hidden_dim), 
                nn.ELU()
            )
        )

        # 隐藏层
        for _ in range(self.hidden_nums-1): # 隐藏层数量-1
            self.layer_lst.append( 
                nn.Sequential( 
                    nn.Linear(self.hidden_dim, self.hidden_dim), 
                    nn.ELU()
                )
            )

        # 注意：这里去掉了原MLP最后的输出层，只保留到特征层
        self.mlp_encoder = nn.Sequential(*self.layer_lst)


        # GRU decoder
        ## 输入: 每个时间步我们输入MLP提取的特征
        ## 输出: 每个时间步预测一个期限的11个IV值
        self.gru = nn.GRU( 
            input_size=self.hidden_dim,   # 输入特征维度 = MLP输出的hidden_dim
            hidden_size=64,               # LSTM隐藏状态维度（可调，通常大于输出）
            num_layers=2,                 # LSTM层数（可调）
            batch_first=True,             # 输入形状为 (batch, seq_len, features)
            dropout=0.1 if hyperparams.get('dropout', False) else 0  # 可选dropout
        )


        # GRU 之后的全连接层: 将 GRU 隐藏状态映射到每个期限的 11 个隐藏波动率
        self.fc_out = nn.Linear(64, self.feature_per_step)

        ### 如果担心初始输出范围不稳定 或许可以加入 Tanh 约束输出范围
        # sekf.activation = nn.Tahn


    def forward(self, x): 
        """
        x: 输入参数，形状 [batch_size, input_dim=4]
        返回: IV曲面, 形状 [batch_size, output_dim=88]
        """
        batch_size = x.size(0) 

        # 1. encode
        encoded = self.mlp_encoder(x)


        # 2. 处理 GRU 输入
        ## 特征向量重复seq_len次，形成序列
        ## gru_input: [batch_size, seq_len=8, hidden_dim]
        gru_input = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)


        # 3. LSTM 解码
        ## lstm_out: [batch_size, seq_len=8, hidden_size=64]
        gru_out, _ = self.gru(gru_input)


        # 4. 将每个时间步 (maturities) 的输出映射到11维 (strikes) IV值
        # 先 reshape 以便并行处理所有时间步
        gru_out_flat = gru_out.reshape(-1, gru_out.size(-1)) # [batch*8, 64]
        output_flat = self.fc_out(gru_out_flat) # [batch*8, 11]


        # 5. 重塑为最终的曲面
        # output: [batch_size, 88] = [batch_size, 8*11]
        output = output_flat.reshape(batch_size, -1)

        # 可选：应用激活函数约束输出范围（如IV通常为正）
        # output = self.activation(output)
        
        return output





