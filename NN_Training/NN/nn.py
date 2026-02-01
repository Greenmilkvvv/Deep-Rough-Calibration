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

        for _ in range(self.hidden_nums):
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

        # 注意: 这里去掉了原MLP最后的输出层, 只保留到特征层
        self.mlp_encoder = nn.Sequential(*self.layer_lst)


        # LSTM decoder
        ## 输入: 每个时间步我们输入MLP提取的特征
        ## 输出: 每个时间步预测一个期限的11个IV值
        self.lstm = nn.LSTM( 
            input_size=self.hidden_dim,   # 输入特征维度 = MLP输出的hidden_dim
            hidden_size=64,               # LSTM隐藏状态维度 (可调, 通常大于输出 )
            num_layers=2,                 # LSTM层数 (可调 )
            batch_first=True,             # 输入形状为 (batch, seq_len, features)
            dropout=0.1 if hyperparams.get('dropout', False) else 0  # 可选dropout
        )


        # LSTM 之后的全连接层: 将 LSTM 隐藏状态映射到每个期限的 11 个隐藏波动率
        self.fc_out = nn.Linear(64, self.feature_per_step)

        ### 如果担心初始输出范围不稳定 或许可以加入 Tanh 约束输出范围
        # sekf.activation = nn.Tahn


    def forward(self, x): 
        """
        x: 输入参数, 形状 [batch_size, input_dim=4]
        返回: IV曲面, 形状 [batch_size, output_dim=88]
        """
        batch_size = x.size(0) 

        # 1. encode
        encoded = self.mlp_encoder(x)


        # 2. 处理 LSTM 输入
        ## 特征向量重复seq_len次, 形成序列
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

        # 可选: 应用激活函数约束输出范围 (如IV通常为正 )
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

        # 注意: 这里去掉了原MLP最后的输出层, 只保留到特征层
        self.mlp_encoder = nn.Sequential(*self.layer_lst)


        # GRU decoder
        ## 输入: 每个时间步我们输入MLP提取的特征
        ## 输出: 每个时间步预测一个期限的11个IV值
        self.gru = nn.GRU( 
            input_size=self.hidden_dim,   # 输入特征维度 = MLP输出的hidden_dim
            hidden_size=64,               # LSTM隐藏状态维度 (可调, 通常大于输出 )
            num_layers=2,                 # LSTM层数 (可调 )
            batch_first=True,             # 输入形状为 (batch, seq_len, features)
            dropout=0.1 if hyperparams.get('dropout', False) else 0  # 可选dropout
        )


        # GRU 之后的全连接层: 将 GRU 隐藏状态映射到每个期限的 11 个隐藏波动率
        self.fc_out = nn.Linear(64, self.feature_per_step)

        ### 如果担心初始输出范围不稳定 或许可以加入 Tanh 约束输出范围
        # sekf.activation = nn.Tahn


    def forward(self, x): 
        """
        x: 输入参数, 形状 [batch_size, input_dim=4]
        返回: IV曲面, 形状 [batch_size, output_dim=88]
        """
        batch_size = x.size(0) 

        # 1. encode
        encoded = self.mlp_encoder(x)


        # 2. 处理 GRU 输入
        ## 特征向量重复seq_len次, 形成序列
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

        # 可选: 应用激活函数约束输出范围 (如IV通常为正 )
        # output = self.activation(output)
        
        return output



# %%
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool # 需要安装 torch_geometric

class NN_pricing_GNN(torch.nn.Module):
    """
    基于GNN的隐含波动率曲面定价器. 
    输入: 模型参数 (H, eta, rho, v0) [重复给每个节点]
    输出: 整个隐含波动率曲面 (88维)
    """
    def __init__(self, hyperparams):
        """
        hyperparams = {
            'node_feature_dim': 4, # 每个节点的特征维度, 即模型参数数量 (H, eta, rho, v0)
            'hidden_dim': 64, # 隐藏层维度
            'gnn_num_layers': 3, # GNN卷积层数量
            'use_jk': True, # 是否使用跳跃连接聚合 (Jumping Knowledge)
            'jk_mode': 'cat', # JK聚合模式: 'cat', 'lstm', 'max'
            'mlp_num_layers': 1, # 输出头MLP的层数 (在GNN层之后) 
            'dropout_rate': 0.0 # Dropout率 用于正则化
        }
        """
        super().__init__()
        
        # 解析超参数
        self.node_feature_dim = hyperparams.get('node_feature_dim', 4)
        hidden_dim = hyperparams.get('hidden_dim', 64)
        gnn_num_layers = hyperparams.get('gnn_num_layers', 3)
        use_jk = hyperparams.get('use_jk', True)
        jk_mode = hyperparams.get('jk_mode', 'cat')
        mlp_num_layers = hyperparams.get('mlp_num_layers', 1)
        dropout_rate = hyperparams.get('dropout_rate', 0.0)
        
        # 输入投影层 (将节点特征映射到隐藏空间)
        self.input_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            torch.nn.ELU()
        )
        
        # 核心GNN层堆叠
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(gnn_num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 跳跃连接聚合 (可选, 稳定深层GNN训练 )
        self.use_jk = use_jk
        if self.use_jk:
            self.jk = JumpingKnowledge(mode=jk_mode)
            jk_out_dim = hidden_dim * gnn_num_layers if jk_mode == 'cat' else hidden_dim
        else:
            jk_out_dim = hidden_dim
            
        self.dropout = torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        
        # 输出头MLP (将每个节点的最终表示映射为一个标量IV值 )
        # 构建一个小的MLP作为输出头
        output_mlp_layers = []
        current_dim = jk_out_dim
        for _ in range(mlp_num_layers - 1):
            output_mlp_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                torch.nn.ELU()
            ])
            if self.dropout:
                output_mlp_layers.append(self.dropout)
            current_dim = hidden_dim
        # 最后一层, 输出每个节点的隐含波动率
        output_mlp_layers.append(nn.Linear(current_dim, 1))
        self.output_mlp = nn.Sequential(*output_mlp_layers)

    def forward(self, x, edge_index, batch=None):
        """
        前向传播. 
        Args:
            x: 节点特征张量, 形状为 [num_nodes, node_feature_dim]
               *对于本任务,  num_nodes = 88, node_feature_dim = 4. 
               *每个节点的特征向量相同, 均为当前的模型参数 (H, eta, rho, v0). 
            edge_index: 图的边索引, 形状为 [2, num_edges]. 
            batch: 批处理索引, 形状为 [num_nodes]. 如果为None,则假定所有节点属于同一个图. 
        Returns:
            隐含波动率曲面, 形状为 [num_nodes] 或 [batch_size * num_nodes_per_graph]. 
        """
        # 1. 输入投影
        x = self.input_proj(x)
        
        # 2. GNN消息传递
        xs = []  # 存储每一层的节点表示
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)  # PyG的核心: 高效邻居聚合
            x = bn(x)
            x = F.elu(x)
            if self.dropout:
                x = self.dropout(x)
            if self.use_jk:
                xs.append(x)
        
        # 3. 跳跃连接聚合
        if self.use_jk:
            x = self.jk(xs)
        # 如果不用JK, x就是最后一层的输出
        
        # 4. 节点级输出预测 (每个节点预测自己的IV值)
        out = self.output_mlp(x)
        
        # 5. 返回展平后的结果 (整个曲面 )
        # 如果你的 batch 索引不为None, 这里返回的就是所有图的拼接结果
        return out.squeeze(-1)  # 从 [num_nodes, 1] -> [num_nodes]


# %%
class NN_pricing_CNN(nn.Module):
    """
    基于CNN的隐含波动率曲面定价器. 
    输入: 模型参数 (H, η, rho, v0) [batch_size, 4]
    输出: 展平的隐含波动率曲面 [batch_size, 88]
    设计原则: 使用卷积层提取曲面空间特征, 通过最大池化下采样, 配合Dropout防止过拟合. 
    """
    def __init__(self, hyperparams):
        """
        Args:
            hyperparams (dict): 包含模型超参数的字典. 
                必需键: 'input_dim' (应固定为4)
                可选键 (带默认值) :
                    - 'init_channels': 第一层卷积核数, 默认 32
                    - 'dense_units': 全连接层单元数, 默认 128
                    - 'dropout_rates': Dropout率列表, 默认 [0.25, 0.25, 0.4, 0.3]
                    - 'conv_kernel_size': 卷积核尺寸, 默认 3
                    - 'pool_kernel_size': 池化核尺寸, 默认 2
        """
        super().__init__()
        
        # 解析超参数, 设置默认值
        self.input_dim = hyperparams.get('input_dim', 4)
        init_channels = hyperparams.get('init_channels', 32)
        dense_units = hyperparams.get('dense_units', 128)
        dropout_rates = hyperparams.get('dropout_rates', [0.25, 0.25, 0.4, 0.3])
        conv_kernel_size = hyperparams.get('conv_kernel_size', 3)
        pool_kernel_size = hyperparams.get('pool_kernel_size', 2)
        
        # 输入校验
        assert self.input_dim == 4, f"输入维度应为4 (H, η, rho, v0) , 但传入 {self.input_dim}"
        assert len(dropout_rates) == 4, "dropout_rates 应包含4个值 (对应3个卷积后和1个全连接后的Dropout) "
        
        # --- 特征投影与图像重塑 ---
        # 将4维参数投影到更高维, 然后重塑为伪图像, 模拟曲面网格
        # 投影维度需能被重塑后的高度整除, 这里选择投影到 8*11 = 88 维
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 88)  # 投影到与输出曲面相同的总点数
        )
        
        # --- CNN 编码器部分 ---
        # 卷积块1: 输入通道=1 (灰度图), 输出通道=init_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=init_channels,
                               kernel_size=conv_kernel_size, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.drop1 = nn.Dropout2d(p=dropout_rates[0])
        
        # 卷积块2: 通道数翻倍
        self.conv2 = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2,
                               kernel_size=conv_kernel_size, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.drop2 = nn.Dropout2d(p=dropout_rates[1])
        
        # 卷积块3: 通道数再翻倍
        self.conv3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4,
                               kernel_size=conv_kernel_size, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel_size)
        self.drop3 = nn.Dropout2d(p=dropout_rates[2])
        
        # --- 全连接解码器部分 ---
        # 计算卷积层展平后的特征维度
        # 假设输入“图像”大小为 8x11 (高x宽)
        def get_flatten_size(h, w):
            # 经过3次 kernel_size=2 的池化
            h = h // (pool_kernel_size ** 3)
            w = w // (pool_kernel_size ** 3)
            # 确保尺寸为正
            return max(1, h), max(1, w)
        
        # 确定展平后的维度
        final_h, final_w = get_flatten_size(8, 11)
        flatten_dim = init_channels * 4 * final_h * final_w
        
        self.fc1 = nn.Linear(flatten_dim, dense_units)
        self.drop_fc = nn.Dropout(p=dropout_rates[3])
        # 输出层: 映射回88个曲面点
        self.fc_out = nn.Linear(dense_units, 88)
        
        # 记录中间值, 便于调试
        self.flatten_dim = flatten_dim
        self.final_h, self.final_w = final_h, final_w

    def forward(self, x):
        """
        前向传播. 
        Args:
            x: 输入张量, 形状为 [batch_size, 4], 包含模型参数. 
        Returns:
            输出张量, 形状为 [batch_size, 88], 即预测的隐含波动率曲面. 
        """
        batch_size = x.size(0)
        
        # 1. 投影与重塑: 参数 -> 伪图像
        x = self.projection(x)  # [batch, 88]
        # 重塑为图像格式: [batch, channels=1, height=8, width=11]
        x = x.view(batch_size, 1, 8, 11)
        
        # 2. 卷积编码路径
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # 3. 展平并全连接解码
        x = x.view(batch_size, -1)  # 自动展平
        # 可选: 添加一个维度校验 (训练时建议注释掉以提升速度) 
        # if x.size(1) != self.flatten_dim:
        #     raise ValueError(f"展平维度不匹配: 预期 {self.flatten_dim}, 实际 {x.size(1)}")
        
        x = F.elu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc_out(x)  # 输出形状: [batch_size, 88]
        
        return x
        

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_pricing_DeepONet(nn.Module):
    """
    基于DeepONet架构的粗糙波动率模型定价器. 
    
    输入: 
        - params: 模型参数 [batch_size, param_dim] (例如 [batch, 4])
        - coords: 曲面坐标 [batch_size, num_points, coord_dim] (例如 [batch, 88, 2])
    输出: 
        - 隐含波动率曲面 [batch_size, num_points] (例如 [batch, 88])
    """
    def __init__(self, hyperparams):
        """
        hyperparams = {
            'param_dim': 4,             # 模型参数维度 (H, η, rho, v0)
            'coord_dim': 2,             # 坐标维度 (T, K)
            'branch_width': 64,         # 分支网络隐藏层宽度
            'branch_depth': 3,          # 分支网络深度 (隐藏层层数 )
            'trunk_width': 64,          # 主干网络隐藏层宽度  
            'trunk_depth': 3,           # 主干网络深度
            'latent_dim': 32,           # 特征向量的最终维度 (b和t的维度 )
            'num_points': 88,           # 输出曲面的点数 (固定网格 )
            'use_bias': True,           # 是否在合并后添加可学习的偏置
            'activation': 'elu'         # 激活函数类型 ('relu', 'elu', 'tanh')
        }
        """
        super().__init__()
        
        # 解析超参数
        self.param_dim = hyperparams.get('param_dim', 4)
        self.coord_dim = hyperparams.get('coord_dim', 2)
        self.branch_width = hyperparams.get('branch_width', 64)
        self.branch_depth = hyperparams.get('branch_depth', 3)
        self.trunk_width = hyperparams.get('trunk_width', 64)
        self.trunk_depth = hyperparams.get('trunk_depth', 3)
        self.latent_dim = hyperparams.get('latent_dim', 32)
        self.num_points = hyperparams.get('num_points', 88)
        self.use_bias = hyperparams.get('use_bias', True)
        act_name = hyperparams.get('activation', 'elu')
        
        # 激活函数选择
        activation_dict = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.01)
        }
        self.activation = activation_dict.get(act_name, nn.ELU())
        
        # --- 1. 分支网络: 处理模型参数 ---
        branch_layers = []
        in_dim = self.param_dim
        # 构建隐藏层
        for i in range(self.branch_depth):
            branch_layers.append(nn.Linear(in_dim, self.branch_width))
            branch_layers.append(nn.BatchNorm1d(self.branch_width))
            branch_layers.append(self.activation)
            if i < self.branch_depth - 1:  # 除最后一层外, 可添加Dropout
                branch_layers.append(nn.Dropout(0.1))
            in_dim = self.branch_width
        # 输出层: 投影到潜空间维度
        branch_layers.append(nn.Linear(self.branch_width, self.latent_dim))
        self.branch_net = nn.Sequential(*branch_layers)
        
        # --- 2. 主干网络: 处理坐标点 ---
        trunk_layers = []
        in_dim = self.coord_dim
        # 构建隐藏层
        for i in range(self.trunk_depth):
            trunk_layers.append(nn.Linear(in_dim, self.trunk_width))
            trunk_layers.append(nn.BatchNorm1d(self.trunk_width))
            trunk_layers.append(self.activation)
            if i < self.trunk_depth - 1:
                trunk_layers.append(nn.Dropout(0.1))
            in_dim = self.trunk_width
        # 输出层: 投影到潜空间维度
        trunk_layers.append(nn.Linear(self.trunk_width, self.latent_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # --- 3. 可学习的偏置项 (每个输出点一个偏置 )---
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_points))
        else:
            self.bias = None
            
        # --- 4. 可选的输出调整层 (轻量MLP, 用于后处理 )---
        # 如果希望合并后的特征再经过非线性变换, 可以启用此层
        self.enable_output_mlp = hyperparams.get('enable_output_mlp', False)
        if self.enable_output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(1, 8),  # 输入是合并后的标量, 先提升维度
                self.activation,
                nn.Linear(8, 1)   # 再降回1维
            )
        else:
            self.output_mlp = None

    def forward(self, params, coords):
        """
        前向传播. 
        
        Args:
            params: 模型参数张量 [batch_size, param_dim]
            coords: 坐标张量 [batch_size, num_points, coord_dim]
            
        Returns:
            隐含波动率曲面 [batch_size, num_points]
        """
        batch_size = params.shape[0]
        num_points = coords.shape[1]  # 通常是88, 但网络可以处理任意点数
        
        # 1. 分支网络: 编码参数 -> [batch, latent_dim]
        branch_out = self.branch_net(params)  # [batch, latent_dim]
        
        # 2. 主干网络: 编码每个坐标点
        # 重塑坐标以便批处理: [batch, num_points, coord_dim] -> [batch*num_points, coord_dim]
        coords_reshaped = coords.view(-1, self.coord_dim)
        trunk_out = self.trunk_net(coords_reshaped)  # [batch*num_points, latent_dim]
        # 恢复形状: [batch, num_points, latent_dim]
        trunk_out = trunk_out.view(batch_size, num_points, self.latent_dim)
        
        # 3. 合并操作: 点积求和 (经典DeepONet方式)
        # branch_out: [batch, latent_dim] -> 扩展为 [batch, 1, latent_dim]
        # trunk_out: [batch, num_points, latent_dim]
        # 结果: [batch, num_points]
        output = torch.einsum('bd,bpd->bp', branch_out, trunk_out)
        
        # 4. 可选: 通过轻量MLP调整每个点的输出
        if self.output_mlp is not None:
            # 将输出视为独立标量进行处理
            output = output.unsqueeze(-1)  # [batch, num_points, 1]
            output = self.output_mlp(output)  # [batch, num_points, 1]
            output = output.squeeze(-1)  # [batch, num_points]
        
        # 5. 添加可学习的偏置 (每个网格点一个 )
        if self.bias is not None:
            # bias: [num_points] 自动广播到 [batch, num_points]
            output = output + self.bias
        
        return output

    def predict_at_new_coords(self, params, new_coords):
        """
        在训练好的模型上预测新坐标点的值. 
        这展示了DeepONet的灵活性: 可以评估任意点, 而不仅是训练时的固定网格. 
        
        Args:
            params: [batch_size, param_dim]
            new_coords: [batch_size, arbitrary_num_points, coord_dim]
            
        Returns:
            [batch_size, arbitrary_num_points]
        """
        return self.forward(params, new_coords)
    

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedBlock(nn.Module):
    """一个标准的GLU块"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 投影变换 (通常无偏置, 或与门共享)
        self.projection = nn.Linear(input_dim, output_dim)
        # 门控变换
        self.gate = nn.Linear(input_dim, output_dim)
        # 可选的残差连接（如果维度匹配）
        self.use_residual = (input_dim == output_dim)

    def forward(self, x):
        residual = x if self.use_residual else 0
        proj = self.projection(x)
        gate = torch.sigmoid(self.gate(x))  # 门控信号在0-1之间
        out = proj * gate + residual
        return out
    

class NN_pricing_GLU(nn.Module):
    """
    基于门控线性单元(GLU)的定价网络。
    保留了MLP的全局连接性, 但通过门控机制增强了表达能力。
    """
    def __init__(self, hyperparams):
        super().__init__()
        input_dim = hyperparams['input_dim']
        hidden_dim = hyperparams['hidden_dim']
        hidden_nums = hyperparams['hidden_nums']
        output_dim = hyperparams['output_dim']

        self.layer_lst = nn.ModuleList()

        # 输入层: 使用GLU块
        self.layer_lst.append(GatedBlock(input_dim, hidden_dim))

        # 隐藏层: 堆叠多个GLU块
        for _ in range(hidden_nums - 1):
            self.layer_lst.append(GatedBlock(hidden_dim, hidden_dim))

        # 输出层: 一个简单的线性投影
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 可选的层归一化, 稳定训练
        self.use_norm = hyperparams.get('use_norm', True)
        if self.use_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(hidden_nums)])

    def forward(self, x):
        for i, layer in enumerate(self.layer_lst):
            x = layer(x)
            if self.use_norm:
                x = self.norms[i](x)
        x = self.output_layer(x)
        return x

