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
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_pricing_CNN(nn.Module):
    """
    基于CNN的隐含波动率曲面定价器. 
    将模型参数 (H, eta, rho, v0) 编码为条件向量, 通过卷积生成 IV 曲面图像 (8x11). 
    """
    def __init__(self, hyperparams):
        """
        hyperparams = {
            'input_dim': 4,            # 模型参数维度 (H, eta, rho, v0)
            'condition_dim': 32,       # 条件向量的维度
            'conv_channels': [32, 64, 128], # 卷积层通道数列表
            'conv_kernel_size': 3,     # 卷积核大小 (推荐 3 或 5)
            'conv_padding': 1,         # 填充, 保持空间尺寸不变
            'use_batch_norm': True,    # 是否使用批归一化
            'dropout_rate': 0.0,       # Dropout 比率
            'activation': 'ELU',       # 激活函数: 'ELU', 'ReLU', 'LeakyReLU'
            'output_height': 8,        # 输出曲面高度 (期限数)
            'output_width': 11         # 输出曲面宽度 (行权价数)
        }
        """
        super().__init__()
        
        # 解析超参数
        input_dim = hyperparams.get('input_dim', 4)
        condition_dim = hyperparams.get('condition_dim', 32)
        conv_channels = hyperparams.get('conv_channels', [32, 64, 128])
        kernel_size = hyperparams.get('conv_kernel_size', 3)
        padding = hyperparams.get('conv_padding', kernel_size // 2)
        use_bn = hyperparams.get('use_batch_norm', True)
        dropout_rate = hyperparams.get('dropout_rate', 0.0)
        activation_name = hyperparams.get('activation', 'ELU')
        self.output_height = hyperparams.get('output_height', 8)
        self.output_width = hyperparams.get('output_width', 11)
        
        # ---  条件编码器 (将模型参数映射为条件向量) ---
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim, condition_dim * 2),
            self._get_activation(activation_name),
            nn.Linear(condition_dim * 2, condition_dim)
        )
        
        # ---  卷积生成器 (由条件向量生成初始特征图, 再逐步上采样) ---
        # 初始全连接层: 将条件向量扩展为空间特征
        self.init_fc = nn.Linear(condition_dim, conv_channels[0] * 4 * 4)
        
        # 构建转置卷积 (上采样) 层序列
        self.deconv_layers = nn.ModuleList()
        in_channels = conv_channels[0]
        
        # 计算所需的上采样倍数以达到目标尺寸 (8x11)
        # 策略: 从 4x4 上采样到 8x11
        scale_factors = [(2, 2), (2, 2)]  # 4x4 -> 8x8 -> 8x11 (最后通过裁剪或自适应池调整)
        
        for i, out_channels in enumerate(conv_channels):
            deconv_block = []
            # 添加转置卷积层 (上采样)
            if i < len(scale_factors):
                deconv_block.append(
                    nn.ConvTranspose2d(in_channels, out_channels, 
                                      kernel_size=3, 
                                      stride=scale_factors[i], 
                                      padding=1,
                                      output_padding=(1, 1) if scale_factors[i] == (2,2) else 0)
                )
            else:
                # 如果不需上采样, 使用普通卷积保持尺寸
                deconv_block.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                )
            
            # 添加批归一化和激活函数
            if use_bn:
                deconv_block.append(nn.BatchNorm2d(out_channels))
            
            deconv_block.append(self._get_activation(activation_name))
            
            # 添加Dropout
            if dropout_rate > 0:
                deconv_block.append(nn.Dropout2d(dropout_rate))
            
            self.deconv_layers.append(nn.Sequential(*deconv_block))
            in_channels = out_channels
        
        # ---  输出层 (将通道数映射为1, 得到单通道IV曲面) ---
        # 先使用一个卷积层平滑特征
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            self._get_activation(activation_name),
            nn.Conv2d(32, 1, kernel_size=1)  # 1x1卷积压缩到单通道
        )
        
        # 如果最终尺寸不是精确的8x11, 使用自适应池或裁剪调整
        self.size_adjust = nn.Identity()  # 先假设尺寸已正确
        
    def _get_activation(self, name):
        """获取激活函数层"""
        activations = {
            'ELU': nn.ELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(0.2),
            'Tanh': nn.Tanh()
        }
        return activations.get(name, nn.ELU())  # 默认ELU
    
    def forward(self, x):
        """
        前向传播. 
        Args:
            x: 输入张量, 形状为 [batch_size, input_dim] 
               其中 input_dim = 4 (H, eta, rho, v0)
        Returns:
            隐含波动率曲面, 形状为 [batch_size, output_height, output_width] 
            即 [batch_size, 8, 11]
        """
        batch_size = x.shape[0]
        
        # 1. 条件编码
        condition = self.condition_encoder(x)  # [batch, condition_dim]
        
        # 2. 生成初始特征图
        init_features = self.init_fc(condition)  # [batch, conv_channels[0]*4*4]
        init_features = init_features.view(batch_size, -1, 4, 4)  # [batch, C, 4, 4]
        
        # 3. 通过转置卷积上采样
        features = init_features
        for deconv_layer in self.deconv_layers:
            features = deconv_layer(features)
        
        # 4. 最终卷积得到IV曲面
        output = self.final_conv(features)  # [batch, 1, H, W]
        
        # 5. 调整尺寸并移除通道维度
        output = output.squeeze(1)  # [batch, H, W]
        
        # 6. 确保输出为精确的 [batch, 8, 11]
        if output.shape[-2:] != (self.output_height, self.output_width):
            output = F.interpolate(output.unsqueeze(1), 
                                 size=(self.output_height, self.output_width), 
                                 mode='bilinear', align_corners=False).squeeze(1)
        
        return output
    
    def predict_iv_surface(self, params_numpy):
        """
        实用方法: 从numpy参数输入预测IV曲面
        Args:
            params_numpy: numpy数组, 形状为 (4,) 或 (batch, 4)
        Returns:
            numpy数组, IV曲面形状为 (8, 11) 或 (batch, 8, 11)
        """
        self.eval()
        with torch.no_grad():
            if params_numpy.ndim == 1:
                params_numpy = params_numpy.reshape(1, -1)
            
            params_tensor = torch.FloatTensor(params_numpy)
            output = self.forward(params_tensor)
            return output.numpy()
        

