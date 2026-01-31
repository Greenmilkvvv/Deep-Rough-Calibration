# %%
import torch

# 查看pytorch安装的版本号
print(torch.__version__) 
# 查看cuda是否可用。True为可用，即是gpu版本pytorch
print(torch.cuda.is_available())
# 返回GPU型号
print(torch.cuda.get_device_name(0)) 
# 返回可以用的cuda（GPU）数量，0代表一个
print(torch.cuda.device_count()) 
# 查看cuda的版本
print(torch.version.cuda) 

print("=" * 50)

import torch_geometric
print(torch_geometric.__version__) 


# %%
import gzip
import numpy as np

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


# %%
yy.reshape(-1, len(maturities), len(strikes))[10].shape
