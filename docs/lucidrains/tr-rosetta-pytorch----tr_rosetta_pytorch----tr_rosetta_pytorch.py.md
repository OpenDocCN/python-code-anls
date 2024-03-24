# `.\lucidrains\tr-rosetta-pytorch\tr_rosetta_pytorch\tr_rosetta_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 中导入 F 模块
import torch.nn.functional as F

# 定义 ELU 激活函数
def elu():
    return nn.ELU(inplace=True)

# 定义 Instance Normalization 层
def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)

# 定义卷积层
def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    # 计算填充大小
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)

# 定义 trRosettaNetwork 类，继承自 nn.Module
class trRosettaNetwork(nn.Module):
    # 初始化函数
    def __init__(self, filters=64, kernel=3, num_layers=61):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers

        # 第一个块
        self.first_block = nn.Sequential(
            conv2d(442 + 2 * 42, filters, 1),
            instance_norm(filters),
            elu()
        )

        # 带有不同扩张率的残差块堆叠
        cycle_dilations = [1, 2, 4, 8, 16]
        dilations = [cycle_dilations[i % len(cycle_dilations)] for i in range(num_layers)]

        self.layers = nn.ModuleList([nn.Sequential(
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters),
            elu(),
            nn.Dropout(p=0.15),
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters)
        ) for dilation in dilations])

        self.activate = elu()

        # 转换为角度图和距离图
        self.to_prob_theta = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
        self.to_prob_phi = nn.Sequential(conv2d(filters, 13, 1), nn.Softmax(dim=1))
        self.to_distance = nn.Sequential(conv2d(filters, 37, 1), nn.Softmax(dim=1))
        self.to_prob_bb = nn.Sequential(conv2d(filters, 3, 1), nn.Softmax(dim=1))
        self.to_prob_omega = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
 
    # 前向传播函数
    def forward(self, x):
        x = self.first_block(x)

        for layer in self.layers:
            x = self.activate(x + layer(x))
        
        prob_theta = self.to_prob_theta(x)      # 角度图 theta
        prob_phi = self.to_prob_phi(x)          # 角度图 phi

        x = 0.5 * (x + x.permute((0,1,3,2)))    # 对称化

        prob_distance = self.to_distance(x)     # 距离图
        # prob_bb = self.to_prob_bb(x)            # beta-链配对（未使用）
        prob_omega = self.to_prob_omega(x)      # 角度图 omega

        return prob_theta, prob_phi, prob_distance, prob_omega
```