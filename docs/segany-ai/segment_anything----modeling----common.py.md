# `.\segment-anything\segment_anything\modeling\common.py`

```
# 导入 torch 库和 torch.nn 模块
import torch
import torch.nn as nn

# 导入 Type 类型提示
from typing import Type

# 定义 MLPBlock 类，继承自 nn.Module
class MLPBlock(nn.Module):
    # 初始化函数，接受 embedding_dim（嵌入维度）、mlp_dim（MLP维度）和激活函数 act（默认为 GELU）
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        # 创建线性层，输入维度为 embedding_dim，输出维度为 mlp_dim
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        # 创建线性层，输入维度为 mlp_dim，输出维度为 embedding_dim
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        # 初始化激活函数
        self.act = act()

    # 前向传播函数，接受输入张量 x，返回输出张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先经过第一个线性层和激活函数，再经过第二个线性层
        return self.lin2(self.act(self.lin1(x)))


# 定义 LayerNorm2d 类，继承自 nn.Module
class LayerNorm2d(nn.Module):
    # 初始化函数，接受通道数 num_channels 和 eps（默认为 1e-6）
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # 初始化可学习参数 weight 和 bias
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    # 前向传播函数，接受输入张量 x，返回归一化后的张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均值和方差
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        # 归一化输入张量 x
        x = (x - u) / torch.sqrt(s + self.eps)
        # 加权和偏置
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
```