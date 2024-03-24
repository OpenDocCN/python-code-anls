# `.\lucidrains\mogrifier\mogrifier\mogrifier.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 定义一个函数 weight，用于创建线性层
def weight(dim_in, dim_out, factorize_k = None):
    # 如果没有指定 factorize_k，则直接返回一个线性层
    if factorize_k is None:
        return nn.Linear(dim_in, dim_out, bias = False)

    # 断言 factorize_k 必须小于 dim_in 和 dim_out，否则抛出异常
    assert factorize_k < dim_in and factorize_k < dim_out, 'k must be of relative lower rank'

    # 如果指定了 factorize_k，则返回一个包含两个线性层的序列
    return nn.Sequential(
        nn.Linear(dim_in, factorize_k, bias = False),
        nn.Linear(factorize_k, dim_out, bias = False)
    )

# 定义一个 Mogrifier 类，继承自 nn.Module
class Mogrifier(nn.Module):
    # 初始化方法
    def __init__(self, dim, iters = 5, factorize_k = None):
        super().__init__()
        self.dim = dim
        self.iters = iters

        # 创建 Q 线性层
        self.Q = weight(dim, dim, factorize_k)
        # 如果迭代次数大于 1，则创建 R 线性层，否则为 None
        self.R = weight(dim, dim, factorize_k) if iters > 1 else None

    # 前向传播方法
    def forward(self, x, h):
        shape = x.shape
        *_, dim = shape
        # 断言输入张量的最后一个维度必须等于 self.dim
        assert dim == self.dim, f'mogrifier accepts a dimension of {self.dim}'

        # 将输入张量 x 和 h 重塑为二维张量
        x, h = map(lambda t: t.reshape(-1, dim), (x, h))

        # 迭代执行 Mogrifier 算法
        for ind in range(self.iters):
            if (ind % 2) == 0:
                x = 2 * self.Q(h).sigmoid() * x
            else:
                h = 2 * self.R(x).sigmoid() * h

        # 将 x 和 h 重塑为原始形状
        x, h = map(lambda t: t.reshape(*shape), (x, h))
        return x, h
```