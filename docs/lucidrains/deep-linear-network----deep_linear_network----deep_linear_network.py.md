# `.\lucidrains\deep-linear-network\deep_linear_network\deep_linear_network.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 functools 库中导入 reduce 函数
from functools import reduce

# 定义矩阵相乘函数 mm
def mm(x, y):
    return x @ y

# 定义一个继承自 nn.Module 的类 DeepLinear
class DeepLinear(nn.Module):
    # 初始化函数，接受输入维度 dim_in 和多个维度参数 *dims
    def __init__(self, dim_in, *dims):
        super().__init__()
        # 将输入维度和参数 dims 组成一个维度列表 dims
        dims = [dim_in, *dims]
        # 将 dims 列表中相邻的维度组成元组，形成维度对列表 pairs
        pairs = list(zip(dims[:-1], dims[1:]))
        # 使用 map 函数对 pairs 中的每个维度对创建一个随机初始化的权重参数，并组成权重列表 weights
        weights = list(map(lambda d: nn.Parameter(torch.randn(d)), pairs))
        # 将权重列表转换为 nn.ParameterList 类型，并赋值给 self.weights
        self.weights = nn.ParameterList(weights)
        # 初始化缓存变量为 None
        self._cache = None

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 如果处于训练模式，重置缓存并返回权重矩阵相乘后的结果
        if self.training:
            self._cache = None
            return reduce(mm, self.weights, x)

        # 如果缓存不为空，直接返回输入 x 与缓存的权重矩阵相乘的结果
        if self._cache is not None:
            return x @ self._cache

        # 从权重列表中取出第一个权重矩阵作为头部，其余作为尾部，计算尾部权重矩阵的乘积，并缓存结果
        head, *tail = self.weights
        weight = reduce(mm, tail, head)
        self._cache = weight
        return x @ weight
```