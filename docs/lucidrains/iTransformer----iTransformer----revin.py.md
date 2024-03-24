# `.\lucidrains\iTransformer\iTransformer\revin.py`

```
# 导入必要的库
from collections import namedtuple
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# 定义一个命名元组，用于存储统计信息
Statistics = namedtuple('Statistics', [
    'mean',
    'variance',
    'gamma',
    'beta'
])

# 可逆实例归一化
# 提议的实例归一化方法，参考 https://openreview.net/forum?id=cGDAkQo1C0p

class RevIN(Module):
    def __init__(
        self,
        num_variates,
        affine = True,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        # 初始化可学习参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_variates, 1), requires_grad = affine)
        self.beta = nn.Parameter(torch.zeros(num_variates, 1), requires_grad = affine)

    def forward(self, x, return_statistics = False):
        assert x.shape[1] == self.num_variates

        # 计算均值和方差
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        var_rsqrt = var.clamp(min = self.eps).rsqrt()
        # 实例归一化
        instance_normalized = (x - mean) * var_rsqrt
        # 重新缩放
        rescaled = instance_normalized * self.gamma + self.beta

        # 定义反向函数
        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min = self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        if not return_statistics:
            return rescaled, reverse_fn

        # 返回统计信息
        statistics = Statistics(mean, var, self.gamma, self.beta)

        return rescaled, reverse_fn, statistics

# 主函数，用于进行简单的测试
if __name__ == '__main__':

    # 创建 RevIN 实例
    rev_in = RevIN(512)

    # 生成随机输入
    x = torch.randn(2, 512, 1024)

    # 进行实例归一化并返回统计信息
    normalized, reverse_fn, statistics = rev_in(x, return_statistics = True)

    # 反向操作
    out = reverse_fn(normalized)

    # 断言输入和输出是否一致
    assert torch.allclose(x, out)
```