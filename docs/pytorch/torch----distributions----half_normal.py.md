# `.\pytorch\torch\distributions\half_normal.py`

```
# mypy: allow-untyped-defs
# 引入 math 模块，用于数学计算
import math

# 引入 torch 库
import torch
# 从 torch 库中引入 inf 常量
from torch import inf
# 从 torch.distributions 模块中引入 constraints 类
from torch.distributions import constraints
# 从 torch.distributions.normal 模块中引入 Normal 类
from torch.distributions.normal import Normal
# 从 torch.distributions.transformed_distribution 模块中引入 TransformedDistribution 类
from torch.distributions.transformed_distribution import TransformedDistribution
# 从 torch.distributions.transforms 模块中引入 AbsTransform 类
from torch.distributions.transforms import AbsTransform

# 定义模块中公开的类列表
__all__ = ["HalfNormal"]


class HalfNormal(TransformedDistribution):
    r"""
    创建一个半正态分布，由 `scale` 参数化，其中::

        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfNormal(torch.tensor([1.0]))
        >>> m.sample()  # 半正态分布，scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): 完整正态分布的标准差
    """
    # 参数约束：scale 必须为正数
    arg_constraints = {"scale": constraints.positive}
    # 支持约束：Y 值必须为非负数
    support = constraints.nonnegative
    # 指示是否支持 rsample 方法
    has_rsample = True

    def __init__(self, scale, validate_args=None):
        # 创建一个以标准正态分布为基础的分布对象
        base_dist = Normal(0, scale, validate_args=False)
        # 使用绝对值变换 AbsTransform 封装基础分布
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # 扩展分布对象的形状
        new = self._get_checked_instance(HalfNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self):
        # 返回基础分布的标准差
        return self.base_dist.scale

    @property
    def mean(self):
        # 返回半正态分布的均值
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def mode(self):
        # 返回半正态分布的众数
        return torch.zeros_like(self.scale)

    @property
    def variance(self):
        # 返回半正态分布的方差
        return self.scale.pow(2) * (1 - 2 / math.pi)

    def log_prob(self, value):
        # 若开启参数验证，验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 计算对数概率：基础分布的对数概率 + log(2)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        # 若值大于等于 0，返回计算的对数概率；否则返回负无穷
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value):
        # 若开启参数验证，验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 返回累积分布函数值：2 * 基础分布的累积分布函数值 - 1
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        # 返回反函数值：基础分布的反函数值 ((prob + 1) / 2)
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        # 返回熵值：基础分布的熵值 - log(2)
        return self.base_dist.entropy() - math.log(2)
```