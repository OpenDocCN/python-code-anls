# `.\pytorch\torch\distributions\inverse_gamma.py`

```py
# 引入 torch 库，包括相关的约束条件和分布类
import torch
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform

# 导出的类和方法列表，仅包含 InverseGamma 类
__all__ = ["InverseGamma"]

# 定义逆伽马分布类，继承自 TransformedDistribution 类
class InverseGamma(TransformedDistribution):
    r"""
    创建一个由 `concentration` 和 `rate` 参数化的逆伽马分布，其中::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = InverseGamma(torch.tensor([2.0]), torch.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): 分布的形状参数（通常称为 alpha）
        rate (float or Tensor): 率 = 1 / 分布的尺度参数（通常称为 beta）
    """
    
    # 参数的约束条件
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    
    # 分布的支持范围为正数
    support = constraints.positive
    
    # 指示该分布支持 rsample 方法（随机采样）
    has_rsample = True

    def __init__(self, concentration, rate, validate_args=None):
        # 基础分布为 Gamma 分布
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        # 定义变换为 PowerTransform(-1)，实现 Y = 1 / X
        neg_one = -base_dist.rate.new_ones(())
        super().__init__(
            base_dist, PowerTransform(neg_one), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        # 扩展分布到指定的 batch_shape
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self):
        # 返回分布的 concentration 参数
        return self.base_dist.concentration

    @property
    def rate(self):
        # 返回分布的 rate 参数
        return self.base_dist.rate

    @property
    def mean(self):
        # 计算分布的均值
        result = self.rate / (self.concentration - 1)
        return torch.where(self.concentration > 1, result, torch.inf)

    @property
    def mode(self):
        # 计算分布的众数
        return self.rate / (self.concentration + 1)

    @property
    def variance(self):
        # 计算分布的方差
        result = self.rate.square() / (
            (self.concentration - 1).square() * (self.concentration - 2)
        )
        return torch.where(self.concentration > 2, result, torch.inf)

    def entropy(self):
        # 计算分布的熵
        return (
            self.concentration
            + self.rate.log()
            + self.concentration.lgamma()
            - (1 + self.concentration) * self.concentration.digamma()
        )
```