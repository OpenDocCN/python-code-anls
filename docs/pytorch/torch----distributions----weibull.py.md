# `.\pytorch\torch\distributions\weibull.py`

```
# mypy: allow-untyped-defs
# 引入 torch 库
import torch
# 从 torch.distributions 模块中引入 constraints 子模块
from torch.distributions import constraints
# 从 torch.distributions.exponential 模块引入 Exponential 类
from torch.distributions.exponential import Exponential
# 从 torch.distributions.gumbel 模块引入 euler_constant 常数
from torch.distributions.gumbel import euler_constant
# 从 torch.distributions.transformed_distribution 模块引入 TransformedDistribution 类
from torch.distributions.transformed_distribution import TransformedDistribution
# 从 torch.distributions.transforms 模块引入 AffineTransform 和 PowerTransform 类
from torch.distributions.transforms import AffineTransform, PowerTransform
# 从 torch.distributions.utils 模块引入 broadcast_all 函数
from torch.distributions.utils import broadcast_all

# 定义该模块的导出列表
__all__ = ["Weibull"]

# 定义 Weibull 类，继承自 TransformedDistribution 类
class Weibull(TransformedDistribution):
    r"""
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
    """
    # 定义参数的约束条件字典
    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    # 定义支持的约束条件
    support = constraints.positive

    # 初始化方法
    def __init__(self, scale, concentration, validate_args=None):
        # 使用 broadcast_all 函数确保 scale 和 concentration 可以广播
        self.scale, self.concentration = broadcast_all(scale, concentration)
        # 计算 concentration 的倒数
        self.concentration_reciprocal = self.concentration.reciprocal()
        # 创建指数分布作为基础分布，参数为 torch.ones_like(self.scale)
        base_dist = Exponential(
            torch.ones_like(self.scale), validate_args=validate_args
        )
        # 定义变换列表
        transforms = [
            PowerTransform(exponent=self.concentration_reciprocal),
            AffineTransform(loc=0, scale=self.scale),
        ]
        # 调用父类 TransformedDistribution 的初始化方法
        super().__init__(base_dist, transforms, validate_args=validate_args)

    # 扩展方法，支持扩展形状
    def expand(self, batch_shape, _instance=None):
        # 获取检查后的新实例
        new = self._get_checked_instance(Weibull, _instance)
        # 扩展 scale 和 concentration 到指定的 batch_shape
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        # 更新 concentration 的倒数
        new.concentration_reciprocal = new.concentration.reciprocal()
        # 扩展基础分布
        base_dist = self.base_dist.expand(batch_shape)
        # 更新变换列表
        transforms = [
            PowerTransform(exponent=new.concentration_reciprocal),
            AffineTransform(loc=0, scale=new.scale),
        ]
        # 调用父类 Weibull 的初始化方法
        super(Weibull, new).__init__(base_dist, transforms, validate_args=False)
        # 继承验证参数标志
        new._validate_args = self._validate_args
        return new

    # 平均值的属性方法
    @property
    def mean(self):
        return self.scale * torch.exp(torch.lgamma(1 + self.concentration_reciprocal))

    # 众数的属性方法
    @property
    def mode(self):
        return (
            self.scale
            * ((self.concentration - 1) / self.concentration)
            ** self.concentration.reciprocal()
        )

    # 方差的属性方法
    @property
    def variance(self):
        return self.scale.pow(2) * (
            torch.exp(torch.lgamma(1 + 2 * self.concentration_reciprocal))
            - torch.exp(2 * torch.lgamma(1 + self.concentration_reciprocal))
        )
    # 计算熵值的方法
    def entropy(self):
        # 计算熵值的公式：欧拉常数乘以 (1 - 浓度的倒数)
        return (
            euler_constant * (1 - self.concentration_reciprocal)
            # 加上以自然对数为底的 self.scale 乘以 浓度的倒数的对数
            + torch.log(self.scale * self.concentration_reciprocal)
            # 加上常数 1
            + 1
        )
```