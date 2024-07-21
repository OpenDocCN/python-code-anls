# `.\pytorch\torch\distributions\cauchy.py`

```py
# mypy: allow-untyped-defs
# 导入 math 和 numbers 模块
import math
from numbers import Number

# 导入 torch 库及其组件
import torch
from torch import inf, nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

# 定义模块导出的内容
__all__ = ["Cauchy"]

# 定义 Cauchy 类，继承自 Distribution 类
class Cauchy(Distribution):
    """
    从柯西（洛伦兹）分布中抽样。独立正态分布随机变量的比值的分布遵循柯西分布。

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        tensor([ 2.3214])

    Args:
        loc (float or Tensor): 分布的模式或中位数。
        scale (float or Tensor): 半宽全峰的一半。
    """
    # 参数约束
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # 支持范围约束
    support = constraints.real
    # 是否具有 rsample 方法
    has_rsample = True

    # 初始化方法
    def __init__(self, loc, scale, validate_args=None):
        # 广播 loc 和 scale 到相同形状
        self.loc, self.scale = broadcast_all(loc, scale)
        # 根据 loc 和 scale 的类型判断批次形状
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展分布到新的批次形状
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Cauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Cauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 均值属性
    @property
    def mean(self):
        return torch.full(
            self._extended_shape(), nan, dtype=self.loc.dtype, device=self.loc.device
        )

    # 众数属性
    @property
    def mode(self):
        return self.loc

    # 方差属性
    @property
    def variance(self):
        return torch.full(
            self._extended_shape(), inf, dtype=self.loc.dtype, device=self.loc.device
        )

    # 从分布中抽样的方法
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).cauchy_()
        return self.loc + eps * self.scale

    # 对数概率密度函数
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            -math.log(math.pi)
            - self.scale.log()
            - (((value - self.loc) / self.scale) ** 2).log1p()
        )

    # 累积分布函数
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.atan((value - self.loc) / self.scale) / math.pi + 0.5

    # 逆累积分布函数
    def icdf(self, value):
        return torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc

    # 熵
    def entropy(self):
        return math.log(4 * math.pi) + self.scale.log()
```