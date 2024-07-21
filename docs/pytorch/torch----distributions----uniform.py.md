# `.\pytorch\torch\distributions\uniform.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
from numbers import Number

import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

# 定义公开的类列表，表明只有 Uniform 类是公开的
__all__ = ["Uniform"]

# 定义 Uniform 类，继承自 Distribution 类
class Uniform(Distribution):
    r"""
    从半开区间 ``[low, high)`` 中生成均匀分布的随机样本。

    示例::

        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        >>> m.sample()  # 在范围 [0.0, 5.0) 中均匀分布
        >>> # xdoctest: +SKIP
        tensor([ 2.3418])

    Args:
        low (float or Tensor): 下限（包含）。
        high (float or Tensor): 上限（不包含）。
    """
    
    # 参数约束字典，规定了参数 low 和 high 的约束条件
    arg_constraints = {
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    
    # 表明该分布类具有 rsample 方法
    has_rsample = True

    # 返回均匀分布的均值
    @property
    def mean(self):
        return (self.high + self.low) / 2

    # 返回均匀分布的众数
    @property
    def mode(self):
        return nan * self.high

    # 返回均匀分布的标准差
    @property
    def stddev(self):
        return (self.high - self.low) / 12**0.5

    # 返回均匀分布的方差
    @property
    def variance(self):
        return (self.high - self.low).pow(2) / 12

    # 初始化方法，接受 low 和 high 作为参数，并进行广播
    def __init__(self, low, high, validate_args=None):
        self.low, self.high = broadcast_all(low, high)

        # 如果 low 和 high 都是数值型，则 batch_shape 为 torch.Size()
        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        
        # 调用父类 Distribution 的初始化方法
        super().__init__(batch_shape, validate_args=validate_args)

        # 如果开启参数验证且存在 low >= high，则抛出异常
        if self._validate_args and not torch.lt(self.low, self.high).all():
            raise ValueError("Uniform is not defined when low>= high")

    # 扩展方法，返回一个具有新的 batch_shape 的 Uniform 分布实例
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Uniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(Uniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 返回分布的支持域，即 [low, high) 的约束条件
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    # 从均匀分布中抽取样本，支持指定 sample_shape
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    # 计算给定值的对数概率密度函数值
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.gt(value).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)
    # 定义累积分布函数（CDF），接受一个值作为参数
    def cdf(self, value):
        # 如果设置了参数验证标志位，验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 计算累积分布函数的结果，公式为 (value - self.low) / (self.high - self.low)
        result = (value - self.low) / (self.high - self.low)
        # 对结果进行限制，确保其在 [0, 1] 的范围内
        return result.clamp(min=0, max=1)
    
    # 定义反向累积分布函数（ICDF），接受一个值作为参数
    def icdf(self, value):
        # 计算反向累积分布函数的结果，公式为 value * (self.high - self.low) + self.low
        result = value * (self.high - self.low) + self.low
        return result
    
    # 定义熵的计算方法
    def entropy(self):
        # 计算区间 [self.low, self.high) 的对数
        return torch.log(self.high - self.low)
```