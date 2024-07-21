# `.\pytorch\torch\distributions\normal.py`

```
# mypy: allow-untyped-defs
# 引入 math 模块，用于数学计算
import math
# 从 numbers 模块中引入 Number 和 Real 类型，用于约束参数类型
from numbers import Number, Real

# 引入 torch 库
import torch
# 从 torch.distributions 模块中引入 constraints
from torch.distributions import constraints
# 从 torch.distributions.exp_family 模块中引入 ExponentialFamily 类
from torch.distributions.exp_family import ExponentialFamily
# 从 torch.distributions.utils 模块中引入 _standard_normal 和 broadcast_all 函数
from torch.distributions.utils import _standard_normal, broadcast_all

# 设置仅导出 Normal 类
__all__ = ["Normal"]


# 定义 Normal 类，继承自 ExponentialFamily
class Normal(ExponentialFamily):
    r"""
    创建一个正态（也称为高斯）分布，由 :attr:`loc` 和 :attr:`scale` 参数化。

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): 分布的均值（通常称为 mu）
        scale (float or Tensor): 分布的标准差（通常称为 sigma）
    """
    
    # 参数约束，loc 必须是实数，scale 必须是正数
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # 分布的支持集合为实数集合
    support = constraints.real
    # 指示可以使用 rsample 方法进行采样
    has_rsample = True
    # 均值载体测度设置为 0
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    # 构造函数，接受 loc 和 scale 参数，并将它们广播成相同的形状
    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        # 如果 loc 和 scale 都是数字，则批处理形状为 torch.Size()
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        # 调用父类的构造函数
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展分布的批处理形状
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 从分布中抽样，可选参数 sample_shape 指定抽样形状
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    # 使用 reparameterization trick 从分布中抽样，可选参数 sample_shape 指定抽样形状
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    # 计算给定值的对数概率密度函数值
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 计算方差
        var = self.scale**2
        # 计算 log(scale)
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        # 计算对数概率密度
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
    # 计算累积分布函数（CDF），用于给定值的概率分布函数值
    def cdf(self, value):
        # 如果开启了参数验证，则验证样本值的有效性
        if self._validate_args:
            self._validate_sample(value)
        # 返回标准正态分布的累积分布函数值
        return 0.5 * (
            1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))
        )

    # 计算反函数的累积分布函数（CDF），即逆变换函数
    def icdf(self, value):
        # 返回对应于给定累积分布函数值的原始分布的值
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    # 计算分布的熵
    def entropy(self):
        # 返回分布的熵的值
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    # 自然参数的属性，用于该分布的参数表示
    @property
    def _natural_params(self):
        # 返回自然参数元组，用于分布参数的计算
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    # 计算对数归一化函数，用于该分布的归一化
    def _log_normalizer(self, x, y):
        # 返回对数归一化函数的值
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
```