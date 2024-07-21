# `.\pytorch\torch\distributions\exponential.py`

```py
# mypy: allow-untyped-defs
# 从 numbers 模块导入 Number 类型
from numbers import Number

# 导入 torch 库
import torch
# 导入 distributions 模块中的 constraints 类
from torch.distributions import constraints
# 从 exp_family 模块导入 ExponentialFamily 类
from torch.distributions.exp_family import ExponentialFamily
# 导入 utils 模块中的 broadcast_all 函数
from torch.distributions.utils import broadcast_all

# 声明该模块导出的符号列表
__all__ = ["Exponential"]

# 定义 Exponential 类，继承自 ExponentialFamily 类
class Exponential(ExponentialFamily):
    r"""
    创建由 rate 参数化的指数分布。

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): 分布的 rate = 1 / scale
    """
    # 参数约束，rate 必须为正数
    arg_constraints = {"rate": constraints.positive}
    # 分布的支持范围为非负数
    support = constraints.nonnegative
    # 表示支持 rsample 方法
    has_rsample = True
    # 均值的载体测度为 0
    _mean_carrier_measure = 0

    @property
    def mean(self):
        # 返回分布的均值，即 rate 的倒数
        return self.rate.reciprocal()

    @property
    def mode(self):
        # 返回分布的众数，即与 rate 形状相同的零张量
        return torch.zeros_like(self.rate)

    @property
    def stddev(self):
        # 返回分布的标准差，即 rate 的倒数
        return self.rate.reciprocal()

    @property
    def variance(self):
        # 返回分布的方差，即 rate 的平方的倒数
        return self.rate.pow(-2)

    def __init__(self, rate, validate_args=None):
        # 使用 broadcast_all 函数将 rate 广播到相同形状
        (self.rate,) = broadcast_all(rate)
        # 如果 rate 是数字类型，batch_shape 为空，否则为 rate 的形状
        batch_shape = torch.Size() if isinstance(rate, Number) else self.rate.size()
        # 调用父类的构造函数，初始化分布的批量形状
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # 获取新实例，并进行类型检查
        new = self._get_checked_instance(Exponential, _instance)
        # 将 batch_shape 转换为 torch.Size 类型
        batch_shape = torch.Size(batch_shape)
        # 扩展 rate 到新的 batch_shape
        new.rate = self.rate.expand(batch_shape)
        # 调用父类的构造函数，初始化新实例的批量形状
        super(Exponential, new).__init__(batch_shape, validate_args=False)
        # 继承 validate_args 属性
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # 计算采样形状的扩展形状
        shape = self._extended_shape(sample_shape)
        # 生成从指数分布采样的值，并根据 rate 进行缩放
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        # 如果启用参数验证，则验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 返回给定值的对数概率密度函数值
        return self.rate.log() - self.rate * value

    def cdf(self, value):
        # 如果启用参数验证，则验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 返回给定值的累积分布函数值
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        # 返回给定概率值对应的反函数值
        return -torch.log1p(-value) / self.rate

    def entropy(self):
        # 返回分布的熵值
        return 1.0 - torch.log(self.rate)

    @property
    def _natural_params(self):
        # 返回自然参数，此处为 -rate
        return (-self.rate,)

    def _log_normalizer(self, x):
        # 返回对数归一化常数，这里为 -log(-x)
        return -torch.log(-x)
```