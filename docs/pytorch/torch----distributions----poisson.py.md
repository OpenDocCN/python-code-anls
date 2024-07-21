# `.\pytorch\torch\distributions\poisson.py`

```py
# 导入必要的模块和类
from numbers import Number  # 导入 Number 类型用于参数类型检查

import torch  # 导入 PyTorch 模块
from torch.distributions import constraints  # 导入约束条件模块
from torch.distributions.exp_family import ExponentialFamily  # 导入指数族分布类
from torch.distributions.utils import broadcast_all  # 导入广播函数

__all__ = ["Poisson"]  # 定义模块中可以导出的公共接口

class Poisson(ExponentialFamily):
    r"""
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """
    # 参数约束，确保 rate 非负
    arg_constraints = {"rate": constraints.nonnegative}
    # 分布的支持集，为非负整数
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.rate  # 返回分布的均值，即 rate 参数本身

    @property
    def mode(self):
        return self.rate.floor()  # 返回分布的众数，即 rate 参数向下取整

    @property
    def variance(self):
        return self.rate  # 返回分布的方差，即 rate 参数本身

    def __init__(self, rate, validate_args=None):
        (self.rate,) = broadcast_all(rate)  # 广播 rate 参数以匹配批次形状
        if isinstance(rate, Number):
            batch_shape = torch.Size()  # 如果 rate 是 Number 类型，批次形状为空
        else:
            batch_shape = self.rate.size()  # 否则，使用 rate 的形状作为批次形状
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Poisson, _instance)  # 获取新实例
        batch_shape = torch.Size(batch_shape)  # 转换批次形状为 torch.Size 对象
        new.rate = self.rate.expand(batch_shape)  # 扩展 rate 参数以匹配新的批次形状
        super(Poisson, new).__init__(batch_shape, validate_args=False)  # 调用父类初始化方法
        new._validate_args = self._validate_args  # 设置新实例的参数验证标志
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)  # 计算采样形状的扩展形状
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))  # 从 Poisson 分布中采样样本

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)  # 如果需要验证参数，则验证采样值
        rate, value = broadcast_all(self.rate, value)  # 广播 rate 和采样值
        return value.xlogy(rate) - rate - (value + 1).lgamma()  # 计算对数概率密度函数

    @property
    def _natural_params(self):
        return (torch.log(self.rate),)  # 返回自然参数，即 rate 的对数

    def _log_normalizer(self, x):
        return torch.exp(x)  # 计算对数归一化常数的指数
```