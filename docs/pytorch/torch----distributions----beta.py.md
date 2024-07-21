# `.\pytorch\torch\distributions\beta.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

# 导出的类名列表，仅包含 Beta 类
__all__ = ["Beta"]

# Beta 类继承自 ExponentialFamily 类
class Beta(ExponentialFamily):
    r"""
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
        tensor([ 0.1046])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    # 参数约束字典，指定了参数的约束条件
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    # 分布的支持范围，单位区间
    support = constraints.unit_interval
    # 具有 rsample 方法，支持从分布中采样
    has_rsample = True

    # 初始化方法，接受两个浮点数或张量作为参数
    def __init__(self, concentration1, concentration0, validate_args=None):
        # 如果 concentration1 和 concentration0 都是实数，则转换为张量
        if isinstance(concentration1, Real) and isinstance(concentration0, Real):
            concentration1_concentration0 = torch.tensor(
                [float(concentration1), float(concentration0)]
            )
        else:
            # 否则，对 concentration1 和 concentration0 进行广播
            concentration1, concentration0 = broadcast_all(
                concentration1, concentration0
            )
            # 将广播后的张量按最后一个维度堆叠
            concentration1_concentration0 = torch.stack(
                [concentration1, concentration0], -1
            )
        # 创建一个 Dirichlet 分布对象
        self._dirichlet = Dirichlet(
            concentration1_concentration0, validate_args=validate_args
        )
        # 调用父类 ExponentialFamily 的初始化方法
        super().__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    # 扩展方法，允许改变批次形状
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        # 扩展 Dirichlet 分布对象
        new._dirichlet = self._dirichlet.expand(batch_shape)
        # 调用父类 Beta 的初始化方法
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 返回分布的均值
    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    # 返回分布的众数
    @property
    def mode(self):
        return self._dirichlet.mode[..., 0]

    # 返回分布的方差
    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))

    # 从分布中采样，返回采样值的第一个维度
    def rsample(self, sample_shape=()):
        return self._dirichlet.rsample(sample_shape).select(-1, 0)

    # 返回给定值的对数概率密度函数值
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 将给定的值和其补值（1-value）堆叠在最后一个维度
        heads_tails = torch.stack([value, 1.0 - value], -1)
        # 调用 Dirichlet 分布的对数概率密度函数
        return self._dirichlet.log_prob(heads_tails)
    # 计算该分布对象的熵
    def entropy(self):
        return self._dirichlet.entropy()

    # 属性方法，返回分布的第一个浓度参数
    @property
    def concentration1(self):
        # 获取第一个浓度参数的值
        result = self._dirichlet.concentration[..., 0]
        # 如果结果是一个数字，则将其封装成张量返回
        if isinstance(result, Number):
            return torch.tensor([result])
        else:
            return result

    # 属性方法，返回分布的第二个浓度参数
    @property
    def concentration0(self):
        # 获取第二个浓度参数的值
        result = self._dirichlet.concentration[..., 1]
        # 如果结果是一个数字，则将其封装成张量返回
        if isinstance(result, Number):
            return torch.tensor([result])
        else:
            return result

    # 私有属性方法，返回自然参数元组，包含第一个和第二个浓度参数
    @property
    def _natural_params(self):
        return (self.concentration1, self.concentration0)

    # 计算对数归一化常数，用于该分布的计算
    def _log_normalizer(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
```