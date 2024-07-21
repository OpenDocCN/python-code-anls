# `.\pytorch\torch\distributions\gamma.py`

```py
# 引入必要的模块和类
from numbers import Number  # 导入数字类型模块

import torch  # 导入 PyTorch 库
from torch.distributions import constraints  # 导入约束条件模块
from torch.distributions.exp_family import ExponentialFamily  # 导入指数分布族基类
from torch.distributions.utils import broadcast_all  # 导入广播函数

__all__ = ["Gamma"]  # 模块的公开接口，仅包括 Gamma 分布

def _standard_gamma(concentration):
    return torch._standard_gamma(concentration)

class Gamma(ExponentialFamily):
    r"""
    创建由形状参数 :attr:`concentration` 和 :attr:`rate` 参数化的 Gamma 分布。

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma 分布，形状参数 concentration=1，速率参数 rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): 分布的形状参数（通常称为 alpha）
        rate (float or Tensor): 速率参数，等于分布的倒数（通常称为 beta）
    """
    # 参数的约束条件
    arg_constraints = {
        "concentration": constraints.positive,  # concentration 必须为正数
        "rate": constraints.positive,  # rate 必须为正数
    }
    support = constraints.nonnegative  # 分布的支持集为非负数
    has_rsample = True  # 支持 rsample 方法
    _mean_carrier_measure = 0  # 均值关联度量为 0

    @property
    def mean(self):
        return self.concentration / self.rate  # 返回 Gamma 分布的均值

    @property
    def mode(self):
        return ((self.concentration - 1) / self.rate).clamp(min=0)  # 返回 Gamma 分布的众数，使用 clamp 函数确保不小于 0

    @property
    def variance(self):
        return self.concentration / self.rate.pow(2)  # 返回 Gamma 分布的方差

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)  # 广播输入的形状参数和速率参数
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = torch.Size()  # 如果参数是单个数字，则批量形状为空
        else:
            batch_shape = self.concentration.size()  # 否则，批量形状等于 concentration 的形状
        super().__init__(batch_shape, validate_args=validate_args)  # 调用父类构造函数初始化分布

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)  # 获取扩展后的新实例
        batch_shape = torch.Size(batch_shape)  # 转换为 Torch 的尺寸对象
        new.concentration = self.concentration.expand(batch_shape)  # 扩展 concentration 至新的批量形状
        new.rate = self.rate.expand(batch_shape)  # 扩展 rate 至新的批量形状
        super(Gamma, new).__init__(batch_shape, validate_args=False)  # 初始化新实例
        new._validate_args = self._validate_args  # 设置验证参数
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)  # 扩展样本形状
        value = _standard_gamma(self.concentration.expand(shape)) / self.rate.expand(
            shape
        )  # 使用标准 Gamma 函数生成样本值
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # 不记录在自动求导图中的最小值截断
        return value
    # 计算对数概率密度函数值
    def log_prob(self, value):
        # 将输入值转换为张量，使用与参数 self.rate 相同的数据类型和设备
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        # 如果启用参数验证，则验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 返回对数概率密度函数的计算结果
        return (
            torch.xlogy(self.concentration, self.rate)  # 第一项
            + torch.xlogy(self.concentration - 1, value)  # 第二项
            - self.rate * value  # 第三项
            - torch.lgamma(self.concentration)  # 第四项
        )

    # 计算熵值
    def entropy(self):
        # 返回熵值的计算结果
        return (
            self.concentration  # 第一项
            - torch.log(self.rate)  # 第二项
            + torch.lgamma(self.concentration)  # 第三项
            + (1.0 - self.concentration) * torch.digamma(self.concentration)  # 第四项
        )

    # 自然参数的属性方法
    @property
    def _natural_params(self):
        # 返回自然参数 (self.concentration - 1, -self.rate)
        return (self.concentration - 1, -self.rate)

    # 计算对数归一化函数
    def _log_normalizer(self, x, y):
        # 返回对数归一化函数的计算结果
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())

    # 计算累积分布函数
    def cdf(self, value):
        # 如果启用参数验证，则验证样本值
        if self._validate_args:
            self._validate_sample(value)
        # 返回累积分布函数的计算结果
        return torch.special.gammainc(self.concentration, self.rate * value)
```