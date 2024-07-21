# `.\pytorch\torch\distributions\fishersnedecor.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
from numbers import Number

import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import broadcast_all

# 定义本文件中可以导出的类名
__all__ = ["FisherSnedecor"]


# 定义 FisherSnedecor 类，继承自 Distribution 类
class FisherSnedecor(Distribution):
    r"""
    Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
        tensor([ 0.2453])

    Args:
        df1 (float or Tensor): degrees of freedom parameter 1
        df2 (float or Tensor): degrees of freedom parameter 2
    """
    # 参数约束，df1 和 df2 必须为正数
    arg_constraints = {"df1": constraints.positive, "df2": constraints.positive}
    # 支持值的约束，必须为正数
    support = constraints.positive
    # 具有 rsample 方法
    has_rsample = True

    # 初始化方法
    def __init__(self, df1, df2, validate_args=None):
        # 广播 df1 和 df2，确保维度一致性
        self.df1, self.df2 = broadcast_all(df1, df2)
        # 创建 Gamma 分布对象 _gamma1 和 _gamma2
        self._gamma1 = Gamma(self.df1 * 0.5, self.df1)
        self._gamma2 = Gamma(self.df2 * 0.5, self.df2)

        # 根据输入类型判断 batch_shape
        if isinstance(df1, Number) and isinstance(df2, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.df1.size()
        # 调用父类 Distribution 的初始化方法
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展方法，返回新的 FisherSnedecor 分布对象
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(FisherSnedecor, _instance)
        batch_shape = torch.Size(batch_shape)
        # 扩展 df1 和 df2
        new.df1 = self.df1.expand(batch_shape)
        new.df2 = self.df2.expand(batch_shape)
        # 扩展 _gamma1 和 _gamma2
        new._gamma1 = self._gamma1.expand(batch_shape)
        new._gamma2 = self._gamma2.expand(batch_shape)
        # 调用父类 Distribution 的初始化方法
        super(FisherSnedecor, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 计算分布的均值
    @property
    def mean(self):
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 2] = nan  # 将 df2 小于等于 2 的位置置为 NaN
        return df2 / (df2 - 2)

    # 计算分布的众数
    @property
    def mode(self):
        mode = (self.df1 - 2) / self.df1 * self.df2 / (self.df2 + 2)
        mode[self.df1 <= 2] = nan  # 将 df1 小于等于 2 的位置置为 NaN
        return mode

    # 计算分布的方差
    @property
    def variance(self):
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 4] = nan  # 将 df2 小于等于 4 的位置置为 NaN
        return (
            2
            * df2.pow(2)
            * (self.df1 + df2 - 2)
            / (self.df1 * (df2 - 2).pow(2) * (df2 - 4))
        )
    def rsample(self, sample_shape=torch.Size(())):
        # 根据传入的采样形状，扩展成完整的形状信息
        shape = self._extended_shape(sample_shape)
        # 从 Gamma 分布中采样 X1 ~ Gamma(df1 / 2, 1 / df1)，X2 ~ Gamma(df2 / 2, 1 / df2)
        X1 = self._gamma1.rsample(sample_shape).view(shape)
        X2 = self._gamma2.rsample(sample_shape).view(shape)
        # 获取 X2 的数据类型的最小非零值，并将 X2 的值限制在这个最小非零值以上
        tiny = torch.finfo(X2.dtype).tiny
        X2.clamp_(min=tiny)
        # 计算 Y = X1 / X2，这里是 F 分布的采样结果
        Y = X1 / X2
        Y.clamp_(min=tiny)
        return Y

    def log_prob(self, value):
        # 如果启用参数验证，则验证输入的采样值
        if self._validate_args:
            self._validate_sample(value)
        # 计算常数部分
        ct1 = self.df1 * 0.5
        ct2 = self.df2 * 0.5
        ct3 = self.df1 / self.df2
        # 计算 t1，是 F 分布对数概率密度函数的一部分
        t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma()
        # 计算 t2，是 F 分布对数概率密度函数的一部分
        t2 = ct1 * ct3.log() + (ct1 - 1) * torch.log(value)
        # 计算 t3，是 F 分布对数概率密度函数的一部分
        t3 = (ct1 + ct2) * torch.log1p(ct3 * value)
        # 返回 F 分布对数概率密度函数的值
        return t1 + t2 - t3
```