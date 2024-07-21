# `.\pytorch\torch\distributions\continuous_bernoulli.py`

```
# mypy: allow-untyped-defs
# 引入数学库和数字模块
import math
from numbers import Number

# 引入PyTorch库及其约束条件、指数族分布和实用函数
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits

# 模块内公开的类只有 ContinuousBernoulli
__all__ = ["ContinuousBernoulli"]

# 定义 ContinuousBernoulli 类，继承自 ExponentialFamily 类
class ContinuousBernoulli(ExponentialFamily):
    r"""
    创建一个连续伯努利分布，其参数由 'probs' 或 'logits'（但不能同时存在）来确定。

    该分布在 [0, 1] 区间内，并由 'probs'（在 (0,1) 区间内）或 'logits'（实数值）参数化。
    注意，与伯努利分布不同的是，'probs' 不对应概率，'logits' 也不对应对数几率，
    但由于与伯努利分布的相似性，使用了相同的名称。详细信息请参见 [1]。

    Args:
        probs (Number, Tensor): (0,1) 区间内的参数
        logits (Number, Tensor): 实数值参数，其 sigmoid 函数匹配 'probs'

    [1] The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    https://arxiv.org/abs/1907.06845
    """

    # 参数约束：'probs' 在单位区间内，'logits' 是实数
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    # 分布的支持在单位区间内
    support = constraints.unit_interval
    # 均值承载度量为 0
    _mean_carrier_measure = 0
    # 具有 rsample 方法
    has_rsample = True

    def __init__(
        self, probs=None, logits=None, lims=(0.499, 0.501), validate_args=None
    ):
        # 确保 'probs' 和 'logits' 中只能指定一个，不能同时指定
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            # 广播 'probs' 参数
            (self.probs,) = broadcast_all(probs)
            # 如果需要，验证 'probs' 的参数是否有效，以便后续进行数值稳定性的修剪
            if validate_args is not None:
                if not self.arg_constraints["probs"].check(self.probs).all():
                    raise ValueError("The parameter probs has invalid values")
            # 对 'probs' 进行数值稳定性的修剪
            self.probs = clamp_probs(self.probs)
        else:
            is_scalar = isinstance(logits, Number)
            # 广播 'logits' 参数
            (self.logits,) = broadcast_all(logits)
        # 确定使用的参数是 'probs' 还是 'logits'
        self._param = self.probs if probs is not None else self.logits
        # 确定批次形状
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        # 设定极限值
        self._lims = lims
        # 调用父类的初始化方法
        super().__init__(batch_shape, validate_args=validate_args)
    # 根据给定的批量形状扩展当前对象，并返回一个新的实例
    def expand(self, batch_shape, _instance=None):
        # 获取一个已检查的 ContinuousBernoulli 实例，或者根据需要创建一个新的实例
        new = self._get_checked_instance(ContinuousBernoulli, _instance)
        # 将当前对象的限制值赋给新实例
        new._lims = self._lims
        # 将批量形状转换为 torch.Size 类型
        batch_shape = torch.Size(batch_shape)
        # 如果当前对象包含 'probs' 属性，则在新实例中扩展 'probs' 属性
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        # 如果当前对象包含 'logits' 属性，则在新实例中扩展 'logits' 属性
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        # 调用父类的构造函数初始化新实例，并关闭参数验证
        super(ContinuousBernoulli, new).__init__(batch_shape, validate_args=False)
        # 继承当前对象的参数验证设置到新实例
        new._validate_args = self._validate_args
        # 返回扩展后的新实例
        return new

    # 返回当前对象 _param 属性的新实例
    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    # 判断当前对象的 'probs' 是否处于不稳定区域之外
    def _outside_unstable_region(self):
        return torch.max(
            torch.le(self.probs, self._lims[0]), torch.gt(self.probs, self._lims[1])
        )

    # 根据当前对象的 '_outside_unstable_region' 方法调整 'probs' 的值
    def _cut_probs(self):
        return torch.where(
            self._outside_unstable_region(),
            self.probs,
            self._lims[0] * torch.ones_like(self.probs),
        )

    # 计算连续伯努利分布的对数归一化常数，基于 'probs' 参数
    def _cont_bern_log_norm(self):
        cut_probs = self._cut_probs()
        cut_probs_below_half = torch.where(
            torch.le(cut_probs, 0.5), cut_probs, torch.zeros_like(cut_probs)
        )
        cut_probs_above_half = torch.where(
            torch.ge(cut_probs, 0.5), cut_probs, torch.ones_like(cut_probs)
        )
        # 计算对数归一化常数
        log_norm = torch.log(
            torch.abs(torch.log1p(-cut_probs) - torch.log(cut_probs))
        ) - torch.where(
            torch.le(cut_probs, 0.5),
            torch.log1p(-2.0 * cut_probs_below_half),
            torch.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = torch.pow(self.probs - 0.5, 2)
        taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        # 根据不稳定区域判断选择返回哪种计算结果
        return torch.where(self._outside_unstable_region(), log_norm, taylor)

    # 计算当前对象的均值属性
    @property
    def mean(self):
        cut_probs = self._cut_probs()
        mus = cut_probs / (2.0 * cut_probs - 1.0) + 1.0 / (
            torch.log1p(-cut_probs) - torch.log(cut_probs)
        )
        x = self.probs - 0.5
        taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * torch.pow(x, 2)) * x
        # 根据不稳定区域判断选择返回哪种计算结果
        return torch.where(self._outside_unstable_region(), mus, taylor)

    # 计算当前对象的标准差属性
    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    # 计算当前对象的方差属性
    @property
    def variance(self):
        cut_probs = self._cut_probs()
        vars = cut_probs * (cut_probs - 1.0) / torch.pow(
            1.0 - 2.0 * cut_probs, 2
        ) + 1.0 / torch.pow(torch.log1p(-cut_probs) - torch.log(cut_probs), 2)
        x = torch.pow(self.probs - 0.5, 2)
        taylor = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        # 根据不稳定区域判断选择返回哪种计算结果
        return torch.where(self._outside_unstable_region(), vars, taylor)

    # 使用懒加载属性返回当前对象的 logits 属性
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)
    # 返回经过限制的概率值，将 logits 转换为概率，并确保是二进制情况
    def probs(self):
        return clamp_probs(logits_to_probs(self.logits, is_binary=True))

    # 返回参数的形状
    @property
    def param_shape(self):
        return self._param.size()

    # 对分布进行采样，可以指定采样的形状
    def sample(self, sample_shape=torch.Size()):
        # 计算采样的形状
        shape = self._extended_shape(sample_shape)
        # 生成均匀分布的随机数
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        with torch.no_grad():
            # 使用逆累积分布函数进行采样
            return self.icdf(u)

    # 对分布进行逆采样，可以指定采样的形状
    def rsample(self, sample_shape=torch.Size()):
        # 计算采样的形状
        shape = self._extended_shape(sample_shape)
        # 生成均匀分布的随机数
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        # 使用逆累积分布函数进行逆采样
        return self.icdf(u)

    # 计算给定值的对数概率密度函数
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 广播 logits 和 value，以便能够计算二元交叉熵
        logits, value = broadcast_all(self.logits, value)
        return (
            -binary_cross_entropy_with_logits(logits, value, reduction="none")
            + self._cont_bern_log_norm()
        )

    # 计算给定值的累积分布函数
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 计算切分概率
        cut_probs = self._cut_probs()
        # 计算累积分布函数值
        cdfs = (
            torch.pow(cut_probs, value) * torch.pow(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        # 处理不稳定区域外的值
        unbounded_cdfs = torch.where(self._outside_unstable_region(), cdfs, value)
        # 将值限制在 [0, 1] 区间内
        return torch.where(
            torch.le(value, 0.0),
            torch.zeros_like(value),
            torch.where(torch.ge(value, 1.0), torch.ones_like(value), unbounded_cdfs),
        )

    # 计算给定值的逆累积分布函数
    def icdf(self, value):
        # 计算切分概率
        cut_probs = self._cut_probs()
        return torch.where(
            self._outside_unstable_region(),
            (
                torch.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - torch.log1p(-cut_probs)
            )
            / (torch.log(cut_probs) - torch.log1p(-cut_probs)),
            value,
        )

    # 计算熵
    def entropy(self):
        # 计算对数概率
        log_probs0 = torch.log1p(-self.probs)
        log_probs1 = torch.log(self.probs)
        return (
            # 计算熵的公式
            self.mean * (log_probs0 - log_probs1)
            - self._cont_bern_log_norm()
            - log_probs0
        )

    # 返回自然参数的元组形式
    @property
    def _natural_params(self):
        return (self.logits,)

    # 计算对数正则化常数作为自然参数的函数
    def _log_normalizer(self, x):
        """根据自然参数计算对数正则化常数"""
        # 判断是否在不稳定区域外
        out_unst_reg = torch.max(
            torch.le(x, self._lims[0] - 0.5), torch.gt(x, self._lims[1] - 0.5)
        )
        # 根据不同情况选择自然参数
        cut_nat_params = torch.where(
            out_unst_reg, x, (self._lims[0] - 0.5) * torch.ones_like(x)
        )
        # 计算对数正则化常数
        log_norm = torch.log(torch.abs(torch.exp(cut_nat_params) - 1.0)) - torch.log(
            torch.abs(cut_nat_params)
        )
        # 利用泰勒展开进行计算
        taylor = 0.5 * x + torch.pow(x, 2) / 24.0 - torch.pow(x, 4) / 2880.0
        return torch.where(out_unst_reg, log_norm, taylor)
```