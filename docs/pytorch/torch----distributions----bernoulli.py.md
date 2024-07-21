# `.\pytorch\torch\distributions\bernoulli.py`

```py
# mypy: allow-untyped-defs
# 从 numbers 模块导入 Number 类型
from numbers import Number

# 导入 torch 库
import torch
# 从 torch 库中导入 nan 方法
from torch import nan
# 从 torch.distributions 模块中导入 constraints 类
from torch.distributions import constraints
# 从 torch.distributions.exp_family 模块中导入 ExponentialFamily 类
from torch.distributions.exp_family import ExponentialFamily
# 从 torch.distributions.utils 模块中导入若干函数
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
# 从 torch.nn.functional 模块中导入 binary_cross_entropy_with_logits 函数
from torch.nn.functional import binary_cross_entropy_with_logits

# 定义 __all__ 列表，表示模块中公开的全部符号
__all__ = ["Bernoulli"]

# 定义 Bernoulli 类，继承自 ExponentialFamily 类
class Bernoulli(ExponentialFamily):
    r"""
    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Bernoulli(torch.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """
    
    # 参数约束字典，定义了 probs 和 logits 的约束条件
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    # 支持集合，表示分布的支持集合为布尔值集合
    support = constraints.boolean
    # 表示分布是否具有枚举支持
    has_enumerate_support = True
    # 均值载体测度
    _mean_carrier_measure = 0

    # 构造函数，接受 probs 或 logits 参数，并验证参数合法性
    def __init__(self, probs=None, logits=None, validate_args=None):
        # 如果 probs 和 logits 同时为 None 或同时不为 None，抛出 ValueError 异常
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        # 如果指定了 probs 参数
        if probs is not None:
            # 检查 probs 是否是标量
            is_scalar = isinstance(probs, Number)
            # 使用 broadcast_all 方法，将 probs 广播成相同形状的张量
            (self.probs,) = broadcast_all(probs)
        else:
            # 检查 logits 是否是标量
            is_scalar = isinstance(logits, Number)
            # 使用 broadcast_all 方法，将 logits 广播成相同形状的张量
            (self.logits,) = broadcast_all(logits)
        # 选择参数为 probs 或 logits 中的一个
        self._param = self.probs if probs is not None else self.logits
        # 如果参数是标量，则批次形状为空尺寸的 torch.Size 对象；否则为参数形状
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        # 调用父类的构造函数进行初始化
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展方法，返回一个新的 Bernoulli 实例，扩展后的批次形状
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Bernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        # 如果当前实例中有 probs 属性，则将其扩展到指定的批次形状
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        # 如果当前实例中有 logits 属性，则将其扩展到指定的批次形状
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        # 调用父类的构造函数进行初始化
        super(Bernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 创建一个新的张量，使用当前参数的 new 方法
    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    # 返回分布的均值属性，即 probs 属性本身
    @property
    def mean(self):
        return self.probs

    # 返回分布的众数属性
    @property
    def mode(self):
        # 计算众数，大于等于 0.5 的 probs 对应 1，其余情况对应 nan
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = nan
        return mode

    # 返回分布的方差属性
    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    # 延迟计算的属性，返回 logits 属性
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    # 延迟计算的属性
    # 根据当前的逻辑张量(logits)，计算对应的概率值，假设二元情况
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)
    
    # 返回参数的形状信息
    @property
    def param_shape(self):
        return self._param.size()
    
    # 从分布中抽取样本
    def sample(self, sample_shape=torch.Size()):
        # 根据给定的样本形状扩展实际形状
        shape = self._extended_shape(sample_shape)
        # 使用伯努利分布生成样本
        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(shape))
    
    # 计算给定值的对数概率
    def log_prob(self, value):
        # 如果需要验证参数，则进行样本验证
        if self._validate_args:
            self._validate_sample(value)
        # 广播 logits 和 value，使其具有相同的形状
        logits, value = broadcast_all(self.logits, value)
        # 计算二元交叉熵损失，返回未进行汇总的每个样本的损失值
        return -binary_cross_entropy_with_logits(logits, value, reduction="none")
    
    # 计算分布的熵
    def entropy(self):
        return binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )
    
    # 枚举分布的所有支持值
    def enumerate_support(self, expand=True):
        # 创建包含所有可能值的张量 [0, 1]
        values = torch.arange(2, dtype=self._param.dtype, device=self._param.device)
        # 将值调整为与批次形状兼容的形状
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        # 如果需要扩展，则将值扩展到整个批次形状
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values
    
    # 返回自然参数 (logits 的对数几率)
    @property
    def _natural_params(self):
        return (torch.logit(self.probs),)
    
    # 计算对数归一化常数
    def _log_normalizer(self, x):
        return torch.log1p(torch.exp(x))
```