# `.\pytorch\torch\distributions\binomial.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 库
import torch
# 从 PyTorch 分布模块导入约束
from torch.distributions import constraints
# 从 PyTorch 分布模块导入 Distribution 类
from torch.distributions.distribution import Distribution
# 从 PyTorch 分布工具模块导入若干函数
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

# 声明此模块对外可见的类为 Binomial
__all__ = ["Binomial"]


# 定义一个函数 _clamp_by_zero，用于将输入张量的负值置为零并保留梯度
def _clamp_by_zero(x):
    # works like clamp(x, min=0) but has grad at 0 is 0.5
    return (x.clamp(min=0) + x - x.clamp(max=0)) / 2


# 定义一个 Binomial 类，继承自 Distribution 类
class Binomial(Distribution):
    r"""
    创建一个二项分布，由参数 :attr:`total_count` 和 :attr:`probs` 或 :attr:`logits`
    （但不能同时有两者）来参数化。:attr:`total_count` 必须与 :attr:`probs`/:attr:`logits` 广播兼容。

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])

        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])

    Args:
        total_count (int or Tensor): 伯努利试验的次数
        probs (Tensor): 事件概率
        logits (Tensor): 事件的对数几率
    """
    
    # 参数约束字典，指定各参数的约束条件
    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
    }
    
    # 表明此分布支持 enumerate 方法
    has_enumerate_support = True

    # 构造方法，初始化二项分布对象
    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        # 如果 probs 和 logits 同时为 None 或同时不为 None，则引发 ValueError
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            # 广播 total_count 和 probs，并将 total_count 转换为和 probs 相同的类型
            (
                self.total_count,
                self.probs,
            ) = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            # 广播 total_count 和 logits，并将 total_count 转换为和 logits 相同的类型
            (
                self.total_count,
                self.logits,
            ) = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)

        # 选择参数为 probs 或 logits 的一个作为内部参数
        self._param = self.probs if probs is not None else self.logits
        # 获取批次形状
        batch_shape = self._param.size()
        # 调用父类的构造方法进行初始化
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展方法，返回一个新的分布对象，扩展后的批次形状
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Binomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Binomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 私有方法，创建一个新的张量，使用与 self._param 相同的类型
    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)
    # 定义一个依赖属性，表示支持的离散值范围为 [0, total_count]
    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    # 计算分布的均值，即总数乘以概率
    @property
    def mean(self):
        return self.total_count * self.probs

    # 计算分布的众数，使用公式 ((total_count + 1) * probs).floor().clamp(max=total_count)
    @property
    def mode(self):
        return ((self.total_count + 1) * self.probs).floor().clamp(max=self.total_count)

    # 计算分布的方差，使用公式 total_count * probs * (1 - probs)
    @property
    def variance(self):
        return self.total_count * self.probs * (1 - self.probs)

    # 延迟加载属性，将概率转换为 logits （对数几率），用于二进制情况
    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    # 延迟加载属性，将 logits （对数几率）转换为概率，用于二进制情况
    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    # 返回参数的形状
    @property
    def param_shape(self):
        return self._param.size()

    # 生成样本，返回符合二项分布的随机数
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.binomial(
                self.total_count.expand(shape), self.probs.expand(shape)
            )

    # 计算给定值的对数概率
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 计算阶乘的对数
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        # 计算归一化项，确保 logit 不超出范围
        normalize_term = (
            self.total_count * _clamp_by_zero(self.logits)
            + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
            - log_factorial_n
        )
        return (
            value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
        )

    # 计算分布的熵
    def entropy(self):
        # 获取总数的整数最大值
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `entropy`."
            )

        # 计算对数概率和的负值，用于计算熵
        log_prob = self.log_prob(self.enumerate_support(False))
        return -(torch.exp(log_prob) * log_prob).sum(0)
    # 定义一个方法 `enumerate_support`，接受一个布尔参数 `expand` 默认为 True
    def enumerate_support(self, expand=True):
        # 计算 `total_count`，即 `self.total_count` 的最大值并转为整数
        total_count = int(self.total_count.max())
        # 如果 `self.total_count` 的最小值不等于 `total_count`，抛出未实现的错误
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `enumerate_support`."
            )
        # 创建一个序列 `values`，从 1 到 `total_count + 1`，使用 `self._param` 的数据类型和设备
        values = torch.arange(
            1 + total_count, dtype=self._param.dtype, device=self._param.device
        )
        # 将 `values` 转换成形状为 `(-1, 1, ..., 1)` 的张量，其中 `...` 表示与 `self._batch_shape` 相同数量的维度
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        # 如果 `expand` 参数为 True，则将 `values` 沿着 `self._batch_shape` 扩展
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        # 返回扩展后的 `values` 张量
        return values
```