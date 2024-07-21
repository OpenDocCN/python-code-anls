# `.\pytorch\torch\distributions\geometric.py`

```
# mypy: allow-untyped-defs
# 从 numbers 模块导入 Number 类
from numbers import Number

# 导入 torch 库
import torch
# 从 torch.distributions 模块导入 constraints 对象
from torch.distributions import constraints
# 从 torch.distributions.distribution 模块导入 Distribution 类
from torch.distributions.distribution import Distribution
# 从 torch.distributions.utils 模块导入一些函数和属性
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
# 从 torch.nn.functional 模块导入 binary_cross_entropy_with_logits 函数
from torch.nn.functional import binary_cross_entropy_with_logits

# 设置模块的公开接口
__all__ = ["Geometric"]

# 定义 Geometric 类，继承自 Distribution 类
class Geometric(Distribution):
    r"""
    创建一个几何分布，由 :attr:`probs` 参数化，
    其中 :attr:`probs` 是伯努利试验成功的概率。

    .. math::

        P(X=k) = (1-p)^{k} p, k = 0, 1, ...

    .. note::
        :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th 试验是第一次成功
        因此在 :math:`\{0, 1, \ldots\}` 中绘制样本，而
        :func:`torch.Tensor.geometric_` 的第 `k`-th 试验是第一次成功，因此在 :math:`\{1, 2, \ldots\}` 中绘制样本。

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Geometric(torch.tensor([0.3]))
        >>> m.sample()  # 底层伯努利有 30% 的概率为 1；70% 的概率为 0
        tensor([ 2.])

    Args:
        probs (Number, Tensor): 抽样 `1` 的概率。必须在 (0, 1] 范围内
        logits (Number, Tensor): 抽样 `1` 的 log-odds。
    """
    # 参数的约束条件
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    # 分布的支持集合
    support = constraints.nonnegative_integer

    # 初始化方法
    def __init__(self, probs=None, logits=None, validate_args=None):
        # 检查参数是否正确设置
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        # 如果使用 probs 参数，则广播它们
        if probs is not None:
            (self.probs,) = broadcast_all(probs)
        else:
            # 如果使用 logits 参数，则广播它们
            (self.logits,) = broadcast_all(logits)
        # 确定批次形状
        probs_or_logits = probs if probs is not None else logits
        if isinstance(probs_or_logits, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = probs_or_logits.size()
        # 调用父类的初始化方法
        super().__init__(batch_shape, validate_args=validate_args)
        # 如果启用参数验证，并且使用了 probs 参数，则进行额外的验证
        if self._validate_args and probs is not None:
            value = self.probs
            # 添加一个额外的检查条件，要求概率大于 0
            valid = value > 0
            if not valid.all():
                invalid_value = value.data[~valid]
                raise ValueError(
                    "Expected parameter probs "
                    f"({type(value).__name__} of shape {tuple(value.shape)}) "
                    f"of distribution {repr(self)} "
                    f"to be positive but found invalid values:\n{invalid_value}"
                )
    # 扩展当前分布对象以适应给定的批次形状，并返回一个新的分布对象
    def expand(self, batch_shape, _instance=None):
        # 获取一个经过检查的新实例，确保类型为 Geometric
        new = self._get_checked_instance(Geometric, _instance)
        # 将批次形状转换为 torch.Size 类型
        batch_shape = torch.Size(batch_shape)
        # 如果当前对象包含名为 "probs" 的属性，则将其扩展到新的批次形状
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
        # 如果当前对象包含名为 "logits" 的属性，则将其扩展到新的批次形状
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
        # 调用父类构造函数，初始化新对象的批次形状，并关闭参数验证
        super(Geometric, new).__init__(batch_shape, validate_args=False)
        # 继承当前对象的参数验证设置到新对象
        new._validate_args = self._validate_args
        # 返回新的分布对象
        return new

    @property
    # 计算几何分布的均值
    def mean(self):
        return 1.0 / self.probs - 1.0

    @property
    # 计算几何分布的众数
    def mode(self):
        return torch.zeros_like(self.probs)

    @property
    # 计算几何分布的方差
    def variance(self):
        return (1.0 / self.probs - 1.0) / self.probs

    @lazy_property
    # 惰性加载属性：将概率转换为 logits
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    # 惰性加载属性：将 logits 转换为概率
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    # 生成指定形状的样本
    def sample(self, sample_shape=torch.Size()):
        # 计算扩展后的形状
        shape = self._extended_shape(sample_shape)
        # 获取一个极小的浮点数，用于避免除零错误
        tiny = torch.finfo(self.probs.dtype).tiny
        with torch.no_grad():
            if torch._C._get_tracing_state():
                # [JIT WORKAROUND] 由于缺乏对 .uniform_() 的支持
                u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
                u = u.clamp(min=tiny)
            else:
                # 在指定形状上生成均匀分布的随机数，限制最小值为 tiny
                u = self.probs.new(shape).uniform_(tiny, 1)
            # 计算并返回几何分布的样本
            return (u.log() / (-self.probs).log1p()).floor()

    # 计算给定值的对数概率
    def log_prob(self, value):
        # 如果启用参数验证，则验证样本值是否合法
        if self._validate_args:
            self._validate_sample(value)
        # 广播输入值和概率，确保形状匹配
        value, probs = broadcast_all(value, self.probs)
        # 克隆概率张量，确保其内存布局为连续
        probs = probs.clone(memory_format=torch.contiguous_format)
        # 处理特殊情况：当概率为 1 且值为 0 时，将概率设为 0，避免无限大结果
        probs[(probs == 1) & (value == 0)] = 0
        # 计算并返回对数概率
        return value * (-probs).log1p() + self.probs.log()

    # 计算几何分布的熵
    def entropy(self):
        return (
            # 使用 logits 计算二分类交叉熵，并除以概率得到熵
            binary_cross_entropy_with_logits(self.logits, self.probs, reduction="none")
            / self.probs
        )
```