# `.\pytorch\torch\distributions\independent.py`

```py
# mypy: allow-untyped-defs
# 导入需要的类型字典
from typing import Dict

# 导入 PyTorch 库
import torch
# 从 torch.distributions 模块中导入 constraints
from torch.distributions import constraints
# 从 torch.distributions.distribution 模块中导入 Distribution 类
from torch.distributions.distribution import Distribution
# 从 torch.distributions.utils 模块中导入 _sum_rightmost 函数
from torch.distributions.utils import _sum_rightmost

# 定义在此模块中公开的类名列表
__all__ = ["Independent"]

# 定义一个新的分布类 Independent，继承自 Distribution 类
class Independent(Distribution):
    """
    重新解释分布的一些批次维度作为事件维度。

    这主要用于改变:meth:`log_prob`方法结果的形状。例如，为了创建一个对角正态分布，
    其形状与多元正态分布相同（因此它们可以互换），可以这样做::

        >>> from torch.distributions.multivariate_normal import MultivariateNormal
        >>> from torch.distributions.normal import Normal
        >>> loc = torch.zeros(3)
        >>> scale = torch.ones(3)
        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        >>> [mvn.batch_shape, mvn.event_shape]
        [torch.Size([]), torch.Size([3])]
        >>> normal = Normal(loc, scale)
        >>> [normal.batch_shape, normal.event_shape]
        [torch.Size([3]), torch.Size([])]
        >>> diagn = Independent(normal, 1)
        >>> [diagn.batch_shape, diagn.event_shape]
        [torch.Size([]), torch.Size([3])]

    Args:
        base_distribution (torch.distributions.distribution.Distribution): 基础分布
        reinterpreted_batch_ndims (int): 要重新解释为事件维度的批次维度的数量
    """

    # 参数约束字典
    arg_constraints: Dict[str, constraints.Constraint] = {}

    # 初始化方法，接受基础分布和要重新解释的批次维度
    def __init__(
        self, base_distribution, reinterpreted_batch_ndims, validate_args=None
    ):
        # 如果要重新解释的批次维度大于基础分布的批次形状长度，抛出值错误异常
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                f"actual {reinterpreted_batch_ndims} vs {len(base_distribution.batch_shape)}"
            )
        # 计算新的形状
        shape = base_distribution.batch_shape + base_distribution.event_shape
        event_dim = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        # 设置基础分布和重新解释的批次维度
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        # 调用父类的初始化方法
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    # 扩展方法，用于创建扩展的 Independent 分布实例
    def expand(self, batch_shape, _instance=None):
        # 获取一个经过检查的新实例
        new = self._get_checked_instance(Independent, _instance)
        batch_shape = torch.Size(batch_shape)
        # 扩展基础分布的批次形状
        new.base_dist = self.base_dist.expand(
            batch_shape + self.event_shape[: self.reinterpreted_batch_ndims]
        )
        new.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        # 调用父类的初始化方法，初始化新实例
        super(Independent, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @property
    # 检查基础分布是否具有 rsample 方法，并返回结果
    def has_rsample(self):
        return self.base_dist.has_rsample

    # 检查基础分布是否支持枚举操作，并根据重新解释的批次维度返回结果
    @property
    def has_enumerate_support(self):
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    # 返回基础分布的支持集合，并根据重新解释的批次维度进行独立处理
    @constraints.dependent_property
    def support(self):
        result = self.base_dist.support
        if self.reinterpreted_batch_ndims:
            result = constraints.independent(result, self.reinterpreted_batch_ndims)
        return result

    # 返回基础分布的均值
    @property
    def mean(self):
        return self.base_dist.mean

    # 返回基础分布的众数（最可能值）
    @property
    def mode(self):
        return self.base_dist.mode

    # 返回基础分布的方差
    @property
    def variance(self):
        return self.base_dist.variance

    # 从基础分布中生成样本，并按指定的样本形状返回
    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    # 从基础分布中生成可微样本，并按指定的样本形状返回
    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    # 计算基础分布对给定值的对数概率，并根据重新解释的批次维度求和
    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    # 计算基础分布的熵，并根据重新解释的批次维度求和
    def entropy(self):
        entropy = self.base_dist.entropy()
        return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)

    # 枚举支持分布的可能值，如果存在重新解释的批次维度，则引发 NotImplementedError
    def enumerate_support(self, expand=True):
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError(
                "Enumeration over cartesian product is not implemented"
            )
        return self.base_dist.enumerate_support(expand=expand)

    # 返回对象的字符串表示，包括基础分布和重新解释的批次维度
    def __repr__(self):
        return (
            self.__class__.__name__
            + f"({self.base_dist}, {self.reinterpreted_batch_ndims})"
        )
```